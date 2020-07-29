import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from tqdm import tqdm
from transformers import BertModel
from torch.utils.data import DataLoader, Subset
from utils import UPOS2IDX, UFEATS2IDX, PAD
import argparse
import os

from pooling import MaskedMeanPooler, WeightedSumPooler, LSTMPooler

# A bit of a hack so that `print` is used instead of `log` on kaggle since that doesn't seem to work properly there
is_kaggle = os.path.exists("/kaggle/input")
log_to_stdout = print if is_kaggle else logging.info

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
log_to_stdout(f"Using device {DEVICE}")

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="preprocessed/train.csv")
parser.add_argument("--dev_path", type=str, default="preprocessed/val.csv")

parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--pretrained_model_name_or_path", type=str, default="bert-base-multilingual-uncased")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=5_000)
parser.add_argument("--max_seq_len", type=int, default=192)

parser.add_argument("--include_upostag", action="store_true")
parser.add_argument("--upostag_emb_size", type=int, default=50)
parser.add_argument("--include_ufeats", action="store_true")
parser.add_argument("--ufeats_emb_size", type=int, default=15)
parser.add_argument("--pooling_type", type=str, default="mean")


class MorphologicalBertForSequenceClassification(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path, dropout=0.2,
                 additional_features=None, pooling_type=None):
        super().__init__()

        self.num_labels = num_labels
        self.dropout = dropout
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.additional_features = additional_features if additional_features is not None else {}
        self.pooling_type = pooling_type
        if len(self.additional_features) == 0:
            self.pooling_type = "N/A"

        # classic: linear(dropout(BERT(input_data)))
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_name_or_path,
                                                    num_labels=num_labels).to(DEVICE)

        self.classifier = nn.Linear(self.bert_model.config.hidden_size + sum(self.additional_features.values()),
                                    out_features=self.num_labels).to(DEVICE)
        self.embedders, self.poolers = nn.ModuleDict(), nn.ModuleDict()
        common_pooler = MaskedMeanPooler().to(DEVICE)
        for feature_name, emb_size in self.additional_features.items():
            log_to_stdout(f"Initializing embedding layer of size {emb_size} for features '{feature_name}'")
            num_embeddings = len(UPOS2IDX) if feature_name == "upostag" else len(UFEATS2IDX[feature_name])
            padding_idx = UPOS2IDX[PAD] if feature_name == "upostag" else UFEATS2IDX[feature_name][PAD]

            # data -> emb(data) -> mean across sequence
            self.embedders[feature_name] = nn.Embedding(num_embeddings=num_embeddings,
                                                        embedding_dim=emb_size,
                                                        padding_idx=padding_idx).to(DEVICE)
            if self.pooling_type == "lstm":
                log_to_stdout(f"Initializing LSTM pooler with hidden state size {emb_size}")
                self.poolers[feature_name] = LSTMPooler(hidden_size=emb_size).to(DEVICE)
            elif self.pooling_type == "weighted":
                log_to_stdout(f"Initializing weighted sum pooler")
                self.poolers[feature_name] = WeightedSumPooler(embedding_size=emb_size).to(DEVICE)
            else:
                log_to_stdout(f"Initializing mean pooler")
                self.pooling_type = "mean"
                self.poolers[feature_name] = common_pooler

    def config(self):
        return {
            "model_type": "bert",
            "num_labels": self.num_labels,
            "based_on": self.pretrained_model_name_or_path,
            "additional_features": self.additional_features,
            "pooling_type": self.pooling_type
        }

    @staticmethod
    def from_pretrained(model_dir):
        pass

    def save_pretrained(self, model_dir):
        pass

    def forward(self, input_ids, token_type_ids, attention_mask, **kwargs):
        # Classic BERT sequence classification settings: last layer's hidden state for [CLS] token: [B, hidden_size]
        _, pooled_output = self.bert_model(input_ids=input_ids.to(DEVICE),
                                           token_type_ids=token_type_ids.to(DEVICE),
                                           attention_mask=attention_mask.to(DEVICE))

        # Concatenate pooled POS tag / universal features embeddings for sequence to BERT sequence representation
        additional_processed = []
        for feature_name in self.additional_features:
            curr_input = kwargs[f"{feature_name}_ids"].to(DEVICE)
            curr_masks = kwargs[f"{feature_name}_mask"].to(DEVICE)
            curr_pooler = self.poolers[feature_name]
            curr_processed = curr_pooler(data=F.dropout(self.embedders[feature_name](curr_input), p=self.dropout),
                                         masks=curr_masks)
            additional_processed.append(curr_processed)

        if len(additional_processed) > 0:
            additional_processed = torch.cat(additional_processed, dim=1)
            pooled_output = torch.cat((pooled_output, additional_processed), dim=1)

        logits = self.classifier(F.dropout(pooled_output, p=self.dropout))
        return logits


class BertController:
    def __init__(self, num_labels, batch_size=16, dropout=0.2, lr=2e-5, early_stopping_rounds=5,
                 validate_every_n_steps=5_000, model_name=None, pretrained_model_name_or_path=None,
                 additional_features=None, pooling_type=None):
        """ `additional_features` is a dict with names and embedding sizes of additional features to consider"""
        self.model_name = time.strftime("%Y%m%d_%H%M%S") if model_name is None else model_name
        self.lr = lr
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.validate_every_n_steps = validate_every_n_steps
        self.early_stopping_rounds = early_stopping_rounds
        self.additional_features = list(additional_features.keys()) if additional_features is not None else []
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if self.pretrained_model_name_or_path is None:
            logging.warning("A pretrained model name or path was not specified, defaulting to base uncased mBERT")
            self.pretrained_model_name_or_path = "bert-base-multilingual-uncased"

        self.model = MorphologicalBertForSequenceClassification(num_labels=num_labels,
                                                                dropout=dropout,
                                                                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                                additional_features=additional_features,
                                                                pooling_type=pooling_type)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def custom_config(self):
        # TODO: should return properties to enable reconstruction of the custom parts of this model
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def train(self, train_dataset):
        self.model.train()
        total_num_batches = (len(train_dataset) + self.batch_size - 1) // self.batch_size
        train_loss = 0.0

        for curr_batch in tqdm(DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)):
            batch_labels = curr_batch["labels"].to(DEVICE)
            del curr_batch["labels"]
            logits = self.model(**curr_batch)  # [B, num_labels]
            curr_loss = self.loss(logits, batch_labels)
            train_loss += float(curr_loss)

            curr_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return train_loss / total_num_batches

    def validate(self, dev_dataset):
        with torch.no_grad():
            self.model.eval()
            total_num_batches = (len(dev_dataset) + self.batch_size - 1) // self.batch_size
            dev_loss = 0.0
            num_correct = 0

            for curr_batch in tqdm(DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)):
                batch_labels = curr_batch["labels"].to(DEVICE)
                del curr_batch["labels"]
                logits = self.model(**curr_batch)  # [B, num_labels]

                curr_loss = self.loss(logits, batch_labels)
                dev_loss += float(curr_loss)

                label_preds = torch.argmax(logits, dim=1)
                num_correct += int(torch.sum(label_preds == batch_labels))

            return {
                "loss": dev_loss / total_num_batches,
                "accuracy": num_correct / len(dev_dataset)
            }

    def fit(self, train_dataset, num_epochs, dev_dataset=None):
        best_dev_acc, rounds_no_increase = 0.0, 0
        stop_early = False

        t_start = time.time()
        for idx_epoch in range(num_epochs):
            log_to_stdout(f"Epoch#{1 + idx_epoch}/{num_epochs}")
            shuffled_indices = torch.randperm(len(train_dataset))

            num_minisets = (len(train_dataset) + self.validate_every_n_steps - 1) // self.validate_every_n_steps
            for idx_miniset in range(num_minisets):
                log_to_stdout(f"Miniset#{1 + idx_miniset}/{num_minisets}")
                curr_subset = Subset(train_dataset, shuffled_indices[idx_miniset * self.validate_every_n_steps:
                                                                     (idx_miniset + 1) * self.validate_every_n_steps])

                train_loss = self.train(curr_subset)
                log_to_stdout(f"Training loss: {train_loss: .4f}")

                if dev_dataset is None or len(curr_subset) < self.validate_every_n_steps // 2:
                    log_to_stdout(f"Skipping validation after training on a small training subset "
                                  f"({len(curr_subset)} examples)")
                    continue

                dev_metrics = self.validate(dev_dataset)
                log_to_stdout(f"Validation accuracy: {dev_metrics['accuracy']:.4f}")
                if dev_metrics["accuracy"] > best_dev_acc:
                    best_dev_acc, rounds_no_increase = dev_metrics["accuracy"], 0
                    log_to_stdout(f"New best, saving checkpoint TODO")
                    # TODO: save checkpoint
                    # ...
                else:
                    rounds_no_increase += 1

                if rounds_no_increase == self.early_stopping_rounds:
                    log_to_stdout(f"Stopping early after no improvement for {rounds_no_increase} checks")
                    log_to_stdout(f"Best accuracy: {best_dev_acc:.4f}")
                    stop_early = True
                    break

            if stop_early:
                break

        log_to_stdout(f"Training took {time.time() - t_start: .3f}s")


if __name__ == "__main__":
    from utils import BertDataset, UFEATS2IDX
    from transformers import BertTokenizer
    import json
    import pandas as pd
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    log_to_stdout("Loading training dataset")
    train_df = pd.read_csv(args.train_path)
    train_features = list(map(lambda features_str: json.loads(features_str), train_df["features"].values))
    train_dataset = BertDataset(sequences=train_df["content"].values,
                                labels=train_df["target"].values,
                                tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len,
                                additional_features=train_features,
                                ufeats_names=list(UFEATS2IDX.keys()) if args.include_ufeats else None)

    dev_df, dev_features, dev_dataset = None, None, None
    if args.dev_path:
        log_to_stdout("Loading validation dataset")
        dev_df = pd.read_csv(args.dev_path)
        dev_features = list(map(lambda features_str: json.loads(features_str), dev_df["features"].values))
        dev_dataset = BertDataset(sequences=dev_df["content"].values,
                                  labels=dev_df["target"].values,
                                  tokenizer=tokenizer,
                                  max_seq_len=args.max_seq_len,
                                  additional_features=dev_features,
                                  ufeats_names=list(UFEATS2IDX.keys()) if args.include_ufeats else None)

    num_labels = len(train_df["target"].value_counts())

    feature_sizes = {}
    if args.include_upostag:
        feature_sizes["upostag"] = args.upostag_emb_size

    if args.include_ufeats:
        for f in list(UFEATS2IDX.keys()):
            feature_sizes[f] = args.ufeats_emb_size

    trainer = BertController(model_name=args.model_name,
                             num_labels=num_labels,
                             batch_size=args.batch_size,
                             dropout=args.dropout,
                             lr=args.lr,
                             early_stopping_rounds=args.early_stopping_rounds,
                             validate_every_n_steps=args.validate_every_n_examples,
                             pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                             additional_features=feature_sizes,
                             pooling_type=args.pooling_type)
    trainer.fit(train_dataset, num_epochs=args.num_epochs, dev_dataset=dev_dataset)
