import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from tqdm import tqdm
from transformers import BertModel
from torch.utils.data import DataLoader, Subset
import argparse

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using device {DEVICE}")

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

parser.add_argument("--include_upostag", action="store_true")  # TODO
parser.add_argument("--include_ufeats", action="store_true")  # TODO


class MorphologicalBertForSequenceClassification(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path, dropout=0.2,
                 additional_features=None, pooling_type=None):
        super().__init__()

        self.num_labels = num_labels
        self.dropout = dropout
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.additional_features = additional_features if additional_features is not None else {}
        self.pooling_type = pooling_type
        if len(self.additional_features) > 0 and self.pooling_type is None:
            self.pooling_type = "average"

        # classic: linear(dropout(BERT(input_data)))
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_name_or_path,
                                                    num_labels=num_labels).to(DEVICE)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size,  # TODO: + embedding sizes of additional features
                                    out_features=self.num_labels).to(DEVICE)

        # TODO: include embeddings for features (maybe ModuleLists, involving embedding->pool)
        self.embedders = nn.ModuleDict()
        for feature_name, emb_size in self.additional_features.items():
            pass

    @staticmethod
    def from_pretrained(model_dir):
        pass

    @staticmethod
    def save_pretrained(model_dir):
        pass

    def forward(self, input_ids, token_type_ids, attention_mask, **kwargs):  # TODO
        # Classic BERT sequence classification settings: last layer's hidden state for [CLS] token: [B, hidden_size]
        _, pooled_output = self.bert_model(input_ids=input_ids.to(DEVICE),
                                           token_type_ids=token_type_ids.to(DEVICE),
                                           attention_mask=attention_mask.to(DEVICE))

        logits = self.classifier(F.dropout(pooled_output, p=self.dropout))
        return logits


class Trainer:
    def __init__(self, num_labels, batch_size=16, dropout=0.2, lr=2e-5, early_stopping_rounds=5,
                 validate_every_n_steps=5_000, model_name=None, pretrained_model_name_or_path=None):
        """ `additional_features` is a dict with names and embedding sizes of additional features to consider"""
        self.model_name = time.strftime("%Y%m%d_%H%M%S") if model_name is None else model_name
        self.lr = lr
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.validate_every_n_steps = validate_every_n_steps
        self.early_stopping_rounds = early_stopping_rounds
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if self.pretrained_model_name_or_path is None:
            logging.warning("A pretrained model name or path was not specified, defaulting to base uncased mBERT")

        self.model = MorphologicalBertForSequenceClassification(num_labels=num_labels,
                                                                dropout=dropout,
                                                                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                                additional_features=None,
                                                                pooling_type=None)
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

    def run(self, train_dataset, num_epochs, dev_dataset=None):
        best_dev_acc, rounds_no_increase = 0.0, 0
        stop_early = False

        t_start = time.time()
        for idx_epoch in range(num_epochs):
            logging.info(f"Epoch#{1 + idx_epoch}/{num_epochs}")
            shuffled_indices = torch.randperm(len(train_dataset))

            num_minisets = (len(train_dataset) + self.validate_every_n_steps - 1) // self.validate_every_n_steps
            for idx_miniset in range(num_minisets):
                logging.info(f"Miniset#{1 + idx_miniset}/{num_minisets}")
                curr_subset = Subset(train_dataset, shuffled_indices[idx_miniset * self.validate_every_n_steps:
                                                                     (idx_miniset + 1) * self.validate_every_n_steps])

                train_loss = self.train(curr_subset)
                logging.info(f"Training loss: {train_loss: .4f}")

                if dev_dataset is None or len(curr_subset) < self.validate_every_n_steps // 2:
                    logging.info(f"Skipping validation after training on a small training subset "
                                 f"({len(curr_subset)} examples)")
                    continue

                dev_metrics = self.validate(dev_dataset)
                logging.info(f"Validation accuracy: {dev_metrics['accuracy']:.4f}")
                if dev_metrics["accuracy"] > best_dev_acc:
                    best_dev_acc, rounds_no_increase = dev_metrics["accuracy"], 0
                    logging.info(f"New best, saving checkpoint TODO")
                    # TODO: save checkpoint
                    # ...
                else:
                    rounds_no_increase += 1

                if rounds_no_increase == self.early_stopping_rounds:
                    logging.info(f"Stopping early after no improvement for {rounds_no_increase} checks")
                    logging.info(f"Best accuracy: {best_dev_acc:.4f}")
                    stop_early = True
                    break

            if stop_early:
                break

        logging.info(f"Training took {time.time() - t_start: .3f}s")


if __name__ == "__main__":
    from utils import SequenceDataset, UFEATS2IDX
    from transformers import BertTokenizer
    import json
    import pandas as pd
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    logging.info("Loading training dataset")
    train_df = pd.read_csv(args.train_path)
    train_features = list(map(lambda features_str: json.loads(features_str), train_df["features"].values))
    train_dataset = SequenceDataset(sequences=train_df["content"].values,
                                    labels=train_df["infringed_on_rule"].values,
                                    tokenizer=tokenizer,
                                    max_seq_len=args.max_seq_len,
                                    additional_features=train_features,
                                    ufeats_names=None)  # TODO

    dev_df, dev_features, dev_dataset = None, None, None
    if args.dev_path:
        logging.info("Loading validation dataset")
        dev_df = pd.read_csv(args.dev_path)
        dev_features = list(map(lambda features_str: json.loads(features_str), dev_df["features"].values))
        dev_dataset = SequenceDataset(sequences=dev_df["content"].values,
                                      labels=dev_df["infringed_on_rule"].values,
                                      tokenizer=tokenizer,
                                      max_seq_len=args.max_seq_len,
                                      additional_features=dev_features,
                                      ufeats_names=None)  # TODO

    num_labels = len(train_df["infringed_on_rule"].value_counts())

    trainer = Trainer(model_name=args.model_name,
                      num_labels=num_labels,
                      batch_size=args.batch_size,
                      dropout=args.dropout,
                      lr=args.lr,
                      early_stopping_rounds=args.early_stopping_rounds,
                      validate_every_n_steps=args.validate_every_n_examples,
                      pretrained_model_name_or_path=args.pretrained_model_name_or_path)
    trainer.run(train_dataset, num_epochs=1, dev_dataset=dev_dataset)
