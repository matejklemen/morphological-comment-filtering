import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from utils import UPOS2IDX, UFEATS2IDX, PAD
import logging
import argparse
import os
import time

from pooling import MaskedMeanPooler, WeightedSumPooler, LSTMPooler

# A bit of a hack so that `print` is used instead of `log` on kaggle since that doesn't seem to work properly there
is_kaggle = os.path.exists("/kaggle/input")
log_to_stdout = print if is_kaggle else logging.info

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--train_path", type=str, default="preprocessed/train.csv")
parser.add_argument("--dev_path", type=str, default="preprocessed/val.csv")

parser.add_argument("--embedding_size", type=int, default=300)  # Note: fastText = 300d by default
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--bidirectional", action="store_true")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=5_000)
parser.add_argument("--max_seq_len", type=int, default=76)

parser.add_argument("--include_upostag", action="store_true")
parser.add_argument("--upostag_emb_size", type=int, default=50)
parser.add_argument("--include_ufeats", action="store_true")
parser.add_argument("--ufeats_emb_size", type=int, default=15)
parser.add_argument("--pooling_type", type=str, default="mean")

parser.add_argument("--lang", type=str, default="hr")


class MorphologicalLSTMForSequenceClassification(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_labels, bidirectional=False, dropout=0.2,
                 additional_features=None, pooling_type=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.additional_features = additional_features if additional_features is not None else {}
        self.pooling_type = pooling_type
        if len(self.additional_features) == 0:
            self.pooling_type = "N/A"

        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional).to(DEVICE)

        self.classifier = nn.Linear(hidden_size + sum(self.additional_features.values()),
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

    def forward(self, input_ids, input_mask, **kwargs):
        # input_ids: [B, max_seq_len, emb_size] -> EMBEDDINGS, not raw tokens!
        # input mask: [B, max_seq_len]
        batch_size = input_ids.shape[0]

        # Use last hidden state as a summary of sequence
        _, (last_hid, _) = self.lstm(input_mask.unsqueeze(2).to(DEVICE) * input_ids.to(DEVICE))

        # Concatenate the LSTM directions, i.e. unpacked_hid = [LSTM_l2r, LSTM_r2l]
        output = F.dropout(last_hid.transpose(0, 1).reshape(batch_size, self.num_directions * self.hidden_size),
                           p=self.dropout)  # [B, hidden_size]

        # Concatenate pooled POS tag / universal features embeddings for sequence to LSTM sequence representation
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
            output = torch.cat((output, additional_processed), dim=1)

        logits = self.classifier(F.dropout(output, p=self.dropout))
        return logits


class LSTMController:
    def __init__(self, embedding_size, hidden_size, num_labels, bidirectional=False, batch_size=16, dropout=0.2,
                 lr=2e-5, early_stopping_rounds=5, validate_every_n_steps=5_000, model_name=None,
                 additional_features=None, pooling_type=None):
        self.model_name = time.strftime("%Y%m%d_%H%M%S") if model_name is None else model_name
        self.lr = lr
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.validate_every_n_steps = validate_every_n_steps
        self.early_stopping_rounds = early_stopping_rounds
        self.additional_features = list(additional_features.keys()) if additional_features is not None else []
        self.model = MorphologicalLSTMForSequenceClassification(embedding_size=embedding_size,
                                                                hidden_size=hidden_size,
                                                                num_labels=num_labels,
                                                                bidirectional=bidirectional,
                                                                dropout=dropout,
                                                                additional_features=additional_features,
                                                                pooling_type=pooling_type).to(DEVICE)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

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
    from utils import FastTextLSTMDataset, UFEATS2IDX
    from torchnlp.word_to_vector import FastText
    import json
    import pandas as pd
    import os
    args = parser.parse_args()

    if not os.path.exists("models"):
        os.makedirs("models")
    ft = FastText(language=args.lang, cache="models")

    log_to_stdout("Loading training dataset")
    train_df = pd.read_csv(args.train_path)
    train_features = list(map(lambda features_str: json.loads(features_str), train_df["features"].values))
    train_dataset = FastTextLSTMDataset(sequences=train_df["content"].values,
                                        labels=train_df["target"].values,
                                        max_seq_len=args.max_seq_len,
                                        ft_embedder=ft,
                                        additional_features=train_features,
                                        ufeats_names=list(UFEATS2IDX.keys()) if args.include_ufeats else None)

    dev_df, dev_features, dev_dataset = None, None, None
    if args.dev_path:
        log_to_stdout("Loading validation dataset")
        dev_df = pd.read_csv(args.dev_path)
        dev_features = list(map(lambda features_str: json.loads(features_str), dev_df["features"].values))
        dev_dataset = FastTextLSTMDataset(sequences=dev_df["content"].values,
                                          labels=dev_df["target"].values,
                                          max_seq_len=args.max_seq_len,
                                          ft_embedder=ft,
                                          additional_features=dev_features,
                                          ufeats_names=list(UFEATS2IDX.keys()) if args.include_ufeats else None)

    del ft
    num_labels = len(train_df["target"].value_counts())

    feature_sizes = {}
    if args.include_upostag:
        feature_sizes["upostag"] = args.upostag_emb_size

    if args.include_ufeats:
        for f in list(UFEATS2IDX.keys()):
            feature_sizes[f] = args.ufeats_emb_size

    trainer = LSTMController(model_name=args.model_name,
                             embedding_size=args.embedding_size,
                             hidden_size=args.hidden_size,
                             bidirectional=args.bidirectional,
                             num_labels=num_labels,
                             batch_size=args.batch_size,
                             dropout=args.dropout,
                             lr=args.lr,
                             early_stopping_rounds=args.early_stopping_rounds,
                             validate_every_n_steps=args.validate_every_n_examples,
                             additional_features=feature_sizes,
                             pooling_type=args.pooling_type)
    trainer.fit(train_dataset, num_epochs=args.num_epochs, dev_dataset=dev_dataset)
