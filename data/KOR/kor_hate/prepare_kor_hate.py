import numpy as np
import pandas as pd

train_path = "labeled/train.tsv"
dev_path = "labeled/dev.tsv"
test_path = "test.no_label.tsv"

train_df = pd.read_csv(train_path, sep="\t")[["comments", "hate"]]
dev_df = pd.read_csv(dev_path, sep="\t")[["comments", "hate"]]
test_df = pd.read_csv(test_path, sep="\t")
# Test set is private
test_df["hate"] = "none"

train_df.columns = ["content", "target"]
dev_df.columns = ["content", "target"]
test_df.columns = ["content", "target"]

# Group "offensive" and "hate" labels into one label -> test set has no labels
train_df.at[np.logical_or(train_df["target"] == "offensive", train_df["target"] == "hate"), "target"] = "hate"
dev_df.at[np.logical_or(dev_df["target"] == "offensive", dev_df["target"] == "hate"), "target"] = "hate"

train_df["target"] = train_df["target"].apply(lambda str_label: {"none": 0, "hate": 1}[str_label])
dev_df["target"] = dev_df["target"].apply(lambda str_label: {"none": 0, "hate": 1}[str_label])
test_df["target"] = test_df["target"].apply(lambda str_label: {"none": 0, "hate": 1}[str_label])

print(f"{train_df.shape[0]} train examples, \n"
      f"{dev_df.shape[0]} dev examples, \n"
      f"{test_df.shape[0]} test examples")

train_df.to_csv("train.csv", sep=",", index=False)
dev_df.to_csv("dev.csv", sep=",", index=False)
test_df.to_csv("test.csv", sep=",", index=False)
