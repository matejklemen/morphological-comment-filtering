import numpy as np
import pandas as pd

train_path = "labeled/train.tsv"
dev_path = "labeled/dev.tsv"

# Test set is private: use dev set as test set
train_df = pd.read_csv(train_path, sep="\t")[["comments", "hate"]]
test_df = pd.read_csv(dev_path, sep="\t")[["comments", "hate"]]

train_df.columns = ["content", "target"]
test_df.columns = ["content", "target"]

# Group "offensive" and "hate" labels into one label -> test set has no labels
train_df.at[np.logical_or(train_df["target"] == "offensive", train_df["target"] == "hate"), "target"] = "hate"
test_df.at[np.logical_or(test_df["target"] == "offensive", test_df["target"] == "hate"), "target"] = "hate"

train_df["target"] = train_df["target"].apply(lambda str_label: {"none": 0, "hate": 1}[str_label])
test_df["target"] = test_df["target"].apply(lambda str_label: {"none": 0, "hate": 1}[str_label])

# Set aside part of training set as dev set
indices = np.random.permutation(train_df.shape[0])
train_indices = indices[: int(0.8 * indices.shape[0])]
dev_indices = indices[int(0.8 * indices.shape[0]):]

dev_df = train_df.iloc[dev_indices].reset_index(drop=True)
train_df = train_df.iloc[train_indices].reset_index(drop=True)

print(f"{train_df.shape[0]} train examples, \n"
      f"{dev_df.shape[0]} dev examples, \n"
      f"{test_df.shape[0]} test examples")

train_df.to_csv("train.csv", sep=",", index=False)
dev_df.to_csv("dev.csv", sep=",", index=False)
test_df.to_csv("test.csv", sep=",", index=False)
