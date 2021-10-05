import numpy as np
import pandas as pd

training_path = "germeval2018.training.txt"
test_path = "germeval2018.test.txt"

train_df = pd.read_csv(training_path, sep="\t", header=None).iloc[:, [0, 1]]
train_df.columns = ["content", "target"]
test_df = pd.read_csv(test_path, sep="\t", header=None).iloc[:, [0, 1]]
test_df.columns = ["content", "target"]

train_df["target"] = train_df["target"].apply(lambda str_label: {"OTHER": 0, "OFFENSE": 1}[str_label])
test_df["target"] = test_df["target"].apply(lambda str_label: {"OTHER": 0, "OFFENSE": 1}[str_label])

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









