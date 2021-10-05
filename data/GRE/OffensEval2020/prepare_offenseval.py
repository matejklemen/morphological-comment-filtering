import os

import numpy as np
import pandas as pd

training_path = "offenseval-gr-training-v1/offenseval-gr-training-v1.tsv"
test_dir = "offenseval-gr-testsetv1"

with open(training_path, "r") as f:
      train_content = list(map(lambda s: s.strip().split("\t"), f.readlines()))
      train_content = train_content[1:]

train_df = pd.DataFrame({
      "content": list(map(lambda l: l[1], train_content)),
      "target": list(map(lambda l: l[2], train_content))
})

with open(os.path.join(test_dir, "offenseval-gr-test-v1.tsv"), "r") as f:
      test_text = list(map(lambda s: s.strip().split("\t"), f.readlines()))
      test_text = test_text[1:]

with open(os.path.join(test_dir, "offenseval-gr-labela-v1.csv"), "r") as f:
      test_labels = list(map(lambda s: s.strip().split(","), f.readlines()))

test_df_txt = pd.DataFrame({
      "content": list(map(lambda l: l[1], test_text))
})

test_df_lbl = pd.DataFrame({
      "target": list(map(lambda l: l[1], test_labels))
})

test_df = pd.concat((test_df_txt, test_df_lbl), axis=1)

train_df["content"] = train_df["content"].apply(lambda s: s.replace("<LF>", ". "))
test_df["content"] = test_df["content"].apply(lambda s: s.replace("<LF>", ". "))

train_df["target"] = train_df["target"].apply(lambda str_label: {"NOT": 0, "OFF": 1}[str_label])
test_df["target"] = test_df["target"].apply(lambda str_label: {"NOT": 0, "OFF": 1}[str_label])

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









