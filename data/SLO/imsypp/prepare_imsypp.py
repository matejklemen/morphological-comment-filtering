import numpy as np
import pandas as pd


if __name__ == "__main__":
    """ 
        Script to prepare IMSYPP-sl dataset:
        - keeps comments that have an agreement in annotation (either single annotation or both annotators agree),
        - removes a few comments that don't have an annotation (because it was not a Slovene tweet),
        - maps 4-label classification problem into a binary one
        - sets aside 10% of training set as a validation set
        - writes data to train/dev/test.csv
    """
    train_df = pd.read_csv("/home/matej/Documents/data/imsypp/IMSyPP_SI_anotacije_round1-no_conflicts.csv")
    test_df = pd.read_csv("/home/matej/Documents/data/imsypp/IMSyPP_SI_anotacije_round2.csv")
    LABEL_MAP = {
        "0 ni sporni govor": 0, "1 nespodobni govor": 1, "2 žalitev": 1, "3 nasilje": 1
    }

    # Each comment is annotated by one or two annotators: keep only those for which annotations agree
    # Also map labels to binary ones (0=is_clean/1=is_hate)
    train_comments, train_labels = [], []
    num_train_skipped = 0
    for tweet, group in train_df.groupby("ID"):
        labels = set(group["vrsta"].values)
        if len(labels) == 1:
            curr_label = labels.pop()
            if curr_label not in LABEL_MAP:
                print(f"Skipping example because label is '{curr_label}'")
                continue

            train_comments.append(group.iloc[0]["besedilo"])
            train_labels.append(LABEL_MAP[curr_label])
        else:
            num_train_skipped += 1

    print(f"Skipped {num_train_skipped} training examples due to disagreement")
    dedup = pd.DataFrame({"content": train_comments, "target": train_labels})
    indices = np.random.permutation(dedup.shape[0])
    bnd = int(0.8 * dedup.shape[0])
    train_indices, dev_indices = indices[:bnd], indices[bnd:]

    train_df = dedup.iloc[train_indices]
    dev_df = dedup.iloc[dev_indices]

    print(f"Training distribution ({train_df.shape[0]} examples):")
    print(train_df["target"].value_counts())
    print(f"Validation distribution ({dev_df.shape[0]} examples):")
    print(dev_df["target"].value_counts())

    train_df.to_csv("train.csv", index=False)
    dev_df.to_csv("dev.csv", index=False)

    test_comments, test_labels = [], []
    num_test_skipped = 0
    for tweet, group in test_df.groupby("ID"):
        labels = set(group["vrsta"].values)
        if len(labels) == 1:
            curr_label = labels.pop()
            if curr_label not in LABEL_MAP:
                print(f"Skipping example because label is '{curr_label}'")
                continue

            test_comments.append(group.iloc[0]["besedilo"])
            test_labels.append(LABEL_MAP[curr_label])
        else:
            num_test_skipped += 1

    test_df = pd.DataFrame({"content": test_comments, "target": test_labels})
    print(f"Test distribution ({test_df.shape[0]} examples):")
    print(test_df["target"].value_counts())
    test_df.to_csv("test.csv", index=False)
