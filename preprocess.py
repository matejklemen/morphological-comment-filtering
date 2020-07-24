""" The following script is used to preprocess text once and cache it to a csv file. Currently, this means obtaining
    the UPOS tags and universal features.

    This is done because it's quite a long process and we do not want to do it every time we make a change. """

import pandas as pd
import os
import argparse
import json

from conllu import parse
from ufal.udpipe import Model, Pipeline, ProcessingError
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=os.path.join("clean_vs_hate_speech", "val.csv"),
                    help="PATH to your data")
parser.add_argument("--data_column", type=str, default="content",
                    help="Column of csv in which the text to be processed is stored")
parser.add_argument("--target_dir", type=str, default="preprocessed",
                    help="DIRECTORY where processed data should be stored")
parser.add_argument("--ud_model_path", type=str, default="models/croatian-set-ud-2.4-190531.udpipe",
                    help="Path to the universal dependencies model which is to be used for obtaining features")


def process_conllu(conllu_data):
    """ Accepts a conllu string, containing processed sequence, and returns a list[list[dict]] containing properties
        of tokens by sentence, i.e. index [i][j] of returned list represents features of j-th token in i-th sentence."""
    sent_features = parse(conllu_data)
    processed = []
    for curr_sent in sent_features:
        converted_sent = []
        for curr_token in curr_sent:
            curr_features = {"form": curr_token["form"]}
            # Unpack universal features; note that some tokens don't have universal features (e.g. punctuation)
            universal_features = curr_token["feats"]
            if universal_features is not None:
                curr_features.update(universal_features)
            curr_features.update({"upostag": curr_token.get("upostag", "N/A")})
            converted_sent.append(curr_features)
        processed.append(converted_sent)

    return processed


if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    err = ProcessingError()
    model = Model.load(args.ud_model_path)
    if not model:
        print(f"Could not load model from {args.ud_model_path}")
        exit(1)
    pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    features = []
    for idx_ex in tqdm(range(df.shape[0])):
        curr_ex = df.iloc[idx_ex][args.data_column]
        # outputs UPOS/XPOS tags, universal features, ... in conllu format, convert it to JSON
        res = pipeline.process(curr_ex, err)
        features.append(json.dumps(process_conllu(res)))

    if not os.path.exists(args.target_dir):
        print("Warning: creating directory to store processed data")
        os.makedirs(args.target_dir)

    # Extract file name from given source path
    file_name = args.data_path.split(os.sep)[-1]
    target_path = os.path.join(args.target_dir, file_name)

    df["features"] = features
    df.to_csv(os.path.join(args.target_dir, file_name), index=False)


