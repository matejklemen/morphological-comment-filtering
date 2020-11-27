""" The following script is used to preprocess text once and cache it to a csv file. Currently, this means obtaining
    the UPOS tags and universal features + renaming columns to a common format.

    This is done because it's quite a long process and we do not want to do it every time we make a change. """

import pandas as pd
import os
import argparse
import json
import stanza

from conllu import parse
from tqdm import tqdm
from utils import PAD

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=os.path.join("clean_vs_hate_speech", "val.csv"),
                    help="PATH to your data")
parser.add_argument("--data_column", type=str, default="content",
                    help="Column of csv in which the text to be processed is stored")
parser.add_argument("--target_column", type=str, default="infringed_on_rule",
                    help="Column of csv in which the target label is stored")
parser.add_argument("--target_dir", type=str, default="preprocessed",
                    help="DIRECTORY where processed data should be stored")


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
            curr_features.update({"upostag": curr_token.get("upostag", PAD)})
            converted_sent.append(curr_features)
        processed.append(converted_sent)

    return processed


def extract_features(stanza_output):
    """ Filter the result returned by a stanza Pipeline, keeping only 'form' (raw word), 'upostag' and universal
    features (if present)"""
    # features of tokens inside sentence(s): each sentence is a list of dicts, containing token features
    relevant_features = []
    for curr_sent in stanza_output.sentences:
        sent_features = []
        for curr_token in curr_sent.tokens:
            processed_feats = {"form": curr_token.text}

            # Note: if FEATURES are not predicted for token, they will not be present in dict, whereas if POS TAG is not
            # predicted, a generic PAD gets written
            token_feats = curr_token.words[0].feats
            if token_feats is not None:
                for feat_val_pair in token_feats.split("|"):
                    feat, val = feat_val_pair.split("=")
                    processed_feats[feat] = val

            token_upos = curr_token.words[0].upos
            if token_upos is None:
                token_upos = PAD

            processed_feats["upostag"] = token_upos
            sent_features.append(processed_feats)
        relevant_features.append(sent_features)

    return relevant_features


if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    # stanza.download("hr", processors="tokenize,pos", package="ftb")
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', package="ewt")

    features = []
    for idx_ex in tqdm(range(df.shape[0])):
        curr_ex = df.iloc[idx_ex][args.data_column]
        output = nlp(curr_ex)
        ex_features = extract_features(output)

        features.append(json.dumps(ex_features))

    if not os.path.exists(args.target_dir):
        print("Warning: creating directory to store processed data")
        os.makedirs(args.target_dir)

    # Extract file name from given source path
    file_name = args.data_path.split(os.sep)[-1]
    target_path = os.path.join(args.target_dir, file_name)

    df["features"] = features
    df = df.rename({args.data_column: "content", args.target_column: "target"}, axis=1)
    df.to_csv(os.path.join(args.target_dir, file_name), index=False)


