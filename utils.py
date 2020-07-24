import torch
import pandas as pd
from torch.utils.data import Dataset
from itertools import chain

PAD = "<PAD>"  # used to pad sequences to common sequence length

UPOS_TAGS = [PAD, "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
             "PUNCT", "SCONJ", "SYM", "VERB", "X"]
UPOS2IDX = {tag: i for i, tag in enumerate(UPOS_TAGS)}
IDX2UPOS = dict(enumerate(UPOS_TAGS))

UFEATS2IDX = {
    "PronType": {
        tag: i for i, tag in enumerate([PAD, "Art", "Dem", "Emp", "Exc", "Ind", "Int", "Neg", "Prs", "Rcp", "Rel",
                                        "Tot"])
    },
    "NumType": {
        tag: i for i, tag in enumerate([PAD, "Card", "Dist", "Frac", "Mult", "Ord", "Range", "Sets"])
    },
    "Poss": {
        tag: i for i, tag in enumerate([PAD, "Yes"])
    },
    "Reflex": {
        tag: i for i, tag in enumerate([PAD, "Yes"])
    },
    "Foreign": {
        tag: i for i, tag in enumerate([PAD, "Yes"])
    },
    "Abbr": {
        tag: i for i, tag in enumerate([PAD, "Yes"])
    },
    "Gender": {
        tag: i for i, tag in enumerate([PAD, "Com", "Fem", "Masc", "Neut"])
    },
    "Animacy": {
        tag: i for i, tag in enumerate([PAD, "Anim", "Hum", "Inan", "Nhum"])
    },
    "NounClass": {
        tag: i for i, tag in enumerate([PAD, "Bantu1", "Bantu2", "Bantu3", "Bantu4", "Bantu5",  "Bantu6", "Bantu7",
                                        "Bantu8", "Bantu9", "Bantu10", "Bantu11", "Bantu12", "Bantu13", "Bantu14",
                                        "Bantu15", "Bantu16", "Bantu17", "Bantu18", "Bantu19", "Bantu20"])
    },
    "Number": {
        tag: i for i, tag in enumerate([PAD, "Coll", "Count", "Dual", "Grpa", "Grpl", "Inv", "Pauc", "Plur", "Ptan",
                                        "Sing", "Tri"])
    },
    "Case": {
        tag: i for i, tag in enumerate([PAD, "Abs", "Acc", "Erg", "Nom", "Abe", "Ben", "Cau", "Cmp", "Cns", "Com",
                                        "Dat", "Dis", "Equ", "Gen", "Ins", "Par", "Tem", "Tra", "Voc", "Abl", "Add",
                                        "Ade", "All", "Del", "Ela", "Ess", "Ill", "Ine", "Lat", "Loc", "Per", "Sub",
                                        "Sup", "Ter"])
    },
    "Definite": {
        tag: i for i, tag in enumerate([PAD, "Com", "Cons", "Def", "Ind", "Spec"])
    },
    "Degree": {
        tag: i for i, tag in enumerate([PAD, "Abs", "Cmp", "Equ", "Pos", "Sup"])
    },
    "VerbForm": {
        tag: i for i, tag in enumerate([PAD, "Conv", "Fin", "Gdv", "Ger", "Inf", "Part", "Sup", "Vnoun"])
    },
    "Mood": {
        tag: i for i, tag in enumerate([PAD, "Adm", "Cnd", "Des", "Imp", "Ind", "Jus", "Nec", "Opt", "Pot", "Prp",
                                        "Qot", "Sub"])
    },
    "Tense": {
        tag: i for i, tag in enumerate([PAD, "Fut", "Imp", "Past", "Pqp", "Pres"])
    },
    "Aspect": {
        tag: i for i, tag in enumerate([PAD, "Hab", "Imp", "Iter", "Perf", "Prog", "Prosp"])
    },
    "Voice": {
        tag: i for i, tag in enumerate([PAD, "Act", "Antip", "Bfoc", "Cau", "Dir", "Inv", "Lfoc", "Mid", "Pass", "Rcp"])
    },
    "Evident": {
        tag: i for i, tag in enumerate([PAD, "Fh", "Nfh"])
    },
    "Polarity": {
        tag: i for i, tag in enumerate([PAD, "Neg", "Pos"])
    },
    "Person": {
        tag: i for i, tag in enumerate([PAD, "0", "1", "2", "3", "4"])
    },
    "Polite": {
        tag: i for i, tag in enumerate([PAD, "Elev", "Form", "Humb", "Infm"])
    },
    "Clusivity": {
        tag: i for i, tag in enumerate([PAD, "Ex", "In"])
    }
}


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_seq_len, label_encoding=None,
                 additional_features=None, ufeats_names=None):
        """
            If `label_encoding` is None, leave labels as they are.
            `ufeats_names` (list) is required because a universal feature for a tag can be missing from
            `additional_features` for two reasons (and we cannot know which one it is):
            (1) either it is not given or
            (2) some universal feature is not applicable to current tag.
        """
        self.ufeats_names = ufeats_names if ufeats_names is not None else []
        self.has_upos, self.has_ufeats = False, len(self.ufeats_names) > 0

        # Basic features (always present)
        self.inputs, self.segments, self.attn_masks, self.labels = [], [], [], []
        # Additional features (required for some models)
        self.upos, self.upos_masks = [], []
        self.ufeats = {u: [] for u in self.ufeats_names}
        self.ufeats_masks = {u: [] for u in self.ufeats_names}

        for i in range(len(sequences)):
            curr_encoded = tokenizer.encode_plus(sequences[i], max_length=max_seq_len,
                                                 pad_to_max_length=True,
                                                 truncation=True,
                                                 truncation_strategy="only_first")
            self.inputs.append(curr_encoded["input_ids"])
            self.segments.append(curr_encoded["token_type_ids"])
            self.attn_masks.append(curr_encoded["attention_mask"])
            self.labels.append(labels[i] if label_encoding is None else label_encoding[labels[i]])

            # flatten features for sentences into features for one continuous sequence
            flat_features = list(chain(*additional_features[i])) if additional_features is not None else []

            # [Features for current example]
            upos_features, upos_masks = [], []
            ufeats = {u: [] for u in self.ufeats_names}
            ufeats_masks = {u: [] for u in self.ufeats_names}

            idx_subword = 0
            # Use "gold" tokens to produce approximate alignment between UPOS tags / universal features and subwords:
            # tokenize word into subwords and add same POS tags/features multiple times for each produced subword
            for token_info in flat_features:
                if idx_subword >= max_seq_len:
                    break

                curr_token = token_info["form"]
                curr_subwords = tokenizer.tokenize(curr_token)

                curr_upos = token_info.get("upostag")
                if curr_upos is not None:
                    upos_features.extend([UPOS2IDX[curr_upos]] * len(curr_subwords))
                    upos_masks.extend([1] * len(curr_subwords))

                for curr_ufeat_name in self.ufeats_names:
                    curr_ufeat = token_info.get(curr_ufeat_name, PAD)
                    # Sometimes, ufeats might have multiple values associated - take first one as a simplification
                    feat_values = curr_ufeat.split(",")
                    if len(feat_values) > 1:
                        curr_ufeat = feat_values[0]

                    encoded_ufeat = UFEATS2IDX[curr_ufeat_name][curr_ufeat]
                    ufeats[curr_ufeat_name].extend([encoded_ufeat] * len(curr_subwords))
                    ufeats_masks[curr_ufeat_name].extend([curr_ufeat != PAD] * len(curr_subwords))

                idx_subword += len(curr_subwords)

            # Pad encoded UPOS tags and masks to max sequence length, additionally masking out [CLS] and [SEP]
            if upos_features:
                self.has_upos = True
                upos_features = [UPOS2IDX[PAD]] + \
                                (upos_features + [UPOS2IDX[PAD]] * (max_seq_len - len(upos_features)))[: max_seq_len - 2] + \
                                [UPOS2IDX[PAD]]
                upos_masks = [0] + (upos_masks + [0] * (max_seq_len - len(upos_masks)))[: max_seq_len - 2] + [0]
                self.upos.append(upos_features)
                self.upos_masks.append(upos_masks)

            # Pad encoded universal features and masks to max sequence length, additionally masking out [CLS] and [SEP]
            for curr_ufeat_name in self.ufeats_names:
                curr_ufeats = ufeats[curr_ufeat_name]
                padded_ufeats = (curr_ufeats + [UFEATS2IDX[curr_ufeat_name][PAD]] * (max_seq_len - len(curr_ufeats)))
                trimmed_ufeats = [UFEATS2IDX[curr_ufeat_name][PAD]] + \
                                 padded_ufeats[: max_seq_len - 2] + \
                                 [UFEATS2IDX[curr_ufeat_name][PAD]]
                self.ufeats[curr_ufeat_name].append(trimmed_ufeats)

                curr_ufeats_masks = ufeats_masks[curr_ufeat_name]
                padded_ufeats_masks = (curr_ufeats_masks + [0] * (max_seq_len - len(curr_ufeats_masks)))
                trimmed_ufeats_masks = [0] + padded_ufeats_masks[: max_seq_len - 2] + [0]
                self.ufeats_masks[curr_ufeat_name].append(trimmed_ufeats_masks)

        self.inputs, self.labels = torch.tensor(self.inputs), torch.tensor(self.labels)
        self.segments, self.attn_masks = torch.tensor(self.segments), torch.tensor(self.attn_masks)
        self.upos, self.upos_masks = torch.tensor(self.upos), torch.tensor(self.upos_masks)
        for curr_ufeat_name in self.ufeats_names:
            self.ufeats[curr_ufeat_name] = torch.tensor(self.ufeats[curr_ufeat_name])
            self.ufeats_masks[curr_ufeat_name] = torch.tensor(self.ufeats_masks[curr_ufeat_name])

    def __getitem__(self, index):
        return_dict = {
            "input_ids": self.inputs[index],
            "token_type_ids": self.segments[index],
            "attention_mask": self.attn_masks[index],
            "labels": self.labels[index]
        }

        if self.has_upos:
            return_dict["upos_ids"] = self.upos[index]
            return_dict["upos_mask"] = self.upos_masks[index]

        for curr_ufeat_name in self.ufeats_names:
            return_dict[curr_ufeat_name] = self.ufeats[curr_ufeat_name][index]
            return_dict[f"{curr_ufeat_name}_mask"] = self.ufeats_masks[curr_ufeat_name][index]

        return return_dict

    def __len__(self):
        return self.inputs.shape[0]


# TODO: remove this as it won't be needed any longer (see SequenceDataset)
def create_bert_example(seq, tokenizer, max_seq_len=300):
    """
        seq: str
            The comment that is to be classified
        max_seq_len: int
            Maximum length of sequence

        Example of how encoding works (total number of tokens is `max_seq_len`)
        Tokens:     [CLS] The quick brown fox jumps over the lazy dog [SEP] [PAD] ...
        Segments:     0    0    0     0    0    0    0    0    0   0    0     0
        Mask:         1    1    1     1    1    1    1    1    1   1    1     0
    """
    tokens = tokenizer.tokenize(seq)
    if len(tokens) > max_seq_len - 2:
        # allow space for [CLS] at start and [SEP] in the middle
        tokens = tokens[: max_seq_len - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    segments = [0] * max_seq_len
    # determines which tokens are valid and which are just padded for batching
    mask = [1] * len(tokens) + [0] * (max_seq_len - len(tokens))

    tokens += [tokenizer.pad_token] * (max_seq_len - len(tokens))

    return tokenizer.convert_tokens_to_ids(tokens), segments, mask

# TODO: remove this as it won't be needed any longer (see SequenceDataset)
def load_data(path, tokenizer):
    inps, seg, mask, y = [], [], [], []
    data = pd.read_csv(path)
    for idx_example in range(data.shape[0]):
        # [0] = example, [1] = label
        curr_data, curr_seg, curr_mask = create_bert_example(data.iloc[idx_example, 0], tokenizer)
        curr_label = data.iloc[idx_example, 1]
        inps.append(curr_data)
        seg.append(curr_seg)
        mask.append(curr_mask)
        y.append(curr_label)

    return torch.tensor(inps), torch.tensor(seg), torch.tensor(mask), torch.tensor(y)


if __name__ == "__main__":
    from transformers import BertTokenizer
    import json
    df = pd.read_csv("preprocessed/val.csv")
    features = list(map(lambda features_str: json.loads(features_str), df["features"].values))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = SequenceDataset(sequences=df["content"].values,
                              labels=df["infringed_on_rule"].values,
                              tokenizer=tokenizer,
                              max_seq_len=10,
                              additional_features=features,
                              ufeats_names=list(UFEATS2IDX.keys()))
