# morphological-comment-filtering
Contains implementation of LSTM and BERT models, augmented with UPOS tag and universal feature embeddings.  
The morphological features get embedded separately for each token and then combined using one of the pooling mechanisms.
These then get concatenated with LSTM/BERT state and passed through a classification layer.

For details, please see our paper **Enhancing deep neural networks with morphological information**.
```
@misc{klemen2020enhancing,
      title={Enhancing deep neural networks with morphological information}, 
      author={Matej Klemen and Luka Krsnik and Marko Robnik-Šikonja},
      year={2020},
      eprint={2011.12432},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Rerunning the experiments
1. Install dependencies.
    ```shell script
    $ pip3 install -r requirements.txt
    $ pip3 install -r requirements-pos.txt
    ```
2. **Obtain the data**. Since Croatian data is proprietary, you can only rerun the English experiments. To obtain the data, 
follow the instructions in the paper: extract all examples where either of the following labels are marked as true -
`toxic`, `severe toxic`, `threats`, `identity hate` (positive label), and then sample the same amount of examples from 
the remaining examples (negative label).  
The files should be in `csv` format, with one column representing the sequence and one representing the 0/1 target.

3. **Preprocess the data**. This extracts the POS tags and universal features using Stanza. It might require 
downloading additional Stanza models for tagging tokens.  
Repeat this for train, dev and test set. 
    ```shell script
    $ python3 preprocess.py \
    --data_path="my_dataset/train.csv" \
    --data_column="content" \  # column in which sequences are stored
    --target_column="infringed_on_rule" \  # column in which targets are stored
    --target_dir="preprocessed"  # directory, in which to store the processed data
    ```
4. **Run the model**. For all the options, see `lstm.py` (for LSTMs) and `bert.py` (for BERTs).
The most important options are `--include_upostag`, `--include_ufeats`, `--upostag_emb_size`, `--ufeats_emb_size`.

## Using with your own data
Assuming you use the same formatting, this should run with any data. 
The two things you will need to tweak to your needs are: 
- Stanza models, used in `preprocess.py` (e.g. if you are processing German data, you wanna use German models) and
- FastText models, used in `lstm.py`.
