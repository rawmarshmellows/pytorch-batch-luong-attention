import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin
import re
import unicodedata

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def load_train_and_val(train_dir, debug_restrict_data, max_length=10, reverse=True):
    train = load_and_preprocess_file(pjoin(train_dir, "train.csv"), debug_restrict_data, max_length, reverse)
    val = load_and_preprocess_file(pjoin(train_dir, "val.csv"), debug_restrict_data, max_length, reverse)
    return train, val


def load_and_preprocess_file(fpath, debug_restrict_data, max_length, reverse):
    data = pd.read_csv(fpath)
    if debug_restrict_data is not None:
        pairs = list(zip(data["source"][:debug_restrict_data], data["target"][:debug_restrict_data]))
    else:
        pairs = list(zip(data["source"], data["target"]))

    pairs = [[normalize_string(p[0]), normalize_string(p[1])] for p in pairs]
    pairs = filter_pairs(pairs, max_length)

    output_data = []
    for source, target in tqdm(pairs):
        if reverse:
            output_data.append({"source": target, "target": source})
        else:
            output_data.append({"source": source, "target": target})
    return pd.DataFrame(output_data)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filter_pair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length and \
        p[0].startswith(eng_prefixes)


def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]

