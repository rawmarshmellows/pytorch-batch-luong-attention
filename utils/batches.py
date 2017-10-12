import torch
import logging
import numpy as np
from torch.autograd import Variable
from utils.tokens import Tokens


def indexes_from_sentence(embedding_map, sentence):
    return [embedding_map.get_index_from_word(word)
            for word in sentence.split(' ')] + [Tokens.EOS_token]


def pad_seq(seq, max_length):
    seq += [Tokens.PAD_token for _ in range(max_length - len(seq))]
    return seq


def batches(data,
            encoder_embedding_map,
            decoder_embedding_map,
            batch_size,
            bucket=False,
            use_cuda=False):

    source = data["source"]
    target = data["target"]

    # Bucket to make training faster
    if bucket:
        sorted_source_target = sorted(zip(source, target), key=lambda p: len(p[0]), reverse=True)
        source, target = zip(*sorted_source_target)
    else:
        shuffled_source_target = np.random.permutation(list(zip(source, target)))
        source, target = zip(*shuffled_source_target)

    n_samples = len(source)

    for i in range(0, n_samples, batch_size):
        source_seqs = []
        target_seqs = []
        source_batch = source[i:i+batch_size]
        target_batch = target[i:i+batch_size]
        logging.debug(f"Source batch:\n{source_batch}")
        logging.debug(f"Target batch:\n{target_batch}")

        for source_, target_ in zip(source_batch, target_batch):
            source_seqs.append(indexes_from_sentence(encoder_embedding_map, source_))
            target_seqs.append(indexes_from_sentence(decoder_embedding_map, target_))

        seq_pairs = sorted(zip(source_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        source_seqs, target_seqs = zip(*seq_pairs)

        source_lengths = [len(s) for s in source_seqs]
        source_padded = [pad_seq(seq, max(source_lengths)) for seq in source_seqs]
        target_lengths = [len(t) for t in target_seqs]
        target_padded = [pad_seq(seq, max(target_lengths)) for seq in target_seqs]

        source_var = Variable(torch.LongTensor(source_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

        if use_cuda:
            source_var = source_var.cuda()
            target_var = target_var.cuda()

        yield {"source_var": source_var,
               "source_lengths": source_lengths,
               "target_var": target_var,
               "target_lengths": target_lengths}

def data_from_batch(batch):

    # max input length, batch size
    source_var = batch["source_var"]

    # batch size
    source_lengths = batch["source_lengths"]

    # max target length, batch size
    target_var = batch["target_var"]

    # batch size
    target_lengths = batch["target_lengths"]

    return source_var, source_lengths, target_var, target_lengths



