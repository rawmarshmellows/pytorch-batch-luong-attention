from utils.tokens import Tokens
from tqdm import tqdm
import torch.nn as nn
import logging


class EmbeddingMap(object):
    def __init__(self):
        self.word2index = {"<PAD>": Tokens.PAD_token,
                           "<UNK>": Tokens.UNK_token,
                           "<SOS>": Tokens.SOS_token,
                           "<EOS>": Tokens.EOS_token}
        self.word2count = {}
        self.index2word = {Tokens.PAD_token: "<PAD>",
                           Tokens.UNK_token: "<UNK>",
                           Tokens.SOS_token: "<SOS>",
                           Tokens.EOS_token: "<EOS>"}
        self.n_words = 4

    def index_words(self, sentence):
        for word in sentence.split():
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def get_index_from_word(self, word):
        if word not in self.word2index:
            return self.word2index["<UNK>"]
        else:
            return self.word2index[word]

    def get_indexes_from_sentences(self, sentence):
        indexes = []
        for word in sentence:
            indexes.append(self.get_index_from_word(word))
        return indexes

    def get_sentence_from_indexes(self, indexes):
        words = []
        for index in indexes:
            words.append(self.index2word[index])
        return words


def create_embedding_map(data):
    embedding_map = EmbeddingMap()
    for d in tqdm(data):
        for row in tqdm(d):
            embedding_map.index_words(row)
    return embedding_map


def create_embedding_maps(train, val, hidden_size, different_vocab=False):
    train_source = train["source"].tolist()
    train_target = train["target"].tolist()
    val_source = val["source"].tolist()
    val_target = val["target"].tolist()

    if different_vocab:
        encoder_embedding_map = create_embedding_map([train_source, val_source])
        decoder_embedding_map = create_embedding_map([train_target, val_target])
        encoder_embedding_matrix = nn.Embedding(encoder_embedding_map.n_words, hidden_size)
        decoder_embedding_matrix = nn.Embedding(decoder_embedding_map.n_words, hidden_size)
        return encoder_embedding_map, decoder_embedding_map, \
               encoder_embedding_matrix, decoder_embedding_matrix
    else:
        embedding_map = create_embedding_map([train_source,
                                              train_target,
                                              val_source,
                                              val_target])
        embedding_matrix = nn.Embedding(embedding_map.n_words, hidden_size)
        return embedding_map, embedding_map, embedding_matrix, embedding_matrix
