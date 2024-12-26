import torch
import numpy as np
from nltk import sent_tokenize, word_tokenize


# Tokenization
def tokenize(text):
    return text.split()


def nltk_tokenize(text):
    sentences = sent_tokenize(text)
    return [word_tokenize(sent) for sent in sentences]


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenize(text)


# Build vocabulary
def build_vocab(texts):
    counter = dict()
    for text in texts:
        for word in text.split():
            counter[word] = counter.get(word, 0) + 1
    vocab = {token: idx + 1 for idx, token in enumerate(counter.keys())}
    vocab['<pad>'] = 0
    vocab['<unk>'] = len(vocab)
    print(f'Vocab length: {len(vocab)}')
    return vocab


def encode_texts(texts, vocab):
    return [torch.tensor([vocab.get(token, len(vocab)) for token in tokenize(text)]) for text in texts]


def load_glove_embeddings(filepath):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def load_fasttext_embeddings(filepath, embedding_dim):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # Skip the first line (header)
        for line in f:
            values = line.split()
            word = values[0]  # The word
            vector = np.asarray(values[1:], dtype='float32')  # The embedding vector
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} word vectors.")
    return embeddings
    

class Vocab:
    """Vocabulary for text."""
    def __init__(self, texts, min_freq=5, reserved_tokens=[]):
        counter = dict()
        for text in texts:
            for word in text.split():
                counter[word] = counter.get(word, 0) + 1
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def get(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


