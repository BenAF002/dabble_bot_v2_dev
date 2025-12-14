import numpy as np
# import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from string import ascii_lowercase
import nltk
nltk.download('wordnet')
nltk.download('cmudict')
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
import os
from pathlib import Path

class Words(Dataset):
    """Dataset class for loading words and their phoneme representations."""

    # class attributes
    alphabet = list(ascii_lowercase)
    ctoi = {c: i for i, c in enumerate(alphabet)}
    itoc = {i: c for i, c in enumerate(alphabet)}
    lemmas = [list(l.lower()) for l in wn.all_lemma_names() if l.isalpha() and len(l) <= 15]

    def enc(cls, word): return [cls.ctoi[c] for c in word] + [0] * (15 - len(word))  # right pad with zeros
    def dec(cls, arr): return ''.join([cls.itoc[i] for i in arr if i != 0])          # remove padding zeros

    def __init__(self, vec_path: str = None):
        super().__init__()
        self.vec_path = Path(__file__).parent / 'simvecs' if vec_path is None else Path(vec_path)

        mappings = []
        with open(self.vec_path, 'r') as file:
            for line in file.read().splitlines():
                m = line.split()
                if m[0].isalpha() and len(m[0]) <= 15:
                    mappings.append(m)

        self.embedded_words = [list(mappings[i][0].lower()) for i in range(len(mappings))]
        self.encoded_words = np.array([self.enc(w) for w in self.embedded_words]).astype(int)
        self.embeddings = np.array([mappings[i][1:] for i in range(len(mappings))]).astype(float)

        