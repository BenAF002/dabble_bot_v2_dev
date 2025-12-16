import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from string import ascii_lowercase
from pathlib import Path
import requests

class Words:
    """
    Word and phoneme representation data manager class
    Loads word embeddings and provides encoding/decoding methods
    """

    # class attributes
    alphabet = list(ascii_lowercase)
    ctoi = {c: i+1 for i, c in enumerate(alphabet)}
    itoc = {i+1: c for i, c in enumerate(alphabet)}

    def enc(cls, word): 
        # 27: '<sos>', 28: '<eos>', 0: padding
        res = [27] + [cls.ctoi[c] for c in word] + [28] + [0] * (16 - len(word))
        return torch.tensor(res, dtype=torch.int32)
    
    def dec(cls, arr): 
        if isinstance(arr, torch.Tensor):
            arr = [i.item() for i in arr]
        return ''.join([cls.itoc[i] for i in arr if i in range(1, 27)])

    def __init__(self):
        super().__init__()
        self.mappings = {}
        
        # use vector embeddings from https://github.com/rahulsrma26/phonetic-word-embedding
        simvecs_url = "https://raw.githubusercontent.com/rahulsrma26/phonetic-word-embedding/refs/heads/dev/vector_embeddings/simvecs"
        response = requests.get(simvecs_url)
        response.raise_for_status()  # Ensure we got a valid response
        if response.status_code == 200:
            for line in response.text.splitlines():
                m = line.split()
                if m[0].isalpha() and len(m[0]) <= 16:  # only words with alphabetic chars and length <= 16
                    self.mappings[m[0].lower()] = torch.tensor([float(x) for x in m[1:]])

        self.encoded_words = torch.stack([self.enc(w) for w in self.mappings.keys()])
        self.embeddings = torch.stack(list(self.mappings.values()))

    def train_test_split(self, train_pct: float = 0.8):
        """Generate train/test splits compatible with PyTorch DataLoader"""

        total_words = len(self.encoded_words)
        indices = np.random.permutation(total_words)
        train_indices = indices[:int(total_words * train_pct)]
        test_indices = indices[int(total_words * train_pct):]
        train_data = WordsDataSplit(self.encoded_words[train_indices], self.embeddings[train_indices])
        test_data = WordsDataSplit(self.encoded_words[test_indices], self.embeddings[test_indices])

        return train_data, test_data


class WordsDataSplit(Dataset):
    """
    Word and phoneme representation dataset for PyTorch DataLoader
    Facilitates train/test splits
    """
    def __init__(self, encoded_words, embeddings):
        super().__init__()
        self.encoded_words = encoded_words
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.encoded_words)
    
    def __getitem__(self, idx: int):
        word = self.encoded_words[idx]
        embedding = self.embeddings[idx]
        return word, embedding