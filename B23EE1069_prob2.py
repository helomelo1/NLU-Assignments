import pandas as pd
import numpy as np
from collections import defaultdict


class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = defaultdict(int)
        self.id_to_word = defaultdict(str)
        self.freq = defaultdict(int)
        self.vocab_size = 0
        self.vocab = list()

    def init_vocab(self, strings):
        for sentence in strings.split('.'):
            words = sentence.split()

            for word in words:
                chars = list(word) + ['</s>']
                for i in range(len(chars) - 1):
                    pair = chars[i] + chars[i + 1]
                    self.freq[pair] += 1

    def train(self, strings, k):
        for _ in range(k):
            if not self.freq:
                break

            most_frequent = max(self.freq, key=lambda x: self.freq[x])
            self.vocab.append(most_frequent)

            t_new = ''.join(most_frequent)
            new_freq = defaultdict(int)

            for pair in self.freq:
                cnt = self.freq[pair]
                if pair == most_frequent:
                    continue
                new_pair = list(pair)
                
                t_l = new_pair[0]
                t_r = new_pair[1]

                if t_l == most_frequent[0] and t_r == most_frequent[1]:
                    new_pair[0] = t_new
                    new_pair.pop(1)

                new_freq[tuple(new_pair)] += cnt

            self.freq = new_freq

    def tokenize(self):
