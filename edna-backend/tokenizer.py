"""
tokenizer.py — Single source of truth for KmerTokenizer
=========================================================
Import this in all training scripts and inference.py:

    from tokenizer import KmerTokenizer

Never reimplement or copy-paste this class elsewhere.
The pre-tokenized dataset on Modal was generated with k=6, max_length=512.
Any change here requires re-running pretokenize_option_a.py and re-uploading.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class KmerTokenizer:
    def __init__(self, k: int = 6):
        self.k = k
        bases = ['A', 'C', 'G', 'T']
        self.vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

        def generate_kmers(length):
            if length == 1:
                return bases
            return [p + b for p in generate_kmers(length - 1) for b in bases]

        self.vocab.extend(generate_kmers(k))
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        logger.info(f"KmerTokenizer: k={k}, vocab_size={self.vocab_size}")

    def tokenize(self, sequence: str, max_length: int = 512) -> List[int]:
        sequence = sequence.upper().replace('N', '')
        kmers = [
            sequence[i:i + self.k]
            for i in range(len(sequence) - self.k + 1)
            if all(b in 'ACGT' for b in sequence[i:i + self.k])
        ]
        ids = [self.token_to_id['[CLS]']]
        ids += [self.token_to_id.get(km, self.token_to_id['[UNK]']) for km in kmers[:max_length - 2]]
        ids.append(self.token_to_id['[SEP]'])
        ids += [self.token_to_id['[PAD]']] * (max_length - len(ids))
        return ids[:max_length]
