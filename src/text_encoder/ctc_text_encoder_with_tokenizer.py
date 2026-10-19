"""
CTC text encoder backed by a HuggingFace tokenizer (SentencePiece-based).
Blank token is reserved at index 0; all tokenizer ids are shifted by +1.
"""

import os
from typing import Optional

import torch
from transformers import AutoTokenizer


class CTCTextEncoderWithTokenizer:
    EMPTY_TOK = ""  # CTC blank

    DEFAULT_HF_MODEL = "t5-small"  # SentencePiece-based, English-capable

    def __init__(
        self,
        hf_model_name: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        self.hf_model_name = hf_model_name or self.DEFAULT_HF_MODEL
        self.tokenizer_path = tokenizer_path
        self._use_sp = False
        self._sp = None
        self._hf = None

        if tokenizer_path and os.path.exists(tokenizer_path):
            import sentencepiece as spm

            self._sp = spm.SentencePieceProcessor()
            self._sp.load(tokenizer_path)
            self._use_sp = True
        else:
            self._hf = AutoTokenizer.from_pretrained(self.hf_model_name)

        self.vocab_size = self._vocab_size()

    def _vocab_size(self):
        if self._use_sp:
            return self._sp.get_piece_size() + 1  # +1 blank
        return self._hf.vocab_size + 1  # +1 blank

    def __len__(self):
        return self.vocab_size

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = "".join(c if (c.isalnum() or c == " ") else " " for c in text)
        text = " ".join(text.split())
        return text

    def encode(self, text) -> "torch.Tensor":
        text = self.normalize_text(text)
        if self._use_sp:
            sp_ids = self._sp.encode_as_ids(text)
            ids = [i + 1 for i in sp_ids]
        else:
            ids = self._hf.encode(text, add_special_tokens=False)
            ids = [i + 1 for i in ids]
        if len(ids) == 0:
            return torch.zeros(0, dtype=torch.long).unsqueeze(0)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode(self, inds) -> str:
        sp_ids = [int(i) - 1 for i in inds if int(i) != 0]
        if self._use_sp:
            return self._sp.decode_ids(sp_ids)
        return self._hf.decode(sp_ids, skip_special_tokens=True)

    def ctc_decode(self, inds) -> str:
        result = []
        prev = 0
        for ind in inds:
            ind = int(ind)
            if ind == 0:
                prev = 0
                continue
            if ind == prev:
                continue
            result.append(ind)
            prev = ind
        return self.decode(result)

    @property
    def alphabet(self):
        if self._use_sp:
            return [self._sp.id_to_piece(i) for i in range(self._sp.get_piece_size())]
        return list(self._hf.get_vocab().keys())

    @property
    def vocab(self):
        return [self.EMPTY_TOK] + list(self.alphabet)

    @property
    def ind2char(self):
        if self._use_sp:
            return {0: self.EMPTY_TOK} | {
                i + 1: self._sp.id_to_piece(i) for i in range(self._sp.get_piece_size())
            }
        vocab = self._hf.get_vocab()
        return {0: self.EMPTY_TOK} | {tid + 1: tok for tok, tid in vocab.items()}

    @property
    def char2ind(self):
        return {v: k for k, v in self.ind2char.items()}
