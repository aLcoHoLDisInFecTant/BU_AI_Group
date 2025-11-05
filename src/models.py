import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        num_classes: int,
        embedding_mode: str = "random",
        glove_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_mode.startswith("glove") and glove_weights is not None:
            self.embedding.weight.data.copy_(glove_weights)
            self.embedding.weight.requires_grad = embedding_mode == "glove_finetune"

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq]
        emb = self.embedding(input_ids)
        out, (h_n, c_n) = self.lstm(emb)
        # Use last hidden state (concatenate directions if bidirectional)
        if self.lstm.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits


class BERTClassifier(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        num_classes: int,
        dropout: float = 0.1,
        freeze_all: bool = False,
        unfreeze_layers: int = 0,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

        if freeze_all:
            for p in self.bert.parameters():
                p.requires_grad = False
        elif unfreeze_layers > 0:
            # Freeze all first
            for p in self.bert.parameters():
                p.requires_grad = False
            # Unfreeze last N encoder layers
            if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
                for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
            # Also unfreeze pooler
            if hasattr(self.bert, "pooler"):
                for p in self.bert.pooler.parameters():
                    p.requires_grad = True
        # Classifier is always trainable
        for p in self.fc.parameters():
            p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Prefer CLS token representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_hidden)
        logits = self.fc(x)
        return logits


def load_glove_embeddings(glove_path: str, vocab: Dict[str, int], embedding_dim: int) -> torch.Tensor:
    """Build embedding matrix from GloVe file for given vocab.
    Tokens not found will be random normal.
    """
    vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            token = parts[0]
            vals = parts[1:]
            if len(vals) != embedding_dim:
                # Skip inconsistent lines
                continue
            vectors[token] = torch.tensor([float(x) for x in vals], dtype=torch.float)

    vocab_size = len(vocab)
    emb_matrix = torch.empty(vocab_size, embedding_dim)
    nn.init.normal_(emb_matrix, mean=0.0, std=0.1)
    for tok, idx in vocab.items():
        if tok in ("<pad>", "<unk>"):
            emb_matrix[idx] = torch.zeros(embedding_dim)
        elif tok in vectors:
            emb_matrix[idx] = vectors[tok]
    return emb_matrix