import os
import re
from collections import Counter
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))  # remove HTML tags
    text = text.lower().strip()
    return text


def tokenize_basic(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text)


def build_vocab(texts: List[str], min_freq: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize_basic(t))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    return vocab, inv_vocab


def encode_text(tokens: List[str], vocab: Dict[str, int], max_length: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids += [vocab[PAD_TOKEN]] * (max_length - len(ids))
    return ids


def read_imdb_csv(csv_path: str, sample_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns: 'review', 'sentiment' -> map to 0/1
    df = df[["review", "sentiment"]].copy()
    df["review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
    df = df.reset_index(drop=True)
    return df


def split_df(df: pd.DataFrame, split: List[float], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1"
    train_ratio, val_ratio, test_ratio = split
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    df = df.sample(frac=1.0, random_state=seed)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    return train_df, val_df, test_df


class TextDatasetRNN(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize_basic(self.texts[idx])
        ids = encode_text(tokens, self.vocab, self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TextDatasetBERT(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_dataloaders(config: Dict[str, Any]) -> Dict[str, Any]:
    csv_path = config.get("csv_path", "../data/IMDB Dataset.csv")
    sample_size = int(config.get("sample_size", 0) or 0)
    seed = int(config.get("seed", 42))
    split = config.get("train_val_test_split", [0.8, 0.1, 0.1])
    batch_size = int(config.get("batch_size", 32))
    max_length = int(config.get("max_length", 200))
    model_type = config.get("model_type", "rnn")

    # Read CSV
    if not os.path.isabs(csv_path):
        # Resolve relative to project root (sentiment-analysis-project)
        root = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.abspath(os.path.join(root, csv_path))

    df = read_imdb_csv(csv_path, sample_size, seed)
    train_df, val_df, test_df = split_df(df, split, seed)

    result = {"num_classes": 2}

    if model_type in ("rnn", "lstm"):
        # Build vocab from train only
        min_freq = int(config.get("min_freq", 2))
        vocab, inv_vocab = build_vocab(train_df["review"].tolist(), min_freq)
        result["vocab"] = vocab
        # Datasets
        train_ds = TextDatasetRNN(train_df["review"].tolist(), train_df["label"].tolist(), vocab, max_length)
        val_ds = TextDatasetRNN(val_df["review"].tolist(), val_df["label"].tolist(), vocab, max_length)
        test_ds = TextDatasetRNN(test_df["review"].tolist(), test_df["label"].tolist(), vocab, max_length)
        # Loaders
        result["train_loader"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        result["val_loader"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        result["test_loader"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        # BERT tokenizer
        bert_model_name = config.get("bert_model_name", "bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name, local_files_only=True)
        train_ds = TextDatasetBERT(train_df["review"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
        val_ds = TextDatasetBERT(val_df["review"].tolist(), val_df["label"].tolist(), tokenizer, max_length)
        test_ds = TextDatasetBERT(test_df["review"].tolist(), test_df["label"].tolist(), tokenizer, max_length)
        result["train_loader"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        result["val_loader"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        result["test_loader"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return result