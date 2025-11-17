from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import calculate_metrics, compute_confusion_matrix, compute_error_buckets


def _forward_batch(model, batch, device):
    if "attention_mask" in batch:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
    else:
        logits = model(
            input_ids=batch["input_ids"].to(device),
        )
    return logits


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: Optional[float] = None,
) -> Dict[str, Any]:
    model.train()
    losses = []
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Train", leave=True):
        optimizer.zero_grad()
        logits = _forward_batch(model, batch, device)
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        if gradient_clip is not None and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        labels = batch["label"].cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
    return {"loss": float(np.mean(losses)), **metrics}


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    losses = []
    all_preds, all_labels, all_lengths = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=True):
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, batch["label"].to(device))
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            if "length" in batch:
                lengths = batch["length"].cpu().numpy()
                all_lengths.extend(lengths.tolist())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    metrics = calculate_metrics(preds_arr, labels_arr)
    result: Dict[str, Any] = {"loss": float(np.mean(losses)), **metrics}
    if labels_arr.size:
        result["confusion_matrix"] = compute_confusion_matrix(preds_arr, labels_arr)
    if all_lengths and len(all_lengths) == len(all_labels):
        lengths_arr = np.array(all_lengths)
        result["error_buckets"] = compute_error_buckets(lengths_arr, labels_arr, preds_arr)
    return result