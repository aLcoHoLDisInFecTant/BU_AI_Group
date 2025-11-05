import argparse
import os
import sys
from typing import Dict, Any

import torch
import torch.nn as nn

# Allow absolute imports from src when running as a script
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import utils
import data_loader
import train as train_lib
from models import RNNClassifier, BERTClassifier, load_glove_embeddings


def build_model(config: Dict[str, Any], rnn_vocab: Dict[str, int]) -> torch.nn.Module:
    model_type = config.get("model_type", "rnn")
    num_classes = int(config.get("num_classes", 2))
    if model_type in ("rnn", "lstm"):
        embedding_mode = config.get("embedding_mode", "random")
        embedding_dim = int(config.get("embedding_dim", 100))
        hidden_dim = int(config.get("rnn_hidden_dim", 128))
        num_layers = int(config.get("rnn_num_layers", 2))
        bidirectional = bool(config.get("bidirectional", True))
        dropout = float(config.get("dropout", 0.3))
        glove_weights = None
        if embedding_mode.startswith("glove"):
            glove_path = config.get("glove_path")
            if glove_path and not os.path.isabs(glove_path):
                root = os.path.dirname(os.path.dirname(__file__))
                glove_path = os.path.abspath(os.path.join(root, glove_path))
            if glove_path and os.path.exists(glove_path):
                glove_weights = load_glove_embeddings(glove_path, rnn_vocab, embedding_dim)
        model = RNNClassifier(
            vocab_size=len(rnn_vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            num_classes=num_classes,
            embedding_mode=embedding_mode,
            glove_weights=glove_weights,
        )
    else:
        bert_model_name = config.get("bert_model_name", "bert-base-uncased")
        dropout = float(config.get("dropout", 0.1))
        freeze_all = bool(config.get("freeze_all", False))
        unfreeze_layers = int(config.get("unfreeze_layers", 0))
        model = BERTClassifier(
            bert_model_name=bert_model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_all=freeze_all,
            unfreeze_layers=unfreeze_layers,
        )
    return model


def run(config_path: str) -> None:
    config = utils.load_json(config_path)
    seed = int(config.get("seed", 42))
    utils.set_seed(seed)
    device = utils.get_device()
    os.makedirs(os.path.join(os.path.dirname(SRC_DIR), "results"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(SRC_DIR), "models_checkpoints"), exist_ok=True)

    loaders = data_loader.get_dataloaders(config)
    rnn_vocab = loaders.get("vocab", None)
    model = build_model(config, rnn_vocab)
    model = model.to(device)

    learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    num_epochs = int(config.get("num_epochs", 3))
    gradient_clip = float(config.get("gradient_clip", 1.0))

    # Only optimize trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"seed": seed, "config_path": config_path, "epochs": []}
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_lib.train_epoch(
            model,
            loaders["train_loader"],
            optimizer,
            criterion,
            device,
            gradient_clip=gradient_clip,
        )
        val_metrics = train_lib.evaluate(model, loaders["val_loader"], criterion, device)
        history["epochs"].append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

    # Final evaluation on test set
    test_metrics = train_lib.evaluate(model, loaders["test_loader"], criterion, device)
    history["test"] = test_metrics

    # Save results and checkpoint
    root = os.path.dirname(SRC_DIR)
    model_type = config.get("model_type", "rnn")
    result_path = os.path.join(root, "results", f"results_seed_{seed}_{model_type}.json")
    utils.save_json(history, result_path)

    ckpt_path = os.path.join(root, "models_checkpoints", f"model_seed_{seed}_{model_type}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved results to: {result_path}")
    print(f"Saved checkpoint to: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        # Resolve relative to current working directory
        config_path = os.path.abspath(config_path)
    run(config_path)


if __name__ == "__main__":
    main()