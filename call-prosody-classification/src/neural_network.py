"""Deep neural network (PyTorch MLP) for binary classification:
    - weighted BCE loss because of target imbalance
    - evaluation with stratified cross-validation, F1-score

Usage:
    python neural_network.py --input clean_data.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryF1Score

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Deep(nn.Module):
    def __init__(self, n_features, n_hidden=60, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)  # logits


def model_train(model, X_train, y_train, batch_size=16, n_epochs=10, lr=1e-4, verbose=True):
    if verbose:
        print("\nTraining Deep model:")

    # weight positives by count of negatives / count of positives
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")


def model_predict_labels(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
    return (logits >= 0).squeeze(1).int().cpu().numpy()


def model_evaluate(X, y, cv_folds=4, batch_size=16):
    print("\nEvaluation with cross-validation:")
    f1_metric = BinaryF1Score()
    cv_scores = []
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(kfold.split(X, y), 1):
        model = Deep(X.shape[1]).to(device)
        model_train(model, X[train], y[train], batch_size, verbose=False)

        y_pred = model_predict_labels(model, X[test])
        f1 = f1_metric(torch.tensor(y_pred), torch.tensor(y[test])).item()
        print(f"CV-fold {fold} F1-score: {f1:.4f}")
        cv_scores.append(f1)

    print("\nCross validation F1-score:")
    print(f"\tDeep model: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})\n")
    return cv_scores


def main():
    ap = argparse.ArgumentParser(description="PyTorch MLP on clean call features")
    ap.add_argument("--input", default="clean_data.csv", help="clean data CSV from preprocess.py")
    ap.add_argument("--folds", type=int, default=4, help="cross-validation folds")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X = pd.read_csv(args.input)
    y = X.pop("label").values
    X = X.values

    model_evaluate(X, y, cv_folds=args.folds)

    # train final model on all the data
    # raw (unscaled) features: scaling gave lower CV F1 with this short training schedule
    model = Deep(X.shape[1]).to(device)
    model_train(model, X, y)
    print("\nConfusion matrix, train set:")
    print(confusion_matrix(y, model_predict_labels(model, X)))


if __name__ == "__main__":
    main()
