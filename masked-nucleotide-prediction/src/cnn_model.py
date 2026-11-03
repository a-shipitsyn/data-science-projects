"""Small 1D convolutional network on the one-hot context window (PyTorch, CPU).

Input: 5 channels (A, C, G, T, unknown) x 61 positions. Two convolution layers see local motifs,
a dilated one widens the view, and a dense layer on the flattened map predicts the centre base.
Training draws fresh random examples from the training genes at every step, so the network
never sees the same example twice.
"""

import numpy as np
import torch
from torch import nn

from genome import Genome
from windows import WIDTH, sample_examples


class ContextCNN(nn.Module):
    def __init__(self, channels: int = 64, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(5, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * WIDTH, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, x):
        return self.head(self.body(x))


def one_hot(windows: np.ndarray) -> torch.Tensor:
    """(n, WIDTH) codes 0-4 -> (n, 5, WIDTH) float tensor."""
    eye = np.eye(5, dtype=np.float32)
    return torch.from_numpy(eye[windows].transpose(0, 2, 1).copy())


class CNNModel:
    def __init__(self, steps: int = 6000, batch: int = 1024, lr: float = 2e-3, seed: int = 0):
        self.steps, self.batch, self.lr, self.seed = steps, batch, lr, seed
        torch.manual_seed(seed)
        self.net = ContextCNN()
        self.history: list[tuple[int, float]] = []

    def fit(self, genome: Genome, val: tuple[np.ndarray, np.ndarray] | None = None,
            eval_every: int = 500) -> "CNNModel":
        """Adam with cosine learning-rate decay; keeps the weights with the best validation log-loss."""
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, self.steps)
        loss_fn = nn.CrossEntropyLoss()
        best, best_state = np.inf, None
        for step in range(1, self.steps + 1):
            windows, targets, _ = sample_examples(genome, self.batch, seed=self.seed * 1_000_003 + step)
            self.net.train()
            optimiser.zero_grad()
            loss = loss_fn(self.net(one_hot(windows)), torch.from_numpy(targets.astype(np.int64)))
            loss.backward()
            optimiser.step()
            schedule.step()
            if val is not None and (step % eval_every == 0 or step == self.steps):
                proba = self.predict_proba(val[0])
                val_loss = -np.mean(np.log(proba[np.arange(len(val[1])), val[1]] + 1e-12))
                self.history.append((step, val_loss))
                if val_loss < best:
                    best, best_state = val_loss, {k: v.clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, windows: np.ndarray, batch: int = 8192) -> np.ndarray:
        self.net.eval()
        parts = [torch.softmax(self.net(one_hot(windows[i:i + batch])), dim=1).numpy()
                 for i in range(0, len(windows), batch)]
        return np.concatenate(parts)

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> "CNNModel":
        self.net.load_state_dict(torch.load(path))
        return self
