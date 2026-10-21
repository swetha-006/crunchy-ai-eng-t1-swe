"""A fully connected neural network implemented in NumPy, without autograd."""
import numpy as np


class NumPyNetwork:
    """10 -> 16 ReLU -> 8 ReLU -> 1 regression output by default."""

    def __init__(self, input_size=10, hidden1=16, hidden2=8, seed=42):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.standard_normal((input_size, hidden1)) * np.sqrt(2.0 / input_size),
            "b1": np.zeros((1, hidden1), dtype=np.float64),
            "W2": rng.standard_normal((hidden1, hidden2)) * np.sqrt(2.0 / hidden1),
            "b2": np.zeros((1, hidden2), dtype=np.float64),
            "W3": rng.standard_normal((hidden2, 1)) * np.sqrt(2.0 / hidden2),
            "b3": np.zeros((1, 1), dtype=np.float64),
        }
        for key in self.params:
            self.params[key] = self.params[key].astype(np.float64, copy=False)

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def mse(y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def forward(self, X, return_cache=False):
        X = np.asarray(X, dtype=np.float64)
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = self.relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = self.relu(z2)
        y_pred = a2 @ p["W3"] + p["b3"]
        if return_cache:
            return y_pred, (X, z1, a1, z2, a2, y_pred)
        return y_pred

    def backward(self, y_true, cache):
        """Return analytical gradients for mean squared error; no autograd."""
        X, z1, a1, z2, a2, y_pred = cache
        p = self.params
        y_true = np.asarray(y_true, dtype=np.float64)
        n = y_true.shape[0]

        dz3 = (2.0 / y_true.size) * (y_pred - y_true)
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ p["W3"].T
        dz2 = da2 * (z2 > 0.0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (z1 > 0.0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
        }

    def step(self, grads, learning_rate):
        for name in self.params:
            self.params[name] -= learning_rate * grads[name]

    def save(self, path, **metadata):
        """Save parameters plus optional numeric metadata to an NPZ file."""
        payload = {**self.params}
        payload.update(metadata)
        np.savez(path, **payload)

    def load_parameters(self, path):
        with np.load(path) as data:
            for name in self.params:
                self.params[name][...] = data[name]
