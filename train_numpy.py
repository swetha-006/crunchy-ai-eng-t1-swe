"""Train the NumPy network on the Diabetes regression dataset."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy_network import NumPyNetwork


def main():
    results = Path("results")
    results.mkdir(exist_ok=True)

    dataset = load_diabetes()
    X_all = dataset.data.astype(np.float64)
    y_all = dataset.target.reshape(-1, 1).astype(np.float64)

    X_trainval_raw, X_test_raw, y_trainval_raw, y_test_raw = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_trainval_raw, y_trainval_raw, test_size=0.2, random_state=43
    )

    # Fit preprocessing only on training data. Keep test data untouched until final evaluation.
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_train = (y_train_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std

    model = NumPyNetwork(input_size=X_train.shape[1], seed=42)
    rng = np.random.default_rng(42)
    epochs, batch_size, learning_rate = 300, 32, 0.005
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        order = rng.permutation(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            pred, cache = model.forward(xb, return_cache=True)
            grads = model.backward(yb, cache)
            model.step(grads, learning_rate)

        train_loss = model.mse(model.forward(X_train), y_train)
        val_loss = model.mse(model.forward(X_val), y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | train MSE={train_loss:.4f} "
                  f"| validation MSE={val_loss:.4f}")

    pred_scaled = model.forward(X_test)
    pred = pred_scaled * y_std + y_mean
    test_mse = float(np.mean((pred - y_test_raw) ** 2))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(np.mean(np.abs(pred - y_test_raw)))

    plot_path = results / "learning_curve.png"
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Training MSE")
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardized targets)")
    plt.title("NumPy Network Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    model_path = results / "numpy_model.npz"
    model.save(
        model_path,
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_mean,
        y_std=y_std,
        test_mse=np.array(test_mse),
        test_rmse=np.array(test_rmse),
        test_mae=np.array(test_mae),
    )

    report = (
        f"Test MSE (original target units): {test_mse:.4f}\n"
        f"Test RMSE (original target units): {test_rmse:.4f}\n"
        f"Test MAE (original target units): {test_mae:.4f}\n"
        f"Epochs: {epochs}\nBatch size: {batch_size}\n"
        f"Train samples: {len(X_train)}\nValidation samples: {len(X_val)}\n"
        f"Test samples: {len(X_test)}\n"
        f"Learning rate: {learning_rate}\n"
    )
    (results / "training_results.txt").write_text(report)
    print("\n" + report)
    print(f"Saved model: {model_path}")
    print(f"Saved learning curve: {plot_path}")


if __name__ == "__main__":
    main()
