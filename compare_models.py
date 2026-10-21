"""Load trained NumPy parameters into PyTorch and compare forward outputs."""
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from pytorch_network import TorchNetwork


def main():
    model_path = Path("results/numpy_model.npz")
    if not model_path.exists():
        raise FileNotFoundError("Run `python train_numpy.py` first.")

    data = load_diabetes()
    X_all = data.data.astype(np.float64)
    _, X_test_raw, _, _ = train_test_split(
        X_all, data.target, test_size=0.2, random_state=42
    )

    with np.load(model_path) as saved:
        X_test = (X_test_raw - saved["x_mean"]) / saved["x_scale"]
        W1, b1 = saved["W1"], saved["b1"]
        W2, b2 = saved["W2"], saved["b2"]
        W3, b3 = saved["W3"], saved["b3"]

    # Reconstruct NumPy predictions directly from saved weights.
    a1 = np.maximum(0.0, X_test @ W1 + b1)
    a2 = np.maximum(0.0, a1 @ W2 + b2)
    numpy_output = a2 @ W3 + b3

    torch_model = TorchNetwork()
    with torch.no_grad():
        torch_model.fc1.weight.copy_(torch.from_numpy(W1.T.copy()))
        torch_model.fc1.bias.copy_(torch.from_numpy(b1.ravel().copy()))
        torch_model.fc2.weight.copy_(torch.from_numpy(W2.T.copy()))
        torch_model.fc2.bias.copy_(torch.from_numpy(b2.ravel().copy()))
        torch_model.fc3.weight.copy_(torch.from_numpy(W3.T.copy()))
        torch_model.fc3.bias.copy_(torch.from_numpy(b3.ravel().copy()))

    torch_model.eval()
    with torch.no_grad():
        torch_output = torch_model(torch.from_numpy(X_test)).cpu().numpy()

    max_diff = float(np.max(np.abs(numpy_output - torch_output)))
    tolerance = 1e-10
    passed = np.allclose(numpy_output, torch_output, atol=tolerance, rtol=1e-10)

    print("NumPy first five outputs:", numpy_output[:5].ravel())
    print("PyTorch first five outputs:", torch_output[:5].ravel())
    print(f"Maximum absolute difference: {max_diff:.3e}")
    print(f"Tolerance: atol={tolerance:.1e}, rtol=1.0e-10")
    print("NumPy/PyTorch equivalence:", "PASSED" if passed else "FAILED")

    torch.save(torch_model.state_dict(), "results/pytorch_model.pt")
    with open("results/model_comparison.txt", "w") as f:
        f.write(f"Maximum absolute difference: {max_diff:.12e}\n")
        f.write(f"Tolerance: atol={tolerance:.1e}, rtol=1.0e-10\n")
        f.write(f"Equivalence: {'PASSED' if passed else 'FAILED'}\n")
    if not passed:
        raise SystemExit("Model outputs did not match within tolerance.")


if __name__ == "__main__":
    main()
