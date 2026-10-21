"""Compare equivalent forward+backward workloads on CPU; timings are measurements, not claims."""
from pathlib import Path
import time
import numpy as np
import torch
from numpy_network import NumPyNetwork
from pytorch_network import TorchNetwork


def time_callable(fn, warmup=20, repeats=100):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000.0


def main():
    path = Path("results/numpy_model.npz")
    if not path.exists():
        raise FileNotFoundError("Run `python train_numpy.py` first.")

    rng = np.random.default_rng(7)
    X = rng.normal(size=(32, 10)).astype(np.float64)
    y = rng.normal(size=(32, 1)).astype(np.float64)

    numpy_model = NumPyNetwork(seed=42)
    numpy_model.load_parameters(path)
    def numpy_work():
        pred, cache = numpy_model.forward(X, return_cache=True)
        grads = numpy_model.backward(y, cache)
        # Do not update parameters; compare forward/backward only.
        return pred, grads

    torch_model = TorchNetwork()
    with np.load(path) as saved, torch.no_grad():
        torch_model.fc1.weight.copy_(torch.from_numpy(saved["W1"].T.copy()))
        torch_model.fc1.bias.copy_(torch.from_numpy(saved["b1"].ravel().copy()))
        torch_model.fc2.weight.copy_(torch.from_numpy(saved["W2"].T.copy()))
        torch_model.fc2.bias.copy_(torch.from_numpy(saved["b2"].ravel().copy()))
        torch_model.fc3.weight.copy_(torch.from_numpy(saved["W3"].T.copy()))
        torch_model.fc3.bias.copy_(torch.from_numpy(saved["b3"].ravel().copy()))

    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    criterion = torch.nn.MSELoss()

    def torch_work():
        torch_model.zero_grad(set_to_none=True)
        pred = torch_model(xt)
        loss = criterion(pred, yt)
        loss.backward()
        return pred

    # CPU-only benchmark for an apples-to-apples device comparison.
    numpy_ms = time_callable(numpy_work)
    torch_ms = time_callable(torch_work)
    ratio = numpy_ms / torch_ms if torch_ms else float("nan")

    report = (
        "Forward + backward benchmark (CPU, batch size 32, float64)\n"
        "Warmup iterations: 20; timed iterations: 100\n"
        "Times are per iteration and include backward computation.\n"
        f"NumPy manual forward+backward: {numpy_ms:.4f} ms\n"
        f"PyTorch autograd forward+backward: {torch_ms:.4f} ms\n"
        f"NumPy/PyTorch time ratio: {ratio:.3f}x\n"
        "Interpretation: a ratio above 1 means NumPy took longer; below 1 means "
        "PyTorch took longer. Results depend on hardware, library builds and load.\n"
    )
    print(report)
    Path("results/benchmark_results.txt").write_text(report)


if __name__ == "__main__":
    main()
