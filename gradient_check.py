"""Compare manual backpropagation gradients against central finite differences."""
import numpy as np
from numpy_network import NumPyNetwork


def numerical_gradient(model, X, y, param_name, index, epsilon=1e-5):
    parameter = model.params[param_name]
    original = parameter[index].copy()
    try:
        parameter[index] = original + epsilon
        loss_plus = model.mse(model.forward(X), y)
        parameter[index] = original - epsilon
        loss_minus = model.mse(model.forward(X), y)
    finally:
        parameter[index] = original
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def main():
    # Use the required architecture. Fixed, non-boundary inputs make this reproducible.
    model = NumPyNetwork(seed=42)
    rng = np.random.default_rng(123)
    X = rng.normal(size=(4, 10)).astype(np.float64)
    y = rng.normal(size=(4, 1)).astype(np.float64)

    predictions, cache = model.forward(X, return_cache=True)
    analytical = model.backward(y, cache)

    max_abs_error = 0.0
    max_rel_error = 0.0
    epsilon = 1e-5

    for name, parameter in model.params.items():
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            numerical[index] = numerical_gradient(model, X, y, name, index, epsilon)

        difference = np.abs(analytical[name] - numerical)
        denominator = np.maximum(
            1e-8, np.abs(analytical[name]) + np.abs(numerical)
        )
        relative = difference / denominator
        abs_error = float(np.max(difference))
        rel_error = float(np.max(relative))
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        print(f"{name}: max absolute error={abs_error:.3e}, "
              f"max relative error={rel_error:.3e}")

    print(f"\nOverall maximum absolute error: {max_abs_error:.3e}")
    print(f"Overall maximum relative error: {max_rel_error:.3e}")
    passed = max_abs_error < 1e-6
    print("Gradient check:", "PASSED" if passed else "FAILED")
    if not passed:
        raise SystemExit("Gradient check failed; debug before training.")


if __name__ == "__main__":
    main()
