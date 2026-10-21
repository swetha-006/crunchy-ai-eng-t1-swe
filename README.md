# AI Engineering Task 1: Neural Network from Scratch

## Overview

This project implements a feed-forward neural network from scratch using NumPy and compares it with an equivalent PyTorch implementation.

The goal is to understand the internal mechanics of neural networks, including forward propagation, loss calculation, backpropagation, gradient descent, and numerical gradient verification.

## Network Architecture

- **Input layer:** 10 features
- **Hidden layer 1:** 16 neurons, ReLU activation
- **Hidden layer 2:** 8 neurons, ReLU activation
- **Output layer:** 1 neuron
- **Loss function:** Mean Squared Error (MSE)
- **Optimization:** Mini-batch gradient descent

## Dataset

The model uses the Diabetes dataset provided by scikit-learn for a regression task. The dataset is divided into training, validation, and test sets. Standardization is applied using statistics fitted on the training data.

## Implementation

### NumPy neural network
The NumPy implementation includes:
- Weight and bias initialization
- Forward propagation
- ReLU activation
- Mean Squared Error loss
- Manual backpropagation using the chain rule
- Mini-batch gradient descent
- Numerical gradient checking using finite differences

### PyTorch comparison
An equivalent network is implemented in PyTorch. The trained NumPy parameters are transferred to the PyTorch model to compare their outputs.

## Verification Results

- **Gradient checking:** Passed
- **Maximum absolute gradient error:** `2.643e-11`
- **NumPy/PyTorch output equivalence:** Passed
- **Maximum absolute output difference:** `2.220e-16`

These checks support the correctness of the gradient calculations and the equivalence of the two model implementations for the tested parameters and inputs.

## Training Results

Training completed for 300 epochs with a batch size of 32 and a learning rate of 0.005.

| Metric | Result |
|---|---:|
| Training samples | 282 |
| Validation samples | 71 |
| Test samples | 89 |
| Final training MSE (standardized target) | 0.3731 |
| Final validation MSE (standardized target) | 0.7096 |
| Test RMSE (original target units) | 53.7698 |
| Test MAE (original target units) | 42.3692 |

The validation MSE starts increasing while the training MSE continues decreasing, suggesting possible overfitting. The learning curve is saved at `results/learning_curve.png`.

## Benchmark

CPU benchmark results for forward and backward computation, using batch size 32 and float64:

| Implementation | Time per iteration |
|---|---:|
| NumPy manual forward and backward | 0.0441 ms |
| PyTorch autograd forward and backward | 0.1850 ms |

These timings are specific to the test environment and may vary with hardware, software versions, and system load.

## Project Structure

```text
.
├── numpy_network.py
├── pytorch_network.py
├── gradient_check.py
├── train_numpy.py
├── compare_models.py
├── benchmark.py
├── environment_check.py
├── requirements.txt
├── MATH.md
└── results/
    ├── training_results.txt
    ├── learning_curve.png
    ├── model_comparison.txt
    ├── benchmark_results.txt
    ├── numpy_model.npz
    └── pytorch_model.pt
```

## Setup

Python 3.12 was used to test this project.

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Project

Check the environment:

```powershell
python environment_check.py
```

Verify manual gradients:

```powershell
python gradient_check.py
```

Train the NumPy model:

```powershell
python train_numpy.py
```

Compare NumPy and PyTorch outputs:

```powershell
python compare_models.py
```

Benchmark both implementations:

```powershell
python benchmark.py
```

## Mathematical Derivation

See [`MATH.md`](MATH.md) for the mathematical derivation of forward propagation, the loss function, backpropagation, and parameter updates.

## Limitations

This is an educational neural network implementation. The benchmark is a small CPU experiment and should not be interpreted as a general performance comparison between NumPy and PyTorch. The model may benefit from early stopping or other regularization techniques.

## License

Add a license if you intend to distribute this project publicly.
