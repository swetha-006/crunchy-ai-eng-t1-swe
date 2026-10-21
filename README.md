# Neural Network from Scratch: NumPy vs PyTorch

A two-hidden-layer regression network implemented manually in NumPy and reproduced in PyTorch.

## Architecture

- Input: 10 features
- Hidden layer 1: 16 units, ReLU
- Hidden layer 2: 8 units, ReLU
- Output: 1 regression value
- Loss: mean squared error
- Dataset: scikit-learn Diabetes dataset

The NumPy implementation does not use autograd. PyTorch autograd is used only in the PyTorch implementation and benchmark.

## Environment

Use Python 3.11 in a virtual environment or Google Colab. `requirements.txt` pins the direct dependencies. Run `python environment_check.py` and save its output as environment evidence. A GPU is optional for this small model; report honestly whether CUDA was available.

## Install

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run in order

```bash
python environment_check.py
python gradient_check.py
python train_numpy.py
python compare_models.py
python benchmark.py
```

Run `gradient_check.py` before training. It checks every weight and bias against central finite differences and exits with an error if the maximum absolute error is at least `1e-6`.

`train_numpy.py` saves the trained weights and learning curve in `results/`.
`compare_models.py` copies the trained NumPy parameters into `nn.Linear` layers and checks matching outputs.
`benchmark.py` measures forward plus backward computation on CPU for both implementations. Actual timings depend on the machine and software environment.

## Results

After running the scripts, include the generated output files from `results/`:
- `learning_curve.png`
- `training_results.txt`
- `model_comparison.txt`
- `benchmark_results.txt`
- `numpy_model.npz`
- `pytorch_model.pt`

Do not claim results until the scripts have actually run. Add your environment output and measured gradient error to your final report.

## Deliverables

- `numpy_network.py`: forward pass, MSE, manual backpropagation, gradient descent
- `gradient_check.py`: numerical gradient verification
- `train_numpy.py`: real-dataset training and learning curve
- `pytorch_network.py`: equivalent PyTorch architecture
- `compare_models.py`: output equivalence check
- `benchmark.py`: measured runtime comparison
- `MATH.md`: backpropagation derivation
