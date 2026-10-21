"""Report environment versions and whether CUDA is available."""
import sys
import numpy as np
import torch
import matplotlib
import sklearn

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU. CPU is sufficient for this small project.")
