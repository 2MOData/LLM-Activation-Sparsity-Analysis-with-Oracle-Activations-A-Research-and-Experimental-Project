"""Vérifie que l'environnement est correct : `uv run python scripts/check_env.py`."""

import platform

import torch
import transformers

print(f"Python        : {platform.python_version()}")
print(f"torch         : {torch.__version__}")
print(f"transformers  : {transformers.__version__}")
print(f"CUDA dispo    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("GPU           : Apple MPS")
