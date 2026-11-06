#!/bin/bash

CUDNN_LIB=$(python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("nvidia.cudnn")
root = Path(next(iter(spec.submodule_search_locations)))
print(root / "lib")
PY
)

export LD_LIBRARY_PATH="$CUDNN_LIB:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Using cuDNN from:"
echo "$CUDNN_LIB"

echo "LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
