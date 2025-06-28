#!/bin/bash

module load cuda/12.5

export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5
export CUDA_HOME=$NVHPC_ROOT/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

source /home/ignacio.alesina/py3.12_pytorch2/bin/activate

echo "=== Python and CUDA Environment ==="
python -c "
import torch
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

set -x
srun python v8.py
