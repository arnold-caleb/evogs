#!/bin/bash
#SBATCH --partition=visualai
#SBATCH --account=visualai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --job-name=cuda12_3dgs_setup
#SBATCH -o slurm_outputs/cuda12_setup_%j.out 
#SBATCH -e slurm_outputs/cuda12_setup_%j.err 
#SBATCH --mail-type=all
#SBATCH --mail-user=aa0008@cs.princeton.edu

echo "=== SETTING UP 3D GAUSSIAN SPLATTING WITH CUDA 12.9 ==="

# Set up conda environment
source ~/miniconda/etc/profile.d/conda.sh

# Set environment variables for cache directories
export TMPDIR=/n/fs/aa-rldiff/tmp
export CONDA_PKGS_DIRS=/n/fs/aa-rldiff/.conda_pkgs
export CONDA_ENVS_PATH=/n/fs/aa-rldiff/.conda_envs
export XDG_CACHE_HOME=/n/fs/aa-rldiff/.cache
export PIP_CACHE_DIR=/n/fs/aa-rldiff/.pip_cache

# Navigate to the project directory
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

echo "=== SYSTEM INFORMATION ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "System CUDA version: $(nvcc --version | grep release)"

echo ""
echo "=== CREATING NEW CUDA 12.9 COMPATIBLE ENVIRONMENT ==="

# Remove old environment if it exists
conda env remove -p /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12 -y 2>/dev/null || true

# Create new environment with Python 3.9 (better CUDA 12.x support)
conda create -p /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12 python=3.9 -y

# Activate the new environment
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

echo ""
echo "=== INSTALLING CUDA 12.x COMPATIBLE PYTORCH ==="

# Install PyTorch with CUDA 12.1 (closest to 12.9, should be compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== VERIFYING PYTORCH CUDA COMPATIBILITY ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== SETTING UP CUDA ENVIRONMENT ==="

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Set compilation flags for CUDA 12.x
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1
export MAX_JOBS=4
export exp_name="dynerf"

echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC version: $(nvcc --version | grep release)"

echo ""
echo "=== INSTALLING BUILD TOOLS ==="

# Install ninja for faster compilation
pip install ninja

# Install other dependencies
pip install plyfile tqdm opencv-python joblib

echo ""
echo "=== COMPILING CUDA EXTENSIONS ==="

# Clean any previous attempts
pip uninstall diff_gaussian_rasterization simple_knn fused_ssim -y 2>/dev/null || true

# Compile and install diff-gaussian-rasterization
echo "Compiling diff-gaussian-rasterization..."
pip install -e submodules/diff-gaussian-rasterization

# Compile and install simple-knn
echo "Compiling simple-knn..."
pip install -e submodules/simple-knn

# Compile and install fused-ssim
echo "Compiling fused-ssim..."
pip install -e submodules/fused-ssim

echo ""
echo "=== TESTING INSTALLATION ==="
python -c "
print('Testing CUDA extensions:')
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print('✓ diff-gaussian-rasterization imported successfully')
except ImportError as e:
    print(f'✗ diff-gaussian-rasterization import failed: {e}')

try:
    from simple_knn import distCUDA2
    print('✓ simple-knn imported successfully')
except ImportError as e:
    print(f'✗ simple-knn import failed: {e}')

try:
    from fused_ssim import SSIM
    print('✓ fused-ssim imported successfully')
except ImportError as e:
    print(f'✗ fused-ssim import failed: {e}')

print('\nTesting PyTorch CUDA functionality:')
import torch
if torch.cuda.is_available():
    # Test basic CUDA operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f'✓ CUDA tensor operations working')
    print(f'✓ GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB')
else:
    print('✗ CUDA not available')
"

echo ""
echo "=== CHECKING COMPILED FILES ==="
echo "Looking for compiled .so files:"
find /n/fs/aa-rldiff/view_synthesis/gaussian-splatting -name "*.so" | grep -E "(diff|simple|fused)"

echo ""
echo "=== FINAL STATUS ==="
echo "✅ 3D Gaussian Splatting setup with CUDA 12.9 complete!"
echo "Environment: /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12"
echo "Project: /n/fs/aa-rldiff/view_synthesis/gaussian-splatting"
echo ""
echo "To activate in future sessions:"
echo "conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12"
echo ""
echo "To test training:"
echo "python train.py --help"
