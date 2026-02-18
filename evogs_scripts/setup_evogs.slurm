#!/bin/bash
#SBATCH --job-name=evogs_setup
#SBATCH --output=slurm_outputs/evogs_setup_%j.out
#SBATCH --error=slurm_outputs/evogs_setup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== SETTING UP EVOGS ENVIRONMENT ==="

source "$CONDA_INIT"

echo "=== SYSTEM INFORMATION ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "System CUDA version: $(nvcc --version | grep release)"

echo ""
echo "=== CREATING CONDA ENVIRONMENT ==="

# Remove old environment if it exists
conda env remove -p "$CONDA_ENV" -y 2>/dev/null || true

# Create new environment
conda create -p "$CONDA_ENV" python=3.9 -y
conda activate "$CONDA_ENV"

echo ""
echo "=== INSTALLING PYTORCH (CUDA 12.1) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== VERIFYING PYTORCH CUDA ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== SETTING UP CUDA ENVIRONMENT ==="
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1
export MAX_JOBS=4

cd "$EVOGS_ROOT"

echo ""
echo "=== INSTALLING BUILD TOOLS ==="
pip install ninja
pip install plyfile tqdm opencv-python joblib

echo ""
echo "=== COMPILING CUDA EXTENSIONS ==="
pip uninstall diff_gaussian_rasterization simple_knn fused_ssim -y 2>/dev/null || true

echo "Compiling diff-gaussian-rasterization..."
pip install -e submodules/diff-gaussian-rasterization

echo "Compiling simple-knn..."
pip install -e submodules/simple-knn

echo "Compiling fused-ssim..."
pip install -e submodules/fused-ssim

echo ""
echo "=== TESTING INSTALLATION ==="
python -c "
print('Testing CUDA extensions:')
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print('✓ diff-gaussian-rasterization OK')
except ImportError as e:
    print(f'✗ diff-gaussian-rasterization failed: {e}')
try:
    from simple_knn import distCUDA2
    print('✓ simple-knn OK')
except ImportError as e:
    print(f'✗ simple-knn failed: {e}')
try:
    from fused_ssim import SSIM
    print('✓ fused-ssim OK')
except ImportError as e:
    print(f'✗ fused-ssim failed: {e}')
import torch
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    y = torch.mm(x, x)
    print(f'✓ CUDA tensor ops OK ({torch.cuda.memory_allocated()/1024**2:.1f} MB)')
"

echo ""
echo "=== SETUP COMPLETE ==="
echo "Environment: $CONDA_ENV"
echo "Project:     $EVOGS_ROOT"
echo ""
echo "To activate:  conda activate $CONDA_ENV"
echo "To test:      python train.py --help"
