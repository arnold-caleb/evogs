## `EvoGS`: 4D Gaussian Splatting as a Learned Dynamical System

<p >
  <strong><a href="https://cs.princeton.edu/~aa0008">Arnold Caleb Asiimwe</a></strong><sup>1</sup>,&nbsp; <strong><a href="https:/cs.columbia.edu/~vondrick">Carl Vondrick</a></strong><sup>2</sup>
  <br>
  <sup>1</sup>Princeton University, &nbsp; <sup>2</sup>Columbia University
  <br>
  <a href="https://arnold-caleb.github.io/evogs"><img src="https://img.shields.io/badge/🌐_Project-Page-blue?style=flat" alt="Project Page"></a>
  <a href="https://arxiv.org/pdf/2512.19648"><img src="https://img.shields.io/badge/📄_arXiv-Paper-red?style=flat" alt="arXiv Paper"></a>
  <a href="https://github.com/arnold-caleb/evogs"><img src="https://img.shields.io/badge/💻_GitHub-Code-black?style=flat" alt="GitHub Code"></a>
</p>

<p >
  <img src="assets/overview.png" alt="EvoGS Overview" width="100%">
  <em>We learn a continuous-time dynamical system that governs Gaussian primitive evolution through neural velocity fields and ODE integration.</em>
</p>


## Abstract

We reinterpret 4D Gaussian Splatting as a continuous-time dynamical system, where scene motion arises from integrating a learned neural velocity field rather than applying per-frame deformations. This formulation, which we call `EvoGS`, treats the Gaussian representation as an evolving physical system whose state evolves continuously under a learned motion law. 
This unlocks capabilities absent in deformation-based approaches:
1. **Sample-efficient learning** from sparse temporal supervision by modeling the underlying motion law
2. **Temporal extrapolation** enabling forward and backward prediction beyond observed time ranges
3. **Compositional dynamics** that allow localized dynamics injection for controllable scene synthesis

<!-- Experiments on dynamic scene benchmarks show that `EvoGS` achieves better motion coherence and temporal consistency compared to deformation-field baselines while maintaining real-time rendering. -->

---

## 🔥 Highlights

- **Continuous-time dynamics**: Scene motion governed by Neural ODEs, not discrete deformations
- **Sparse supervision**: Learn from every 3rd frame and interpolate the rest
- **Temporal extrapolation**: Predict motion beyond the training time range
- **Real-time rendering**: Maintains 100+ FPS rendering speeds
- **Modular implementation**: Clean, well-documented codebase with 60+ configuration presets

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration System](#configuration-system)
- [Code Structure](#code-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- conda or virtualenv

### Step 1: Clone Repository
```bash
git clone https://github.com/arnold-caleb/evogs.git
cd evogs
git submodule update --init --recursive
```

### Step 2: Create Environment
```bash
conda env create -f environment.yml
conda activate evogs
```

### Step 3: Install Submodules
```bash
# Install differential Gaussian rasterization
pip install submodules/diff-gaussian-rasterization

# Install simple-knn
pip install submodules/simple-knn
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

---

## 🚀 Quick Start

### Train `EvoGS` on D-NeRF Lego Scene

```bash
# 1. Download D-NeRF dataset
mkdir -p data/dnerf
cd data/dnerf
# Download from: https://github.com/albertpumarola/D-NeRF
# Extract lego scene to data/dnerf/lego/

# 2. Train with velocity field (`EvoGS`)
./scripts/train_velocity_field.sh lego arguments/dnerf/lego_velocity.py

# 3. Render results
python render.py \
  --model_path output/dnerf_velocity/lego_YYYYMMDD_HHMMSS \
  --configs arguments/dnerf/lego_velocity.py
```

### Train with Sparse Supervision

```bash
# Train on every 3rd frame, interpolate the rest
./scripts/train_with_sparse_supervision.sh lego arguments/dnerf/lego_velocity.py
```

---

## 📂 Dataset Preparation

`EvoGS` supports three dynamic scene benchmarks:

### D-NeRF (Synthetic)
```bash
# Download from: https://github.com/albertpumarola/D-NeRF
# Scenes: lego, mutant, trex, jumpingjacks, standup, hellwarrior, hook, bouncingballs
mkdir -p data/dnerf
# Extract to: data/dnerf/{scene_name}/
```

### DyNeRF (Real-world)
```bash
# Download from: https://github.com/USTC3DV/DyNeRF-Dataset
# Scenes: cut_roasted_beef, coffee_martini, flame_salmon, cook_spinach, sear_steak, flame_steak
mkdir -p data/dynerf
# Extract to: data/dynerf/{scene_name}/
```

### HyperNeRF (Deformable)
```bash
# Download from: https://github.com/google/hypernerf
# Format: Nerfies dataset format
mkdir -p data/hypernerf
# Extract to: data/hypernerf/{scene_name}/
```

---

## 🎯 Training

### Basic Training

```bash
python train.py \
  --source_path data/dnerf/lego \
  --model_path output/evogs_lego \
  --configs arguments/dnerf/lego_velocity.py \
  --iterations 7000
```

### Training Modes

#### 1. Standard Velocity Field (`EvoGS`)
```bash
./scripts/train_velocity_field.sh lego arguments/dnerf/lego_velocity.py
```

#### 2. Sparse Temporal Supervision
Train on every 3rd frame to test generalization:
```bash
./scripts/train_with_sparse_supervision.sh lego arguments/dnerf/lego_velocity.py
```

#### 3. Displacement Field (4DGaussians Baseline)
```bash
python train.py \
  --source_path data/dynerf/cut_roasted_beef \
  --model_path output/displacement/cut_roasted_beef \
  --configs arguments/dynerf/cut_roasted_beef_displacement.py
```

### Training from Static Checkpoint

For better initialization, train a static frame first:
```bash
# 1. Train static frame 0
python train.py \
  --source_path data/dynerf/cut_roasted_beef \
  --model_path output/static/cut_roasted_beef \
  --configs arguments/static/cut_roasted_beef_frame0.py

# 2. Train velocity field from checkpoint
python train.py \
  --source_path data/dynerf/cut_roasted_beef \
  --model_path output/velocity/cut_roasted_beef \
  --configs arguments/dynerf/cut_roasted_beef_velocity.py \
  --start_checkpoint output/static/cut_roasted_beef/chkpnt30000.pth
```

---

## 📊 Evaluation

### Render Test Views
```bash
python render.py \
  --model_path output/evogs_lego/lego_YYYYMMDD_HHMMSS \
  --configs arguments/dnerf/lego_velocity.py \
  --skip_train  # Only render test views
```

### Compute Metrics
```bash
./scripts/evaluate_model.sh output/evogs_lego/lego_YYYYMMDD_HHMMSS
```

Metrics computed:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)

---

## ⚙️ Configuration System

`EvoGS` uses a clean, inheritance-based configuration system that eliminates code duplication.

### Configuration Structure

```
arguments/
├── dnerf/                   # D-NeRF dataset (synthetic)
│   ├── base.py             # Base config for all D-NeRF scenes
│   ├── base_velocity.py    # `EvoGS` velocity field defaults
│   ├── lego.py             # Lego scene (2 lines!)
│   ├── lego_velocity.py    # Lego with velocity field (1 line!)
│   └── README.md           # Full documentation
│
├── dynerf/                  # DyNeRF dataset (real-world)
│   ├── base.py             # Base config
│   ├── base_velocity.py    # Velocity field defaults
│   ├── base_displacement.py # Displacement field defaults
│   ├── base_sparse.py      # Sparse supervision defaults
│   └── README.md           # Full documentation
│
└── hypernerf/               # HyperNeRF dataset
    ├── base.py             # Base config (nerfies format)
    └── README.md           # Full documentation
```

### Example: Creating a New Scene Config

```python
# arguments/dnerf/my_scene_velocity.py
"""My custom scene with velocity field"""
_base_ = './base_velocity.py'

# Only override what's different!
OptimizationParams = dict(
    iterations = 10000,  # Longer training
)
```

That's it! All other parameters inherit from `base_velocity.py`.

See `arguments/dnerf/README.md` for complete documentation.

---

## 🏗️ Code Structure

### Core Implementation

```
scene/velocity/              # `EvoGS` velocity field (our contribution!)
├── __init__.py             # Public API
├── field.py                # VelocityField neural network architecture
├── integration.py          # ODEIntegrator with RK4/Euler methods
├── network.py              # velocity_network wrapper
├── quaternion_utils.py     # Quaternion kinematics for rotations
└── utils.py                # Utilities (divergence, rollout, etc.)

scene/                       # Gaussian Splatting scene
├── gaussian_model.py       # 3D Gaussian primitives
├── deformation.py          # Deformation/velocity networks
├── hexplane.py             # HexPlane grid feature encoding
├── dataset_readers.py      # Data loaders for all datasets
└── sparse_temporal_sampler.py  # Sparse supervision sampler

train.py                     # Main training script
render.py                    # Rendering script
gaussian_renderer/           # Differentiable rasterization
utils/                       # General utilities
```

### Key Features of the Velocity Field

**Continuous-time dynamics:**
```python
# scene/velocity/field.py
def forward(self, x, t):
    """Compute velocity v(x,t) at position x and time t"""
    return self.velocity_network(x, t)

# scene/velocity/integration.py
def integrate(self, x0, t_span, method='rk4'):
    """Integrate from x0 over t_span using Neural ODE"""
    return odeint(self.ode_func, x0, t_span, method=method)
```

**Dual rotation modes:**
- Standard: Predict quaternion directly
- Angular velocity (geometrically correct): Predict ω, integrate via `dq/dt = 0.5 * q ⊗ [0, ω]`

---

## 📖 Key Concepts

### What is a Velocity Field?

Instead of learning per-frame deformations `Δx_t`, `EvoGS` learns a **velocity field** `v(x,t)` that describes how each point moves:

```
Traditional: x_t = x_0 + Δx_t              (discrete)
`EvoGS`:    dx/dt = v(x,t)                  (continuous)
            x_t = x_0 + ∫₀ᵗ v(x(τ),τ) dτ   (integrated)
```

This formulation:
- ✅ Naturally handles sparse/missing frames
- ✅ Enables temporal extrapolation
- ✅ Enforces motion smoothness
- ✅ Allows compositional dynamics

### Neural ODE Integration

We use `torchdiffeq` for numerical integration:
- **Euler** (fast, 1st order accuracy): Good for quick experiments
- **RK4** (slower, 4th order accuracy): Better for long-range integration

---

## 🔬 Experiments

### Ablation Studies

We provide configs for all ablations in the paper:

```bash
# Sparse supervision (every 3rd frame)
arguments/dynerf/cut_roasted_beef_sparse_stride3.py

# Future reconstruction (train on first 50%, test on last 50%)
arguments/dynerf/cut_roasted_beef_future_velocity.py

# Multi-anchor (reduce integration drift)
arguments/dynerf/cut_roasted_beef_multi_anchor.py

# RK4 integration
arguments/dynerf/cut_roasted_beef_velocity_rk4_a100.py

# XYZ-only (no rotation/scale evolution)
arguments/dynerf/cut_roasted_beef_velocity_xyz_only.py
```

See configuration READMEs for full list.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{asiimwe2025evogs,
  title={EvoGS: 4D Gaussian Splatting as a Learned Dynamical System},
  author={Asiimwe, Arnold Caleb and Vondrick, Carl},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This project builds upon several excellent codebases:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the base renderer
- [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians) for dynamic scene extensions
- [HexPlane](https://github.com/Caoang327/HexPlane) for efficient 4D feature grids
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for Neural ODE integration

We thank the authors of D-NeRF, DyNeRF, and HyperNeRF for releasing their datasets.

---

## 📄 License

This project is licensed under the Gaussian Splatting License. See [LICENSE.md](LICENSE.md) for details.

---

## 🐛 Issues and Questions

- **Issues:** Please open a GitHub issue
- **Questions:** Check the configuration READMEs first, then open an issue
- **Contributions:** Pull requests welcome!

---

## 🔗 Links

- **Project Page:** https://arnold-caleb.github.io/evogs
- **Paper (arXiv):** Coming soon
- **D-NeRF Dataset:** https://github.com/albertpumarola/D-NeRF
- **DyNeRF Dataset:** https://github.com/USTC3DV/DyNeRF-Dataset
- **HyperNeRF Dataset:** https://github.com/google/hypernerf

---

<p align="center">
  Made with ❤️ at Princeton University
  <br>
  <em>"Everything flows" — Heraclitus</em>
</p>
