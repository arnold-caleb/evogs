# DyNeRF Dataset Configurations

This directory contains configuration files for training on DyNeRF dataset scenes. DyNeRF consists of real-world dynamic scenes captured with iPhone cameras.

## Structure

The configs are organized using **inheritance** to minimize code duplication:

```
dynerf/
├── base.py                    # Base config for all DyNeRF scenes
├── base_velocity.py          # Base config for velocity field (EvoGS) models
├── base_displacement.py      # Base config for displacement field (4DGaussians) models
├── base_sparse.py            # Base config for sparse supervision experiments
│
├── {scene}.py                # Scene-specific configs (inherit from base*.py)
├── {scene}_velocity.py       # Scene with velocity field
├── {scene}_displacement.py   # Scene with displacement field
└── ...                       # Various ablations and experiments
```

## Base Configurations

### `base.py`
Common settings for all DyNeRF scenes:
- HexPlane grid resolution: [64, 64, 64, 150]
- 14K iterations, batch size 4
- Standard densification schedule

### `base_velocity.py` (EvoGS)
Velocity field settings for continuous dynamics:
- Enables Neural ODE integration
- Freezes opacity (prevents holes)
- Minimal Gaussian updates (velocity field handles motion)
- Velocity coherence regularization

### `base_displacement.py`
Displacement field settings (4DGaussians baseline):
- Direct Δx = f(x,t) prediction (no ODE)
- Shorter training (3K iterations)
- Conservative densification

### `base_sparse.py`
Sparse temporal supervision:
- Trains on every 3rd frame
- Position-only motion
- Spatial coherence loss
- No densification (maintain correspondence)

## Scene Configs

### Standard Scenes
- `cut_roasted_beef.py` - Hand cutting meat
- `coffee_martini.py` - Pouring coffee
- `cook_spinach.py` - Cooking spinach
- `flame_salmon_1.py` - Flaming salmon
- `flame_steak.py` - Flaming steak
- `sear_steak.py` - Searing steak

### Velocity Field Variants
- `{scene}_velocity.py` - Standard velocity field
- `{scene}_velocity_xyz_only.py` - Position-only (no scale/rotation evolution)
- `{scene}_velocity_coherence.py` - Enhanced coherence regularization
- `{scene}_velocity_strong_coh.py` - Very strong coherence
- `{scene}_velocity_regularized.py` - Additional velocity magnitude regularization
- `{scene}_velocity_optimized.py` - Tuned hyperparameters
- `{scene}_velocity_canonical.py` - Query at canonical position
- `{scene}_velocity_init_from_static.py` - Initialize from static checkpoint
- `{scene}_velocity_rk4_a100.py` - RK4 integration for 40GB GPU
- `{scene}_velocity_rk4_l40.py` - RK4 integration for 24GB GPU

### Sparse Supervision Variants
- `{scene}_sparse.py` - Sparse supervision
- `{scene}_sparse_stride3.py` - Stride 3 sparse supervision
- `{scene}_sparse_stride3_adaptive.py` - With adaptive coherence
- `{scene}_sparse_displacement.py` - Sparse with displacement field
- `{scene}_sparse_stride3_displacement.py` - Stride 3 with displacement

### Multi-Anchor Experiments
- `{scene}_multi_anchor.py` - 3 anchors (t=0, 0.5, 1.0)
- `{scene}_4anchors.py` - 4 anchors with forward-only integration
- `{scene}_future_velocity.py` - Future reconstruction (train on first half)
- `{scene}_future_velocity_adaptive.py` - Future with adaptive coherence
- `{scene}_future_displacement.py` - Future with displacement field

### Other Variants
- `{scene}_hq.py` - High quality (longer training)
- `{scene}_static_only.py` - Static only (no deformation)
- `{scene}_no_hexplane.py` - MLP-only deformation
- `{scene}_original.py` - Original 4DGaussians configuration

## Usage

### Basic Training
```bash
python train.py \
  --source_path data/dynerf/cut_roasted_beef \
  --model_path output/dynerf_velocity/cut_roasted_beef \
  --configs arguments/dynerf/cut_roasted_beef_velocity.py
```

### Using GitHub Scripts
```bash
./scripts/train_velocity_field.sh cut_roasted_beef arguments/dynerf/cut_roasted_beef_velocity.py
```

### Sparse Supervision
```bash
./scripts/train_with_sparse_supervision.sh cut_roasted_beef arguments/dynerf/cut_roasted_beef_sparse_stride3.py
```

## Creating New Configs

To create a new scene or variant:

1. Choose the appropriate base:
   - `base.py` - Standard hexplane
   - `base_velocity.py` - Velocity field (EvoGS)
   - `base_displacement.py` - Displacement field
   - `base_sparse.py` - Sparse supervision

2. Create a new config file:
```python
"""Your scene description"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # Override specific parameters
)

OptimizationParams = dict(
    # Override specific parameters
)
```

3. Only specify parameters that differ from the base!

## Tips

- **Start with `{scene}_velocity.py`** for standard EvoGS training
- **Use `xyz_only` variants** if you see artifacts in scale/rotation
- **Try RK4 integration** if you have sufficient GPU memory (much better accuracy)
- **Sparse supervision** tests generalization (harder, but proves the model learns dynamics)
- **Multi-anchor** reduces integration drift for long sequences

