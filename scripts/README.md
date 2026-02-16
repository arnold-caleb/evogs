# EvoGS Training Scripts

This directory contains clean, configurable training scripts for EvoGS (Evolving Gaussian Splatting).

## Quick Start

1. **Configure paths**: Copy and edit the configuration file
   ```bash
   cp scripts/config_paths.sh.example scripts/config_paths.sh
   # Edit config_paths.sh to set your dataset and output directories
   ```

2. **Train a velocity field model**:
   ```bash
   ./scripts/train_velocity_field.sh lego arguments/dnerf/lego_velocity.py
   ```

3. **Train with sparse temporal supervision**:
   ```bash
   ./scripts/train_with_sparse_supervision.sh lego arguments/dnerf/lego_velocity.py 2
   ```

4. **Evaluate trained model**:
   ```bash
   ./scripts/evaluate_model.sh output/velocity_field/lego_20250209_143022
   ```

## Scripts

### `train_velocity_field.sh`
Trains a dynamic scene using Neural ODE velocity fields.

**Usage:**
```bash
./scripts/train_velocity_field.sh <dataset_name> <config_file> [options]
```

**Arguments:**
- `dataset_name`: Name of your dataset (e.g., `lego`, `cut_roasted_beef`)
- `config_file`: Path to configuration file (e.g., `arguments/dnerf/lego_velocity.py`)

**Options:**
- `--output_dir DIR`: Custom output directory
- `--iterations N`: Number of training iterations
- `--port PORT`: Visualization server port

**Example:**
```bash
./scripts/train_velocity_field.sh lego arguments/dnerf/lego_velocity.py --iterations 20000
```

### `train_with_sparse_supervision.sh`
Trains with sparse temporal supervision for testing continuous dynamics.

**Usage:**
```bash
./scripts/train_with_sparse_supervision.sh <dataset_name> <config_file> <stride>
```

**Arguments:**
- `stride`: Train on every Nth frame (e.g., 2 = every other frame)

**Example:**
```bash
./scripts/train_with_sparse_supervision.sh lego arguments/dnerf/lego_velocity.py 2
```

This trains on frames 0, 2, 4, 6, ... and interpolates 1, 3, 5, 7, ...

### `evaluate_model.sh`
Evaluates a trained model by rendering test views and computing metrics.

**Usage:**
```bash
./scripts/evaluate_model.sh <model_path> [--skip_train] [--skip_test]
```

**Example:**
```bash
./scripts/evaluate_model.sh output/velocity_field/lego_20250209_143022
```

This will:
- Render training and test views
- Compute PSNR, SSIM, and LPIPS metrics
- Save results to `model_path/results.json`

## Configuration

### Dataset Organization

Expected dataset structure:
```
DATA_ROOT/
├── dnerf/
│   ├── lego/
│   │   ├── poses_bounds.npy
│   │   ├── cam00/
│   │   │   └── images/
│   │   └── ...
│   └── ...
├── dynerf/
│   └── ...
└── hypernerf/
    └── ...
```

### Config Files

Configuration files define model architecture and training hyperparameters:
- `arguments/dnerf/`: Configurations for D-NeRF datasets
- `arguments/dynerf/`: Configurations for DyNeRF datasets
- `arguments/hypernerf/`: Configurations for HyperNeRF datasets

See `arguments/dnerf/base_velocity.py` for an example velocity field configuration.

## Output Structure

Training outputs are saved to:
```
OUTPUT_ROOT/
└── velocity_field/
    └── DATASET_NAME_TIMESTAMP/
        ├── point_cloud/
        │   └── iteration_XXXX/
        │       ├── point_cloud.ply
        │       └── deformation.pth  # Velocity field weights
        ├── cameras.json
        └── cfg_args
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- torchdiffeq (for ODE integration)

See `environment.yml` for full dependencies.

## Cluster Usage

For SLURM clusters, see `slurm_scripts/` for batch job templates.

## Troubleshooting

**Issue: Dataset not found**
- Check that `DATA_ROOT` in `config_paths.sh` points to your dataset directory
- Verify dataset structure matches expected format

**Issue: CUDA out of memory**
- Reduce `batch_size` in config file
- Use Euler integration instead of RK4 (set `ode_method_train='euler'`)
- Enable chunking for large scenes

**Issue: Training diverges**
- Reduce learning rate in config file
- Check that canonical model (stage 1) converged properly
- Verify velocity field activates after sufficient densification

## Citation

If you use this code, please cite:

```bibtex
@article{evogs2025,
  title={EvoGS: Dynamic Scene Modeling with Evolving Gaussian Splatting},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

