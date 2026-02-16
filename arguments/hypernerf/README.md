# HyperNeRF Dataset Configurations

This directory contains configuration files for training on HyperNeRF dataset scenes. HyperNeRF captures topological changes and non-rigid deformations (food being eaten, hands moving, objects being manipulated).

## Structure

The configs use **inheritance** to minimize duplication:

```
hypernerf/
├── base.py           # Base config for all HyperNeRF scenes
├── default.py        # (legacy, same as base.py)
└── {scene}.py        # Scene-specific configs (inherit from base.py)
```

## Base Configuration

### `base.py`
Common settings for all HyperNeRF scenes:
- Dataset type: `nerfies` (HyperNeRF's data format)
- HexPlane grid resolution: [64, 64, 64, 150]
- 14K iterations, batch size 2
- Deeper deformation network (defor_depth=1)
- Higher multires: [1, 2, 4]

Most scenes use the default settings without modifications!

## Scenes

### General Scenes
- `3dprinter.py` - 3D printer in action (100 frames)
- `aleks-teapot.py` - Teapot being manipulated
- `americano.py` - Making americano coffee
- `banana.py` - Banana scene
- `broom2.py` - Broom scene
- `espresso.py` - Making espresso
- `keyboard.py` - Keyboard typing
- `tamping.py` - Coffee tamping
- `torchocolate.py` - Torchocolate scene

### Food Manipulation
- `chicken.py` - Chicken being handled (80 frames)
- `chickchicken.py` - Chicken scene variant
- `cut-lemon1.py` - Cutting lemon
- `slice-banana.py` - Slicing banana
- `split-cookie.py` - Splitting cookie

### Hand Scenes
- `hand1-dense-v2.py` - Dense hand capture
- `cross-hands1.py` - Crossing hands
- `oven-mitts.py` - Oven mitts scene

### VRIG Dataset (High Resolution)
- `vrig-3dprinter.py` - High-res 3D printer
- `vrig-chicken.py` - High-res chicken
- `vrig-peel-banana.py` - High-res banana peeling

## Usage

### Basic Training
```bash
python train.py \
  --source_path data/hypernerf/chicken \
  --model_path output/hypernerf/chicken \
  --configs arguments/hypernerf/chicken.py
```

### Using GitHub Scripts
```bash
./scripts/train_velocity_field.sh chicken arguments/hypernerf/chicken.py
```

## Creating New Configs

For a new scene:

```python
"""Your scene description"""
_base_ = './base.py'

# Most scenes don't need any overrides!
# Only override if you need different temporal resolution:
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 80]  # Your custom temporal resolution
    },
)
```

## Tips

- **Default settings work well** for most HyperNeRF scenes
- **Adjust temporal resolution** (last dimension) based on sequence length
- **Batch size 2** is standard due to higher resolution than D-NeRF
- HyperNeRF scenes benefit from the **deeper deformation network** (defor_depth=1)

