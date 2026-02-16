# D-NeRF Dataset Configurations

Clean, DRY (Don't Repeat Yourself) configuration structure for D-NeRF synthetic datasets.

## Structure

```
dnerf/
├── base.py              # Base configuration for all D-NeRF scenes
├── base_velocity.py     # Base configuration with velocity field enabled
├── lego.py              # Scene-specific configs (25 frames)
├── lego_velocity.py     # Scene with velocity field
├── mutant.py            # Scene-specific configs (75 frames)
├── mutant_velocity.py   # Scene with velocity field
└── ...                  # Other scenes
```

## Usage

All common parameters are defined in `base.py`. Each scene file only specifies:
1. Scene name (in docstring)
2. Temporal resolution (25 or 75 frames)
3. Any scene-specific overrides

Velocity variants inherit from `base_velocity.py` which adds Neural ODE dynamics.

## Scenes

### 25-frame scenes:
- `lego.py` / `lego_velocity.py`

### 75-frame scenes:
- `bouncingballs.py`
- `hellwarrior.py`
- `hook.py`
- `jumpingjacks.py` / `jumpingjacks_velocity.py`
- `mutant.py` / `mutant_velocity.py`
- `standup.py`
- `trex.py`

## Design Principles

- **DRY**: All common code is in base files
- **Minimal**: Scene configs only specify what's unique
- **Clear**: Docstrings indicate scene and frame count
- **Maintainable**: Changes to base parameters apply to all scenes

