# Interactive Training Commands for Debugging

## Option 1: Full Interactive Shell (Recommended for Multiple Breakpoints)

Get an interactive shell on a GPU node, then run commands manually:

```bash
srun --job-name=evogs_debug \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --mem=64GB \
     --time=4:00:00 \
     --account=visualai \
     --partition=visualai \
     --gres=gpu:l40:1 \
     --pty bash
```

Once you're in the interactive session:

```bash
# Activate environment
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

# Navigate to project
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

# Check GPU
nvidia-smi

# Run with Python debugger (pdb)
python -m pdb train.py \
    -s data/dynerf/cut_roasted_beef \
    --port 6019 \
    --expname debug/cut_roasted_beef \
    --configs arguments/dynerf/cut_roasted_beef_velocity.py \
    --iterations 1000

# Or use ipdb (better UI, if installed)
python -m ipdb train.py \
    -s data/dynerf/cut_roasted_beef \
    --port 6019 \
    --expname debug/cut_roasted_beef \
    --configs arguments/dynerf/cut_roasted_beef_velocity.py \
    --iterations 1000
```

---

## Option 2: Direct srun with Command

Run training directly with debugger in one command:

```bash
srun --job-name=evogs_debug \
     --nodes=1 \
     --cpus-per-task=8 \
     --mem=64GB \
     --time=4:00:00 \
     --account=visualai \
     --partition=visualai \
     --gres=gpu:l40:1 \
     --pty bash -c "
source /u/aa0008/miniconda/etc/profile.d/conda.sh && \
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12 && \
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting && \
python -m pdb train.py \
    -s data/dynerf/cut_roasted_beef \
    --port 6019 \
    --expname debug/cut_roasted_beef \
    --configs arguments/dynerf/cut_roasted_beef_velocity.py \
    --iterations 1000
"
```

---

## Option 3: Use Helper Scripts

We've created helper scripts for you:

### Interactive Shell (Best for exploring)
```bash
./evogs_scripts/train_interactive.sh cut_roasted_beef arguments/dynerf/cut_roasted_beef_velocity.py
```

### Quick Debug Run (Best for testing breakpoints)
```bash
./evogs_scripts/debug_train.sh --pdb   # Use Python's pdb
./evogs_scripts/debug_train.sh --ipdb  # Use ipdb (better colors/UI)
```

---

## Adding Breakpoints in Code

### Method 1: Add breakpoint() in your code (Python 3.7+)
```python
# In train.py or any file
def scene_reconstruction(...):
    # Your code
    breakpoint()  # Execution stops here
    # More code
```

### Method 2: Use pdb.set_trace()
```python
import pdb

def scene_reconstruction(...):
    # Your code
    pdb.set_trace()  # Execution stops here
    # More code
```

### Method 3: Use ipdb (prettier)
```python
import ipdb

def scene_reconstruction(...):
    # Your code
    ipdb.set_trace()  # Execution stops here with syntax highlighting
    # More code
```

---

## Common PDB Commands

Once you hit a breakpoint:

```
h          - Help (list all commands)
l          - List current location in code
ll         - List entire current function
n          - Next line (step over)
s          - Step into function
c          - Continue execution until next breakpoint
p <var>    - Print variable value
pp <var>   - Pretty-print variable
w          - Show stack trace
u          - Move up in stack
d          - Move down in stack
b <line>   - Set breakpoint at line number
cl         - Clear all breakpoints
q          - Quit debugger
```

---

## Example Debugging Session

```bash
# 1. Get interactive GPU node
srun --gres=gpu:l40:1 --time=2:00:00 --account=visualai --partition=visualai --pty bash

# 2. Setup environment
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

# 3. Add breakpoint in train.py at line 124 (train_time_max logic)
# Add: breakpoint()

# 4. Run with debugger
python train.py \
    -s data/dynerf/cut_roasted_beef \
    --expname debug/test \
    --configs arguments/dynerf/cut_roasted_beef_velocity.py \
    --iterations 100

# 5. When breakpoint hits, inspect variables:
(Pdb) p train_time_max
0.5
(Pdb) p len(train_cams)
1200
(Pdb) l
# Shows code around current line
(Pdb) c
# Continue execution
```

---

## Tips

1. **Start with fewer iterations** (`--iterations 100`) for faster debugging cycles
2. **Use ipdb instead of pdb** for better syntax highlighting (install: `pip install ipdb`)
3. **Request shorter time** (`--time=1:00:00`) if you're just testing
4. **Use `--gres=gpu:a100:1`** if L40 queue is busy
5. **Add `--detect_anomaly`** flag to catch NaN/Inf errors with better stack traces

---

## Troubleshooting

### If srun hangs waiting for resources:
```bash
# Check queue status
squeue -u $USER

# Try different GPU type
--gres=gpu:a100:1  # Instead of l40

# Or different partition
--partition=nlp  # If visualai is busy
```

### If conda activate fails:
```bash
# Initialize conda first
conda init bash
source ~/.bashrc
```

### If port 6009 is already in use:
```bash
# Change port in command
--port 6019  # or 6029, 6039, etc.
```


