"""
Sparse Temporal Sampler for efficient 4D Gaussian Splatting training.

Samples only a subset of frames (e.g., every 2nd or 5th frame) to:
1. Speed up training (fewer frames per epoch)
2. Force velocity field to learn smooth temporal interpolation
3. Improve generalization (can't overfit to individual frames)
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional


class SparseTemporalSampler(Sampler):
    """
    Sampler that selects frames at regular intervals (stride).
    
    For datasets with structure: [cam0_t0, cam0_t1, ..., cam0_t299, cam1_t0, ...]
    This sampler will select indices where frame_idx % stride == 0.
    
    Args:
        data_source: Dataset to sample from
        num_cameras: Number of cameras in the dataset
        num_frames: Number of time steps per camera
        temporal_stride: Sample every Nth frame (default: 2 = every other frame)
        shuffle: Whether to shuffle the selected indices (default: True)
    """
    
    def __init__(
        self,
        data_source,
        num_cameras: int,
        num_frames: Optional[int] = None,
        temporal_stride: int = 2,
        frame_offset: int = 0,
        shuffle: bool = True
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_cameras = num_cameras
        total_samples = len(data_source)

        if num_cameras <= 0:
            raise ValueError("SparseTemporalSampler requires num_cameras > 0")

        inferred_frames = total_samples // num_cameras
        if inferred_frames * num_cameras != total_samples:
            print(f"[SPARSE WARNING] Dataset length ({total_samples}) is not divisible by cameras ({num_cameras}). "
                  f"Using floor division result {inferred_frames} for frames per camera.")

        if num_frames is None:
            self.num_frames = inferred_frames
        else:
            if num_frames != inferred_frames:
                print(f"[SPARSE WARNING] Provided num_frames ({num_frames}) differs from inferred ({inferred_frames}). "
                      f"Using inferred value.")
            self.num_frames = inferred_frames

        self.temporal_stride = temporal_stride
        stride = max(self.temporal_stride, 1)
        self.frame_offset = frame_offset % stride if self.num_frames > 0 else 0
        self.shuffle = shuffle
        
        # Build list of valid indices
        self.valid_indices = []
        for cam_idx in range(num_cameras):
            base_idx = cam_idx * self.num_frames
            for frame_idx in range(self.num_frames):
                if frame_idx >= self.frame_offset and ((frame_idx - self.frame_offset) % stride == 0):
                    # Dataset index: cam_idx * num_frames + frame_idx
                    idx = base_idx + frame_idx
                    if idx >= total_samples:
                        continue
                    self.valid_indices.append(idx)
        
        self.num_samples = len(self.valid_indices)
        
        supervised_frames = max(0, self.num_frames - self.frame_offset)
        supervised_frames = (supervised_frames + stride - 1) // stride
        print(f"SparseTemporalSampler: Using {self.num_samples}/{len(data_source)} frames "
              f"(stride={self.temporal_stride}, offset={self.frame_offset})")
        print(f"  Cameras: {num_cameras}, Frames per camera: {supervised_frames}/{self.num_frames}")
    
    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Shuffle the valid indices
            indices = torch.randperm(self.num_samples).tolist()
            return iter([self.valid_indices[i] for i in indices])
        else:
            return iter(self.valid_indices)
    
    def __len__(self) -> int:
        return self.num_samples


class AdaptiveSparseTemporalSampler(Sampler):
    """
    Adaptive sampler that changes stride during training.
    
    Starts dense (stride=1) early in training for densification,
    then becomes sparse (stride>1) later for efficiency and generalization.
    
    Args:
        data_source: Dataset to sample from
        num_cameras: Number of cameras
        num_frames: Number of frames per camera
        initial_stride: Stride at start of training (default: 1 = all frames)
        final_stride: Stride after transition (default: 2 = every other frame)
        transition_iter: When to switch strides (default: 5000)
        current_iter: Current training iteration (will be updated externally)
    """
    
    def __init__(
        self,
        data_source,
        num_cameras: int,
        num_frames: int,
        initial_stride: int = 1,
        final_stride: int = 2,
        transition_iter: int = 5000,
        shuffle: bool = True
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        self.initial_stride = initial_stride
        self.final_stride = final_stride
        self.transition_iter = transition_iter
        self.shuffle = shuffle
        self.current_iter = 0  # Will be updated by training loop
        
        # Pre-compute indices for both strides
        self.indices_dense = self._build_indices(initial_stride)
        self.indices_sparse = self._build_indices(final_stride)
        
        print(f"AdaptiveSparseTemporalSampler:")
        print(f"  Initial ({initial_stride}): {len(self.indices_dense)} frames")
        print(f"  Final ({final_stride}): {len(self.indices_sparse)} frames (after iter {transition_iter})")
    
    def _build_indices(self, stride: int):
        """Build list of valid indices for given stride."""
        indices = []
        for cam_idx in range(self.num_cameras):
            for frame_idx in range(self.num_frames):
                if frame_idx % stride == 0:
                    idx = cam_idx * self.num_frames + frame_idx
                    indices.append(idx)
        return indices
    
    def set_iteration(self, iteration: int):
        """Update current iteration (call from training loop)."""
        self.current_iter = iteration
    
    def __iter__(self) -> Iterator[int]:
        # Choose indices based on current iteration
        if self.current_iter < self.transition_iter:
            indices = self.indices_dense
        else:
            indices = self.indices_sparse
        
        if self.shuffle:
            perm = torch.randperm(len(indices)).tolist()
            return iter([indices[i] for i in perm])
        else:
            return iter(indices)
    
    def __len__(self) -> int:
        if self.current_iter < self.transition_iter:
            return len(self.indices_dense)
        else:
            return len(self.indices_sparse)

