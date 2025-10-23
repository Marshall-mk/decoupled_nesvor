"""
NeSVoR2: Neural Slice-to-Volume Reconstruction
Medical MRI Super-Resolution using Implicit Neural Representations

This package provides a complete pipeline for reconstructing high-resolution
3D MRI volumes from low-resolution, thick-slice 2D acquisitions.

Main components:
- model: Neural network architectures (INR, NeSVoR)
- preprocess: Preprocessing (segmentation, bias correction, registration)
- utils: Utility functions (data I/O, transformations, metrics)
- data: Data structures and slice acquisition models
- main: Entry point for running the full pipeline
"""

__version__ = "2.0.0"
__author__ = "NeSVoR2 Team"

# Export commonly used classes and functions
from utils import Stack, Slice, Volume, RigidTransform
from model.models import INR, NeSVoR

__all__ = [
    "Stack",
    "Slice",
    "Volume",
    "RigidTransform",
    "INR",
    "NeSVoR",
]
