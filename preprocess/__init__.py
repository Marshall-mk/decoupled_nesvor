"""
Preprocessing Module for NeSVoR2

This module provides preprocessing functions for MRI reconstruction pipeline:

1. **Masking/Segmentation**:
   - brain_segmentation: Deep learning-based brain extraction
   - volume_intersect: Compute volume mask from stack intersection
   - stack_intersect: Compute stack masks from slice overlap
   - otsu_thresholding: Automatic threshold-based masking
   - thresholding: Manual threshold masking

2. **Bias Field Correction**:
   - n4_bias_field_correction: N4ITK intensity inhomogeneity correction

3. **Quality Assessment**:
   - assess: Compute quality metrics for input stacks

4. **Registration** (SVoRT):
   - svort_predict: Transformer-based motion correction
   - run_svort: Core SVoRT registration pipeline
   - parse_data: Prepare data for registration

Example Usage:
-------------
```python
from preprocess import (
    brain_segmentation,
    n4_bias_field_correction,
    assess,
)
from preprocess.svort import svort_predict

# Segment brain
for stack in stacks:
    stack.mask = brain_segmentation(stack, device=device)

# Correct bias field
for stack in stacks:
    stack.slices = n4_bias_field_correction(stack)

# Assess quality
results = assess(stacks)

# Register with SVoRT
slices = svort_predict(stacks, device=device, svort=True)
```
"""

from preprocess.bias_field import n4_bias_field_correction
from preprocess.brain_segmentation import brain_segmentation
from preprocess.intersection import volume_intersect, stack_intersect
from preprocess.thresholding import otsu_thresholding, thresholding
from preprocess.assessment import assess

# SVoRT registration functions
from preprocess.svort import svort_predict, run_svort, parse_data

__all__ = [
    # Masking/Segmentation
    "brain_segmentation",
    "volume_intersect",
    "stack_intersect",
    "otsu_thresholding",
    "thresholding",
    # Bias correction
    "n4_bias_field_correction",
    # Quality assessment
    "assess",
    # Registration
    "svort_predict",
    "run_svort",
    "parse_data",
]
