"""
SVoRT: Slice-to-Volume Registration Transformer

This package implements SVoRT, a deep learning method for motion correction
in fetal MRI reconstruction. It uses transformer architecture to predict
rigid transformations for each slice, enabling robust reconstruction even
with severe inter-slice motion.

Main Components:
---------------
- models: SVoRT and SVoRTv2 network architectures
- inference: High-level API for running registration
- attention: Transformer components (attention, positional encoding, ResNet)
- svr_utils: Utility functions for reconstruction and evaluation

Example Usage:
-------------
```python
from preprocess.svort import svort_predict

# Run SVoRT registration on a list of stacks
slices = svort_predict(
    dataset=stacks,
    device=torch.device("cuda"),
    svort_version="v2",
    svort=True,      # Enable SVoRT
    vvr=False,       # Disable traditional registration
    force_vvr=False,
    force_scanner=False,
)
```

References:
----------
Xu, J., et al. "NeSVoR: Implicit Neural Representation for Slice-to-Volume
Reconstruction in MRI" (2022). IEEE TMI.
"""

from preprocess.svort.models import SVoRT, SVoRTv2
from preprocess.svort.inference import svort_predict, run_svort, parse_data
from preprocess.svr import simulate_slices, SRR_CG, stack_registration

__all__ = [
    "SVoRT",
    "SVoRTv2",
    "svort_predict",
    "run_svort",
    "parse_data",
    "simulate_slices",
    "SRR_CG",
    "stack_registration",
]
