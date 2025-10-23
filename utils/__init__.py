"""
Utility Module for NeSVoR2

This module provides utility functions and classes for medical image processing,
particularly focused on MRI super-resolution reconstruction.

Submodules:
-----------
gen_utils: General utility functions
    - Logging utilities (TrainLogger, setup_logger, log_result)
    - Loss functions (ncc_loss, ssim_loss)
    - Image processing (resample, gaussian_blur, meshgrid)
    - Training helpers (MovingAverage, set_seed)

data_utils: Data structures and transformations
    - Image classes (Image, Slice, Volume, Stack)
    - Coordinate transformations (RigidTransform, affine2transformation)
    - File I/O (load_volume, save_volume, load_stack)
    - Transform utilities (mat2euler, euler2mat, transform_points)

Example usage:
-------------
    from utils import load_stack, resample, ncc_loss

    # Load MRI stack
    stack = load_stack('input.nii.gz', device='cuda')

    # Resample to higher resolution
    resampled = resample(stack, [1.0, 1.0, 2.0], [0.5, 0.5, 1.0])

    # Compute similarity loss
    loss = ncc_loss(image1, image2, reduction='mean')
"""

# Import general utilities
from .gen_utils import (
    # Logging utilities
    set_seed,  # Set random seeds for reproducibility
    makedirs,  # Create directories safely
    merge_args,  # Merge argument namespaces
    # Image processing
    resample,  # Resample image to new resolution
    meshgrid,  # Create coordinate meshgrid
    gaussian_blur,  # Apply Gaussian blur
    # Training utilities
    MovingAverage,  # Track moving average of metrics
    # Loss functions
    ncc_loss,  # Normalized cross-correlation loss
    ssim_loss,  # Structural similarity loss
    get_PSF,  # Generate point spread function
    resolution2sigma,  # Convert resolution to blur sigma
    # Logging classes and functions
    log_params,  # Log model parameters
    log_args,  # Log command-line arguments
    setup_logger,  # Configure logging system
    log_result,  # Log at RESULT level
    LazyLog,  # Lazy evaluation for expensive logs
    TrainLogger,  # Formatted training progress logger
    LogIO,  # Redirect output to logger
    # Type aliases
    PathType,  # Type for file paths
    DeviceType,  # Type for PyTorch devices
    RST,
    rst,
    rst2txt,
    show_link,
)

# Import data utilities
from .data_utils import (
    # NIfTI file I/O
    compare_resolution_affine,  # Check if images match
    affine2transformation,  # Convert NIfTI affine to internal format
    transformation2affine,  # Convert internal format to NIfTI affine
    save_nii_volume,  # Save volume to NIfTI file
    load_nii_volume,  # Load volume from NIfTI file
    # Image data structures
    Image,  # Base class for 3D medical images
    Stack,  # Stack of 2D slices
    Volume,  # 3D volume
    Slice,  # Single 2D slice
    # High-level I/O functions
    load_volume,  # Load volume with transformation
    load_slices,  # Load multiple slices from folder
    load_stack,  # Load stack from NIfTI
    load_mask,  # Load mask volume
    save_slices,  # Save slices to folder
    # Rigid transformation utilities
    axisangle2mat,  # Convert axis-angle to matrix
    mat2axisangle,  # Convert matrix to axis-angle
    RigidTransform,  # Rigid transformation class
    # Transformation format conversions
    mat_first2last,  # Change matrix convention
    mat_last2first,  # Change matrix convention
    ax_first2last,  # Change axis-angle convention
    ax_last2first,  # Change axis-angle convention
    mat_update_resolution,  # Update transformation for resolution change
    ax_update_resolution,  # Update axis-angle for resolution change
    # Euler angle conversions
    mat2euler,  # Convert matrix to Euler angles (degrees)
    euler2mat,  # Convert Euler angles to matrix
    # Point-based transformations
    point2mat,  # Convert three points to transformation
    mat2point,  # Convert transformation to three points
    # Apply transformations to points
    mat_transform_points,  # Transform points using matrix
    ax_transform_points,  # Transform points using axis-angle
    transform_points,  # Transform points using RigidTransform
    # Transform initialization
    init_stack_transform,  # Create evenly-spaced stack transform
    init_zero_transform,  # Create identity transform
    # Rotation utilities
    average_rotation,  # Compute average of multiple rotations
)

# Import data utilities from data module (used in training)
from data.data import PointDataset  # Point-based dataset for training
