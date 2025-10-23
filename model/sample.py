"""
Sampling from Trained Implicit Neural Representations

This module provides functions to sample (query) trained INR models at arbitrary
3D coordinates. This enables:
- Generating high-resolution 3D volumes from the learned representation
- Simulating 2D slices at any position/orientation
- Querying at arbitrary points for analysis

Key Concepts:
-----------
1. **Continuous Representation**:
   - INR represents volume as a continuous function: f(x,y,z) = intensity
   - Can query at any coordinate, not limited to original grid
   - Enables super-resolution: output can be finer than input

2. **PSF Simulation During Sampling**:
   - Can simulate Point Spread Function (blur) when sampling
   - Monte Carlo sampling: average multiple random points around query location
   - Useful for simulating thick slices from thin-slice representation

3. **Mask-Based Sampling**:
   - Only sample within a mask (region of interest)
   - Saves computation and memory
   - Mask defines output volume shape and orientation

Functions:
---------
- override_sample_mask: Modify mask (resolution/orientation) for sampling
- sample_volume: Generate 3D volume from INR model
- sample_points: Query INR at arbitrary 3D points
- sample_slice: Generate single 2D slice with optional PSF
- sample_slices: Generate multiple 2D slices in batch

Typical Workflow:
----------------
1. Train INR model on low-resolution, thick slices
2. Create mask defining high-resolution output volume
3. Sample model to generate high-resolution reconstruction
4. Optionally sample slices for comparison with inputs
"""

from typing import List, Union, Optional
import os
import torch
from model.models import INR
from utils import (
    resolution2sigma,
    meshgrid,
    PathType,
    Slice,
    Volume,
    load_volume,
    load_mask,
    transform_points,
    RigidTransform,
)


def override_sample_mask(
    mask: Volume,
    new_mask: Union[PathType, None, Volume] = None,
    new_resolution: Optional[float] = None,
    new_orientation: Union[PathType, None, Volume, RigidTransform] = None,
) -> Volume:
    """
    Create a modified version of the sampling mask with new properties.

    This function allows you to change the mask's:
    - Binary mask pattern (which voxels to sample)
    - Resolution (voxel size)
    - Orientation (rigid transformation in space)

    Use cases:
    - Sample at higher resolution than the mask
    - Change output volume orientation to match a reference
    - Use a different anatomical mask

    Args:
        mask: Original Volume mask defining sampling region
        new_mask: New binary mask (Volume, file path, or None to keep original)
        new_resolution: New voxel size (mm), or None to keep original
        new_orientation: New orientation (Volume, RigidTransform, file path, or None)

    Returns:
        Modified Volume mask with updated properties

    Note:
        - If new_mask is provided, replaces entire mask
        - If new_resolution or new_orientation provided, resamples the mask
        - Resampling uses nearest-neighbor to preserve binary mask

    Example:
        >>> # Load original mask at 1mm resolution
        >>> mask = load_mask('brain_mask.nii.gz')
        >>>
        >>> # Sample at 0.5mm resolution (super-resolution)
        >>> high_res_mask = override_sample_mask(mask, new_resolution=0.5)
        >>>
        >>> # Match orientation to reference volume
        >>> ref_vol = load_volume('reference.nii.gz')
        >>> aligned_mask = override_sample_mask(mask, new_orientation=ref_vol)
    """
    # Replace mask if provided
    if new_mask is not None:
        if isinstance(new_mask, Volume):
            mask = new_mask
        elif isinstance(new_mask, (str, os.PathLike)):
            # Load mask from file
            mask = load_mask(new_mask, device=mask.device)
        else:
            raise TypeError("unknwon type for mask")

    # Extract new transformation if provided
    transformation = None
    if new_orientation is not None:
        if isinstance(new_orientation, Volume):
            # Use transformation from Volume
            transformation = new_orientation.transformation
        elif isinstance(new_orientation, RigidTransform):
            # Use provided transformation directly
            transformation = new_orientation
        elif isinstance(new_orientation, (str, os.PathLike)):
            # Load transformation from file
            transformation = load_volume(
                new_orientation,
                device=mask.device,
            ).transformation

    # Resample mask if resolution or orientation changed
    if transformation or new_resolution:
        mask = mask.resample(new_resolution, transformation)

    return mask


def sample_volume(
    model: INR,
    mask: Volume,
    psf_resolution: float,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> Volume:
    """
    Generate a 3D volume by sampling the INR model at all masked voxels.

    This is the main function for generating high-resolution reconstructions
    from a trained model. It:
    1. Queries the model at each voxel center in the mask
    2. Optionally simulates PSF blur via Monte Carlo sampling
    3. Returns a Volume with the sampled intensities

    Args:
        model: Trained INR model to sample from
        mask: Volume defining where to sample (mask=True) and output shape
        psf_resolution: Resolution for PSF simulation (0 = no blur, sharp sampling)
        batch_size: Number of points to process at once (larger = faster but more memory)
        n_samples: Number of Monte Carlo samples per point for PSF (more = smoother but slower)

    Returns:
        Volume with sampled intensities at all masked voxels

    Note:
        - Model is set to eval mode (no dropout, batchnorm in inference mode)
        - Uses torch.no_grad() internally for memory efficiency
        - Output has same shape, resolution, and transformation as mask
        - Unmasked voxels keep their original values (typically 0)

    Example:
        >>> # Train model on low-res slices
        >>> model = train(slices, args)
        >>>
        >>> # Create high-resolution mask (0.5mm isotropic)
        >>> mask = dataset.mask.resample(resolution=0.5)
        >>>
        >>> # Sample at high resolution with PSF blur
        >>> volume = sample_volume(
        ...     model, mask,
        ...     psf_resolution=0.5,  # Match output resolution
        ...     batch_size=2048,
        ...     n_samples=128
        ... )
        >>>
        >>> # Save result
        >>> save_volume(volume, 'reconstruction.nii.gz')
    """
    model.eval()  # Set to evaluation mode

    # Clone mask to create output volume
    img = mask.clone()

    # Sample at all masked voxel centers
    img.image[img.mask] = sample_points(
        model,
        img.xyz_masked,  # 3D coordinates of masked voxels
        psf_resolution,
        batch_size,
        n_samples,
    )

    return img


def sample_points(
    model: INR,
    xyz: torch.Tensor,
    resolution: float = 0,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> torch.Tensor:
    """
    Query the INR model at arbitrary 3D points.

    This is a low-level function that evaluates the model at given coordinates.
    Useful for:
    - Sampling at arbitrary (non-grid) locations
    - Implementing custom sampling patterns
    - Analysis and visualization

    Args:
        model: Trained INR model to query
        xyz: (..., 3) 3D coordinates to query (in world space, mm)
        resolution: Resolution for PSF simulation (0 = sharp, no blur)
        batch_size: Points to process at once (memory vs. speed trade-off)
        n_samples: Monte Carlo samples per point for PSF blur

    Returns:
        (...,) intensity values at query points

    Note:
        - Coordinates should be in world space (mm), not normalized
        - Model handles normalization to [0, 1]^3 internally
        - If resolution > 0, averages n_samples random points around each query
        - Batching is automatic - works with any number of points

    Example:
        >>> # Query along a line
        >>> t = torch.linspace(0, 100, 1000)  # 1000 points from 0 to 100mm
        >>> xyz = torch.stack([t, t * 0, t * 0], -1)  # Line along x-axis
        >>> intensities = sample_points(model, xyz, resolution=1.0)
        >>> plt.plot(t, intensities)  # Intensity profile along line
        >>>
        >>> # Query on a sphere surface
        >>> theta = torch.linspace(0, 2*np.pi, 100)
        >>> phi = torch.linspace(0, np.pi, 50)
        >>> xyz = sphere_coords(theta, phi, radius=50)  # (100, 50, 3)
        >>> intensities = sample_points(model, xyz)  # (100, 50)
    """
    # Store original shape for reshaping output
    shape = xyz.shape[:-1]

    # Flatten to (N, 3) for batch processing
    xyz = xyz.view(-1, 3)

    # Allocate output tensor
    v = torch.empty(xyz.shape[0], dtype=torch.float32, device=xyz.device)

    # Process in batches
    with torch.no_grad():
        for i in range(0, xyz.shape[0], batch_size):
            # Extract batch
            xyz_batch = xyz[i : i + batch_size]

            # Generate PSF samples if needed
            # sample_batch adds Gaussian noise if resolution > 0
            xyz_batch = model.sample_batch(
                xyz_batch,
                None,  # No transformation (already in world space)
                resolution2sigma(resolution, isotropic=True),
                0 if resolution <= 0 else n_samples,  # 0 samples = no PSF
            )

            # Forward pass through model
            v_b = model(xyz_batch).mean(-1)  # Average over PSF samples

            # Store results
            v[i : i + batch_size] = v_b

    # Reshape to original shape
    return v.view(shape)


def sample_slice(
    model: INR,
    slice: Slice,
    mask: Volume,
    output_psf_factor: float = 1.0,
    n_samples: int = 128,
) -> Slice:
    """
    Generate a 2D slice by sampling the INR model at slice pixel positions.

    This simulates acquiring a 2D slice from the reconstructed 3D volume.
    Useful for:
    - Comparing with original input slices
    - Generating slices at arbitrary positions/orientations
    - Simulating different acquisition parameters

    Args:
        model: Trained INR model to sample from
        slice: Slice object defining position, orientation, and resolution
        mask: Volume mask (only sample pixels inside mask)
        output_psf_factor: PSF scaling (1.0 = match slice resolution, 0 = no blur)
        n_samples: Monte Carlo samples for PSF simulation

    Returns:
        Slice with sampled intensities (same shape as input slice)

    Note:
        - Output slice has same transformation as input slice
        - Pixels outside mask are set to 0
        - PSF factor = 1.0 simulates original slice thickness
        - PSF factor = 0.0 gives sharp sampling (no through-plane blur)

    Algorithm:
        1. Create grid of pixel coordinates in slice's local space
        2. Transform to world space using slice transformation
        3. Check which pixels are inside mask
        4. Sample model at those pixels with PSF
        5. Put sampled values back into slice

    Example:
        >>> # Original low-res input slice
        >>> slice_input = slices[0]
        >>>
        >>> # Sample model to get reconstructed version
        >>> slice_recon = sample_slice(
        ...     model, slice_input, mask,
        ...     output_psf_factor=1.0,  # Match original PSF
        ...     n_samples=128
        ... )
        >>>
        >>> # Compare input vs. reconstruction
        >>> residual = slice_input.image - slice_recon.image
        >>> print(f"RMS error: {residual.pow(2).mean().sqrt()}")
    """
    # Clone slice and zero out image
    slice_sampled = slice.clone(zero=True)

    # Create grid of pixel coordinates (H, W, 3)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)

    # Transform to world space and check if inside mask
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0

    # Sample only pixels inside mask
    if m.any():
        # Generate PSF samples around each pixel
        xyz_masked = model.sample_batch(
            xyz[m],
            slice_sampled.transformation,
            resolution2sigma(
                slice_sampled.resolution_xyz * output_psf_factor, isotropic=False
            ),
            0 if output_psf_factor <= 0 else n_samples,
        )

        # Forward pass through model, average over PSF samples
        v = model(xyz_masked).mean(-1)

        # Update slice with sampled values
        slice_sampled.mask = m.view(slice_sampled.mask.shape)
        slice_sampled.image[slice_sampled.mask] = v.to(slice_sampled.image.dtype)

    return slice_sampled


def sample_slices(
    model: INR,
    slices: List[Slice],
    mask: Volume,
    output_psf_factor: float = 1.0,
    n_samples: int = 128,
) -> List[Slice]:
    """
    Generate multiple 2D slices by sampling the INR model.

    Batch version of sample_slice for processing multiple slices efficiently.

    Args:
        model: Trained INR model to sample from
        slices: List of Slice objects to sample
        mask: Volume mask (only sample pixels inside mask)
        output_psf_factor: PSF scaling (1.0 = match slice resolution)
        n_samples: Monte Carlo samples for PSF simulation

    Returns:
        List of sampled Slices (same length as input)

    Note:
        - Each slice can have different position/orientation/resolution
        - Uses eval mode and no_grad for efficiency
        - Processes slices sequentially (could parallelize for very large batches)

    Example:
        >>> # Sample all input slices from model
        >>> slices_recon = sample_slices(
        ...     model, slices_input, mask,
        ...     output_psf_factor=1.0
        ... )
        >>>
        >>> # Compute per-slice errors
        >>> for i, (inp, recon) in enumerate(zip(slices_input, slices_recon)):
        ...     error = (inp.image - recon.image).pow(2).mean().sqrt()
        ...     print(f"Slice {i}: RMSE = {error:.4f}")
        >>>
        >>> # Save reconstructed slices
        >>> save_slices(slices_recon, 'output_slices/')
    """
    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        slices_sampled = []
        for i, slice in enumerate(slices):
            # Sample each slice
            slices_sampled.append(
                sample_slice(model, slice, mask, output_psf_factor, n_samples)
            )

    return slices_sampled
