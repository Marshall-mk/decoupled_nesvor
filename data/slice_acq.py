"""
Slice Acquisition Simulation for MRI Super-Resolution

This module simulates the physical process of acquiring 2D MRI slices from a 3D volume.
It implements the forward model (volume → slices) and adjoint operation (slices → volume)
used in iterative reconstruction algorithms.

Key Concepts:
-----------
1. **Forward Model**: Simulates how a 3D volume is sampled to produce 2D slices
   - Each slice has a position and orientation (rigid transformation)
   - Point Spread Function (PSF) models blur from slice thickness
   - Produces weighted averages of voxels within slice thickness

2. **Adjoint Operation**: Back-projects slices into volume space
   - Used in gradient-based optimization for reconstruction
   - Not exactly the inverse, but the transpose of the forward operator

3. **Sparse Matrix Representation**:
   - Each slice-volume relationship is a sparse matrix (most voxels don't contribute)
   - Uses PyTorch sparse COO tensors for memory efficiency
   - Coefficients represent PSF-weighted contributions

4. **Memory Management**:
   - Adaptive batch sizing to handle GPU memory constraints
   - Automatically reduces batch size on OOM errors

Mathematical Model:
------------------
For each slice i:
    slice_i = Σ_j PSF(dist(pixel, voxel_j)) * volume_j

Where:
    - dist() is the 3D distance between pixel and voxel centers
    - PSF() is the Point Spread Function (typically Gaussian)
    - The sum is over voxels within the PSF support region

Functions:
---------
- slice_acquisition_torch: Forward model (volume → slices with PSF)
- slice_acquisition_no_psf_torch: Fast forward model (no PSF, trilinear interpolation)
- slice_acquisition_adjoint_torch: Adjoint operation (slices → volume)
- _construct_coef: Build sparse coefficient matrix for a batch of slices
- _construct_slice_coef: Build sparse coefficients for a single slice
"""

from typing import Optional, cast, Sequence
import logging
import torch
import torch.nn.functional as F
from utils import Volume, Slice, mat_transform_points


# Global batch size for processing multiple slices at once
# Automatically adjusted downward on GPU out-of-memory errors
BATCH_SIZE = 64


def _construct_coef(
    idxs, transforms, vol_shape, slice_shape, vol_mask, slice_mask, psf, res_slice
):
    """
    Construct sparse coefficient matrix for a batch of slices.

    This function builds the sparse matrix that represents how each slice samples
    the volume. Each entry (i,j) in the matrix represents the contribution of
    volume voxel j to slice pixel i, weighted by the PSF.

    Args:
        idxs: List of slice indices to process
        transforms: (N, 3, 4) rigid transformations for all slices
        vol_shape: (D, H, W) shape of the volume
        slice_shape: (H, W) shape of each slice
        vol_mask: Optional (D, H, W) binary mask for volume
        slice_mask: Optional (N, 1, H, W) binary masks for slices
        psf: (D_psf, H_psf, W_psf) Point Spread Function kernel
        res_slice: Slice resolution (in-plane pixel spacing)

    Returns:
        Sparse COO tensor of shape (n_slice_pixels, n_volume_voxels)
        where n_slice_pixels = len(idxs) * H * W
        and n_volume_voxels = D * H * W

    Note:
        - The matrix is very sparse (typically <1% non-zero entries)
        - Uses COO format for efficient storage and matrix-vector multiplication
        - Coalesced to combine duplicate indices (same pixel-voxel pair)
    """
    # Build coefficients for each slice separately
    slice_ids = []
    volume_ids = []
    psf_vs = []
    for i in range(len(idxs)):
        slice_id, volume_id, psf_v = _construct_slice_coef(
            i,
            transforms[idxs[i]],
            vol_shape,
            slice_shape,
            vol_mask,
            slice_mask[idxs[i]] if slice_mask is not None else None,
            psf,
            res_slice,
        )
        slice_ids.append(slice_id)
        volume_ids.append(volume_id)
        psf_vs.append(psf_v)

    # Concatenate all slice coefficients into one sparse matrix
    slice_id = torch.cat(slice_ids)
    del slice_ids
    volume_id = torch.cat(volume_ids)
    del volume_ids
    ids = torch.stack((slice_id, volume_id), 0)
    del slice_id, volume_id
    psf_v = torch.cat(psf_vs)
    del psf_vs

    # Create sparse COO tensor and coalesce (combine duplicate indices)
    coef = torch.sparse_coo_tensor(
        ids,
        psf_v,
        [
            slice_shape[0] * slice_shape[1] * len(idxs),  # Total slice pixels
            vol_shape[0] * vol_shape[1] * vol_shape[2],  # Total volume voxels
        ],
    ).coalesce()

    return coef


def _construct_slice_coef(
    i, transform, vol_shape, slice_shape, vol_mask, slice_mask, psf, res_slice
):
    """
    Construct sparse coefficient matrix for a single slice.

    This function:
    1. Generates coordinates for all pixels in the slice
    2. Transforms them to volume space using the slice transformation
    3. For each pixel, finds nearby volume voxels within PSF support
    4. Computes PSF-weighted contributions
    5. Returns sparse indices and values

    Args:
        i: Index of this slice in the batch (for offset calculation)
        transform: (3, 4) rigid transformation for this slice
        vol_shape: (D, H, W) shape of the volume
        slice_shape: (H, W) shape of the slice
        vol_mask: Optional (D, H, W) binary mask for volume
        slice_mask: Optional (1, H, W) binary mask for this slice
        psf: (D_psf, H_psf, W_psf) Point Spread Function kernel
        res_slice: Slice resolution (in-plane pixel spacing)

    Returns:
        slice_id: (N_entries,) flat indices of slice pixels
        volume_id: (N_entries,) flat indices of volume voxels
        psf_v: (N_entries,) PSF weights for each entry

        where N_entries is the number of non-zero coefficients

    Algorithm:
        1. Create PSF volume and get its coordinates and values
        2. Create slice and get its coordinates
        3. Transform slice coords to volume space
        4. For each slice pixel, add PSF offsets to get contributing voxels
        5. Filter out voxels outside volume bounds
        6. Flatten indices for sparse matrix representation
    """
    # Add batch dimension to transform
    transform = transform[None]

    # Create Volume from PSF to get coordinates and values
    psf_volume = Volume(psf, psf > 0, resolution_x=1)
    psf_xyz = psf_volume.xyz_masked_untransformed  # (N_psf, 3) PSF support coordinates
    psf_v = psf_volume.v_masked  # (N_psf,) PSF values

    # Create slice mask (use provided mask or all ones)
    if slice_mask is not None:
        _slice = slice_mask
    else:
        _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=psf.device)

    # Get slice pixel coordinates in slice's local coordinate system
    slice_xyz = Slice(_slice, _slice, resolution_x=res_slice).xyz_masked_untransformed

    # Transform slice coordinates to volume coordinate system
    slice_xyz = mat_transform_points(transform, slice_xyz, trans_first=True)

    # Transform PSF coordinates relative to slice origin
    # (PSF is centered at origin, so subtract translation before transforming rotation)
    psf_xyz = mat_transform_points(
        transform, psf_xyz - transform[:, :, -1], trans_first=True
    )

    # Compute center of volume (used as origin for coordinate system)
    shift_xyz = (
        torch.tensor(vol_shape[::-1], dtype=psf.dtype, device=psf.device) - 1
    ) / 2.0

    # For each slice pixel, add PSF offsets to get all contributing volume voxels
    # Broadcasting: (n_pixel, 1, 3) + (1, n_psf, 3) = (n_pixel, n_psf, 3)
    slice_xyz = shift_xyz + psf_xyz.reshape((1, -1, 3)) + slice_xyz.reshape((-1, 1, 3))

    # Mask out voxels outside volume bounds
    # (n_pixel, n_psf) boolean mask
    inside_mask = torch.all((slice_xyz > 0) & (slice_xyz < (shift_xyz * 2)), -1)

    # Keep only voxels inside volume, round to integer indices
    # (n_masked, 3) integer coordinates
    slice_xyz = slice_xyz[inside_mask].round().long()

    # Generate slice pixel indices (flat indices in the batch of slices)
    # Start from i * slice_size for this slice's offset in the batch
    slice_id = torch.arange(
        i * slice_shape[0] * slice_shape[1],
        (i + 1) * slice_shape[0] * slice_shape[1],
        dtype=torch.long,
        device=psf.device,
    )

    # If using slice mask, keep only masked pixels
    if slice_mask is not None:
        slice_id = slice_id.view_as(slice_mask)[slice_mask]

    # Expand slice indices to match PSF support size, then apply inside_mask
    # Each pixel contributes to multiple voxels (PSF support region)
    slice_id = slice_id[..., None].expand(-1, psf_v.shape[0])[inside_mask]

    # Expand PSF values to match all pixels, then apply inside_mask
    psf_v = psf_v[None].expand(inside_mask.shape[0], -1)[inside_mask]

    # Convert 3D volume coordinates to flat indices
    # Using row-major ordering: idx = x + W*y + W*H*z
    volume_id = (
        slice_xyz[:, 0]
        + slice_xyz[:, 1] * vol_shape[2]
        + slice_xyz[:, 2] * (vol_shape[1] * vol_shape[2])
    )

    return slice_id, volume_id, psf_v


def slice_acquisition_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    psf: torch.Tensor,
    slice_shape: Sequence,
    res_slice: float,
    need_weight: bool,
):
    """
    Forward slice acquisition: simulate 2D slice extraction from 3D volume.

    This is the main forward model that simulates how MRI acquires 2D slices from
    the underlying 3D anatomy. It accounts for:
    - Slice positioning and orientation (rigid transformations)
    - Slice thickness blur (PSF)
    - Partial volume effects

    Args:
        transforms: (N, 3, 4) rigid transformations for N slices
        vol: (1, 1, D, H, W) input volume to sample from
        vol_mask: Optional (1, 1, D, H, W) binary mask for volume
        slices_mask: Optional (N, 1, H_s, W_s) binary masks for slices
        psf: (D_psf, H_psf, W_psf) Point Spread Function kernel
        slice_shape: (H_s, W_s) shape of output slices
        res_slice: Slice resolution (in-plane pixel spacing)
        need_weight: If True, also return weight map (for debugging)

    Returns:
        slices: (N, 1, H_s, W_s) simulated 2D slices
        weights: (N, 1, H_s, W_s) normalization weights (if need_weight=True)

    Note:
        - Uses sparse matrix representation for memory efficiency
        - Automatically handles GPU memory constraints via batch processing
        - Falls back to faster no-PSF version if PSF is trivial (single voxel)
        - Normalizes by weights to account for partial volume effects

    Example:
        >>> transforms = torch.randn(10, 3, 4)  # 10 slice transformations
        >>> vol = torch.randn(1, 1, 128, 128, 128)  # Volume to sample
        >>> psf = create_gaussian_psf(sigma=1.5)  # Blur kernel
        >>> slices = slice_acquisition_torch(
        ...     transforms, vol, None, None, psf, (128, 128), 1.0, False
        ... )
        >>> print(slices.shape)  # (10, 1, 128, 128)
    """
    slice_shape = tuple(slice_shape)
    global BATCH_SIZE

    # Fast path: if PSF is trivial (single point) and weights not needed,
    # use trilinear interpolation instead of sparse matrix multiply
    if psf.numel() == 1 and need_weight == False:
        return slice_acquisition_no_psf_torch(
            transforms, vol, vol_mask, slices_mask, slice_shape, res_slice
        )

    # Apply volume mask to zero out background voxels
    if vol_mask is not None:
        vol = vol * vol_mask

    vol_shape = vol.shape[-3:]
    _slices = []
    _weights = []
    i = 0

    # Process slices in batches to manage memory usage
    while i < transforms.shape[0]:
        succ = False
        try:
            # Construct sparse coefficient matrix for this batch of slices
            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            )

            # Matrix-vector multiply: slices = coef @ vol
            s = torch.mv(coef, vol.view(-1)).to_dense().reshape((-1, 1) + slice_shape)

            # Compute normalization weights (sum of PSF coefficients per pixel)
            weight = torch.sparse.sum(coef, 1).to_dense().reshape_as(s)
            del coef
            succ = True

        except RuntimeError as e:
            # Handle GPU out-of-memory by reducing batch size
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                logging.debug("OOM, reduce batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise e

        if succ:
            _slices.append(s)
            _weights.append(weight)
            i += BATCH_SIZE

    # Concatenate all batches
    slices = torch.cat(_slices)
    weights = torch.cat(_weights)

    # Normalize by weights where significant (avoid division by zero)
    m = weights > 1e-2
    slices[m] = slices[m] / weights[m]

    # Apply slice masks to zero out background pixels
    if slices_mask is not None:
        slices = slices * slices_mask

    if need_weight:
        return slices, weights
    else:
        return slices


def slice_acquisition_adjoint_torch(
    transforms: torch.Tensor,
    psf: torch.Tensor,
    slices: torch.Tensor,
    slices_mask: Optional[torch.Tensor],
    vol_mask: Optional[torch.Tensor],
    vol_shape: Sequence,
    res_slice: float,
    equalize: bool,
):
    """
    Adjoint slice acquisition: back-project 2D slices into 3D volume space.

    This is the adjoint (transpose) of the forward slice acquisition operator.
    It's used in gradient-based optimization for reconstruction:
    - Computes the gradient of the data term with respect to the volume
    - Back-projects residuals from image space to volume space

    Mathematical relation:
        If forward is: s = A * v
        Then adjoint is: v = A^T * s
        where A is the slice acquisition matrix

    Args:
        transforms: (N, 3, 4) rigid transformations for N slices
        psf: (D_psf, H_psf, W_psf) Point Spread Function kernel
        slices: (N, 1, H_s, W_s) input slices to back-project
        slices_mask: Optional (N, 1, H_s, W_s) binary masks for slices
        vol_mask: Optional (1, 1, D, H, W) binary mask for volume
        vol_shape: (D, H, W) shape of output volume
        res_slice: Slice resolution (in-plane pixel spacing)
        equalize: If True, normalize by sum of weights (like filtered back-projection)

    Returns:
        vol: (1, 1, D, H, W) reconstructed volume

    Note:
        - This is NOT the inverse operation (that would require solving a system)
        - It's the transpose, which is computationally much faster
        - Used in iterative optimization: vol += learning_rate * adjoint(slices - forward(vol))
        - Equalization helps distribute contributions evenly across the volume

    Example:
        >>> transforms = torch.randn(10, 3, 4)
        >>> slices = torch.randn(10, 1, 128, 128)  # Slice residuals
        >>> psf = create_gaussian_psf(sigma=1.5)
        >>> vol = slice_acquisition_adjoint_torch(
        ...     transforms, psf, slices, None, None, (128, 128, 128), 1.0, True
        ... )
        >>> print(vol.shape)  # (1, 1, 128, 128, 128)
    """
    vol_shape = tuple(vol_shape)
    global BATCH_SIZE

    # Apply slice masks to zero out background pixels
    if slices_mask is not None:
        slices = slices * slices_mask

    vol = None
    weight = None
    slice_shape = slices.shape[-2:]
    i = 0

    # Process slices in batches to manage memory usage
    while i < transforms.shape[0]:
        succ = False
        try:
            # Construct transpose of coefficient matrix for this batch
            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            ).t()  # Transpose for adjoint operation

            # Matrix-vector multiply: vol += coef^T @ slices
            v = torch.mv(coef, slices[i : i + BATCH_SIZE].view(-1))

            # If equalizing, also accumulate weights for normalization
            if equalize:
                w = torch.sparse.sum(coef, 1)
            del coef
            succ = True

        except RuntimeError as e:
            # Handle GPU out-of-memory by reducing batch size
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                logging.debug("OOM, reduce batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise e

        if succ:
            # Accumulate contributions from all batches
            if vol is None:
                vol = v
            else:
                vol += v
            if equalize:
                if weight is None:
                    weight = w
                else:
                    weight += w
            i += BATCH_SIZE

    # Convert from sparse to dense and reshape
    vol = cast(torch.Tensor, vol)
    vol = vol.to_dense().reshape((1, 1) + vol_shape)

    # Equalize: normalize by sum of weights (like FBP filter)
    if equalize:
        weight = cast(torch.Tensor, weight)
        weight = weight.to_dense().reshape_as(vol)
        m = weight > 1e-2
        vol[m] = vol[m] / weight[m]

    # Apply volume mask to zero out background voxels
    if vol_mask is not None:
        vol = vol * vol_mask

    return vol


def slice_acquisition_no_psf_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    slice_shape: Sequence,
    res_slice: float,
) -> torch.Tensor:
    """
    Fast slice acquisition without PSF using trilinear interpolation.

    This is a simplified version of slice_acquisition_torch that doesn't model
    slice thickness blur (PSF). Instead, it uses PyTorch's grid_sample for fast
    trilinear interpolation. This is useful when:
    - Slice thickness is negligible (thin slices)
    - Speed is more important than accuracy
    - PSF is not available or not needed

    Args:
        transforms: (N, 3, 4) rigid transformations for N slices
        vol: (1, 1, D, H, W) input volume to sample from
        vol_mask: Optional (1, 1, D, H, W) binary mask for volume
        slices_mask: Optional (N, 1, H_s, W_s) binary masks for slices
        slice_shape: (H_s, W_s) shape of output slices
        res_slice: Slice resolution (in-plane pixel spacing)

    Returns:
        output_slices: (N, 1, H_s, W_s) simulated 2D slices

    Note:
        - Much faster than full PSF-based acquisition
        - Uses PyTorch's grid_sample (hardware-accelerated)
        - Automatically handles out-of-bounds coordinates (returns 0)
        - Uses trilinear interpolation (smoother than nearest neighbor)

    Algorithm:
        1. Generate coordinates for all slice pixels
        2. Transform to volume coordinate system
        3. Normalize to [-1, 1] range (required by grid_sample)
        4. Sample volume at transformed coordinates
        5. Apply masks if provided

    Example:
        >>> transforms = torch.randn(10, 3, 4)
        >>> vol = torch.randn(1, 1, 128, 128, 128)
        >>> slices = slice_acquisition_no_psf_torch(
        ...     transforms, vol, None, None, (128, 128), 1.0
        ... )
        >>> print(slices.shape)  # (10, 1, 128, 128)
    """
    slice_shape = tuple(slice_shape)
    device = transforms.device

    # Create dummy slice to get pixel coordinates
    _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=device)
    slice_xyz = Slice(_slice, _slice, resolution_x=res_slice).xyz_masked_untransformed

    # Transform slice coordinates to volume space
    # Broadcasting: (N, 1, 3, 4) @ (1, n_pixels, 3) -> (N, n_pixels, 3)
    slice_xyz = mat_transform_points(
        transforms[:, None], slice_xyz[None], trans_first=True
    ).view((transforms.shape[0], 1) + slice_shape + (3,))

    # Allocate output
    output_slices = torch.zeros_like(slice_xyz[..., 0])

    # If using slice masks, work only with masked pixels
    if slices_mask is not None:
        masked_xyz = slice_xyz[slices_mask]
    else:
        masked_xyz = slice_xyz

    # Normalize coordinates to [-1, 1] range (required by grid_sample)
    # grid_sample expects coordinates in normalized device coordinates
    masked_xyz = masked_xyz / (
        (torch.tensor(vol.shape[-3:][::-1], dtype=masked_xyz.dtype, device=device) - 1)
        / 2
    )

    # Apply volume mask
    if vol_mask is not None:
        vol = vol * vol_mask

    # Sample volume using trilinear interpolation
    # grid_sample expects (N, C, D, H, W) volume and (N, D_out, H_out, W_out, 3) coordinates
    # We reshape to (1, 1, 1, n_pixels, 3) for efficient sampling
    masked_v = F.grid_sample(vol, masked_xyz.view(1, 1, 1, -1, 3), align_corners=True)

    # Put sampled values back into output slices
    if slices_mask is not None:
        output_slices[slices_mask] = masked_v
    else:
        output_slices = masked_v.reshape((transforms.shape[0], 1) + slice_shape)

    return output_slices
