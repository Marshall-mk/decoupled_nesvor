"""
Point Dataset for Implicit Neural Representation Training

This module provides a dataset class for training coordinate-based neural networks
on medical imaging data. It's specifically designed for MRI super-resolution using
Implicit Neural Representations (INRs).

Key Concepts:
-----------
1. **Point-Based Representation**:
   - Instead of storing images as grids, stores individual sampled points
   - Each point has: (x,y,z) coordinates, intensity value, slice index, transformation
   - Enables continuous representation (can query at any coordinate)

2. **Coordinate-Based Neural Networks**:
   - Network takes 3D coordinates as input, outputs intensity values
   - Represents the entire 3D volume as a function: f(x,y,z) = intensity
   - Can be queried at arbitrary resolution (super-resolution capability)

3. **Multi-Slice Registration**:
   - Each slice has its own rigid transformation
   - Transformations align slices into common 3D space
   - Allows training on misaligned/motion-corrupted data

4. **Memory-Efficient Training**:
   - Processes points in random batches (not entire images)
   - Automatically shuffles data every epoch
   - Scales to large datasets that don't fit in GPU memory

Mathematical Formulation:
-----------------------
Given N slices with transformations T_i:
    For each point p in slice i:
        - Local coords: (x_local, y_local, z_local=0)
        - World coords: (x, y, z) = T_i(x_local, y_local, 0)
        - Network learns: f(x, y, z) = intensity

Training objective:
    min Σ_p ||f(transform(p)) - observed_intensity(p)||^2

Classes:
-------
PointDataset: Main dataset class for coordinate-based training
    - Stores all slice data as flattened point arrays
    - Provides batching and shuffling
    - Computes bounding box and mask for the volume
"""

from typing import Dict, List
import torch
from utils import gaussian_blur, Volume, Slice, RigidTransform, transform_points


class PointDataset(object):
    """
    Dataset for training coordinate-based neural networks on multi-slice MRI data.

    This class aggregates multiple 2D slices (with their transformations) into a
    single point-based dataset suitable for training Implicit Neural Representations.
    Each point represents a pixel from one of the input slices, stored with its
    local coordinates, intensity value, slice index, and transformation.

    The dataset is flattened into arrays for efficient batching:
        - xyz: (N_total,) array of 3D coordinates (in slice-local space)
        - v: (N_total,) array of intensity values
        - slice_idx: (N_total,) array indicating which slice each point belongs to
        - transformation: (N_slices,) array of rigid transformations
        - resolution: (N_slices, 3) array of per-slice resolutions

    Attributes:
        xyz: (N_total, 3) local coordinates of all points from all slices
        v: (N_total,) intensity values at each point
        slice_idx: (N_total,) index of which slice each point came from
        transformation: RigidTransform with N_slices transformations
        resolution: (N_slices, 3) resolution for each slice (x, y, z)
        count: Current position in dataset (for batching)
        epoch: Current epoch number (increments when dataset wraps)
        mask_threshold: Threshold for computing volume mask (default=1)

    Example:
        >>> # Load multiple slices
        >>> slices = [load_slice(f) for f in slice_files]
        >>>
        >>> # Create dataset
        >>> dataset = PointDataset(slices)
        >>>
        >>> # Get batch for training
        >>> batch = dataset.get_batch(batch_size=4096, device='cuda')
        >>> # batch contains: {'xyz': coords, 'v': intensities, 'slice_idx': indices}
        >>>
        >>> # Transform to world coordinates
        >>> xyz_world = dataset.xyz_transformed  # Uses transformations
        >>>
        >>> # Get bounding box for network domain
        >>> bbox = dataset.bounding_box  # (2, 3) tensor: [min_xyz, max_xyz]
        >>>
        >>> # Compute volume mask (region with data)
        >>> mask_volume = dataset.mask  # Volume object with binary mask
    """

    def __init__(self, slices: List[Slice]) -> None:
        """
        Initialize dataset from a list of Slice objects.

        Args:
            slices: List of Slice objects, each containing:
                - xyz_masked_untransformed: (N_i, 3) local coordinates
                - v_masked: (N_i,) intensity values
                - transformation: RigidTransform for this slice
                - resolution_xyz: (3,) pixel spacing in x, y, z

        Note:
            - Only masked (foreground) pixels are included
            - All data is concatenated into flat arrays
            - Total points N_total = Σ_i N_i
        """
        self.mask_threshold = 1  # Threshold for volume mask computation

        # Lists to accumulate data from all slices
        xyz_all = []
        v_all = []
        slice_idx_all = []
        transformation_all = []
        resolution_all = []

        # Extract data from each slice
        for i, slice in enumerate(slices):
            # Get local coordinates (in slice's own coordinate system)
            xyz = slice.xyz_masked_untransformed  # (N_i, 3)

            # Get intensity values
            v = slice.v_masked  # (N_i,)

            # Create slice index array (all points from this slice get index i)
            slice_idx = torch.full(v.shape, i, device=v.device)  # (N_i,)

            # Append to lists
            xyz_all.append(xyz)
            v_all.append(v)
            slice_idx_all.append(slice_idx)
            transformation_all.append(slice.transformation)
            resolution_all.append(slice.resolution_xyz)

        # Concatenate all slices into single arrays
        self.xyz = torch.cat(xyz_all)  # (N_total, 3)
        self.v = torch.cat(v_all)  # (N_total,)
        self.slice_idx = torch.cat(slice_idx_all)  # (N_total,)

        # Stack transformations and resolutions (one per slice, not per point)
        self.transformation = RigidTransform.cat(transformation_all)  # (N_slices,)
        self.resolution = torch.stack(resolution_all, 0)  # (N_slices, 3)

        # Initialize batch counter
        self.count = self.v.shape[0]
        self.epoch = 0

    @property
    def bounding_box(self) -> torch.Tensor:
        """
        Compute axis-aligned bounding box of all points in world coordinates.

        This defines the spatial extent of the data, used to:
        - Set the domain for coordinate-based networks
        - Normalize coordinates to a standard range
        - Determine output volume size

        Returns:
            (2, 3) tensor: [[x_min, y_min, z_min],
                           [x_max, y_max, z_max]]

        Note:
            - Adds padding of 2 * max_resolution on all sides
            - This ensures full PSF support at boundaries
            - Coordinates are in world space (after transformations)
        """
        # Get maximum resolution for padding
        max_r = self.resolution.max()

        # Transform all points to world coordinates
        xyz_transformed = self.xyz_transformed  # (N_total, 3)

        # Find min/max along each axis, add padding
        xyz_min = xyz_transformed.amin(0) - 2 * max_r  # (3,)
        xyz_max = xyz_transformed.amax(0) + 2 * max_r  # (3,)

        # Stack into bounding box
        bounding_box = torch.stack([xyz_min, xyz_max], 0)  # (2, 3)

        return bounding_box

    @property
    def mean(self) -> float:
        """
        Compute robust mean intensity (trimmed mean, excluding outliers).

        Uses the inter-decile range (10th to 90th percentile) to avoid
        influence of outliers, background noise, or imaging artifacts.

        Returns:
            Scalar mean intensity value

        Note:
            - Only computes on first 256^3 points if dataset is very large
            - This is used for normalizing network inputs/outputs
            - More robust than simple mean for medical images
        """
        # Use subset of data if very large (for speed)
        v_subset = (
            self.v if self.v.numel() < 256 * 256 * 256 else self.v[: 256 * 256 * 256]
        )

        # Compute 10th and 90th percentiles
        q1, q2 = torch.quantile(
            v_subset,
            torch.tensor([0.1, 0.9], dtype=self.v.dtype, device=self.v.device),
        )

        # Compute mean of values in inter-decile range
        return self.v[torch.logical_and(self.v > q1, self.v < q2)].mean().item()

    def get_batch(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        """
        Get a batch of points for training.

        Implements epoch-based random batching:
        1. During an epoch: returns sequential batches without shuffling
        2. At epoch end: shuffles all data and restarts from beginning
        3. Increments epoch counter each time dataset wraps around

        Args:
            batch_size: Number of points to return
            device: Device to place batch tensors on (usually 'cuda')

        Returns:
            Dictionary with keys:
                - 'xyz': (batch_size, 3) local coordinates
                - 'v': (batch_size,) intensity values
                - 'slice_idx': (batch_size,) slice indices

        Note:
            - Automatically shuffles when reaching end of dataset
            - Maintains synchronized order across xyz, v, slice_idx
            - Used in training loop: for epoch in epochs: for batch in dataset: ...

        Example:
            >>> dataset = PointDataset(slices)
            >>> for epoch in range(100):
            ...     while dataset.epoch == epoch:  # Until wrapped to next epoch
            ...         batch = dataset.get_batch(4096, 'cuda')
            ...         loss = train_step(batch)
        """
        # Check if we need to start a new epoch
        if self.count + batch_size > self.xyz.shape[0]:
            # Reset counter and increment epoch
            self.count = 0
            self.epoch += 1

            # Shuffle all arrays in synchronized manner
            idx = torch.randperm(self.xyz.shape[0], device=device)
            self.xyz = self.xyz[idx]
            self.v = self.v[idx]
            self.slice_idx = self.slice_idx[idx]

        # Extract batch at current position
        batch = {
            "xyz": self.xyz[self.count : self.count + batch_size],
            "v": self.v[self.count : self.count + batch_size],
            "slice_idx": self.slice_idx[self.count : self.count + batch_size],
        }

        # Advance counter for next batch
        self.count += batch_size

        return batch

    @property
    def xyz_transformed(self) -> torch.Tensor:
        """
        Transform all points from local to world coordinates.

        For each point, applies its corresponding slice's rigid transformation:
            p_world = T[slice_idx[i]](p_local[i])

        Returns:
            (N_total, 3) tensor of world coordinates

        Note:
            - Uses slice_idx to look up correct transformation for each point
            - Transformations are applied in batch for efficiency
            - This is the "registration" step - aligning all slices

        Example:
            >>> xyz_local = dataset.xyz  # Local coordinates
            >>> xyz_world = dataset.xyz_transformed  # After rigid alignment
            >>> # Network is trained on xyz_world, evaluated on arbitrary coords
        """
        return transform_points(
            self.transformation[self.slice_idx],  # (N_total, 3, 4) transformations
            self.xyz,  # (N_total, 3) local coordinates
        )

    @property
    def mask(self) -> Volume:
        """
        Compute a 3D binary mask indicating the region with data coverage.

        This creates a Volume object representing where the slices provide data.
        Used for:
        - Masking out regions with no observations during reconstruction
        - Initializing volume for iterative algorithms
        - Visualizing data coverage

        Algorithm:
        1. Transform all points to world coordinates
        2. Compute bounding box with padding
        3. Discretize space at finest resolution
        4. Count points falling in each voxel (histogram in 3D)
        5. Apply Gaussian blur (spatial smoothing)
        6. Threshold to create binary mask

        Returns:
            Volume object with:
                - data: (D, H, W) binary mask (1 = data, 0 = no data)
                - mask: Same as data (all True where data exists)
                - transformation: Centers the volume in world space
                - resolution: Finest resolution from input slices

        Note:
            - Voxel size = minimum resolution across all slices
            - Threshold adapts to point density (accounts for variable slice thickness)
            - Gaussian blur prevents small holes/islands in mask

        Example:
            >>> dataset = PointDataset(slices)
            >>> mask_vol = dataset.mask
            >>> print(mask_vol.data.shape)  # e.g., (128, 128, 128)
            >>> coverage = mask_vol.data.sum() / mask_vol.data.numel()
            >>> print(f"Data covers {coverage*100:.1f}% of volume")
        """
        with torch.no_grad():  # Don't track gradients (this is just preprocessing)
            # Get resolution range
            resolution_min = self.resolution.min()  # Finest resolution (smallest voxel)
            resolution_max = (
                self.resolution.max()
            )  # Coarsest resolution (largest voxel)

            # Transform points to world coordinates
            xyz = self.xyz_transformed  # (N_total, 3)

            # Compute bounding box with generous padding
            xyz_min = xyz.amin(0) - resolution_max * 10  # (3,)
            xyz_max = xyz.amax(0) + resolution_max * 10  # (3,)

            # Compute shape at finest resolution
            shape_xyz = ((xyz_max - xyz_min) / resolution_min).ceil().long()  # (3,)
            shape = (
                int(shape_xyz[2]),
                int(shape_xyz[1]),
                int(shape_xyz[0]),
            )  # (D, H, W)

            # Convert coordinates to voxel indices
            kji = ((xyz - xyz_min) / resolution_min).round().long()  # (N_total, 3)

            # Create histogram: count points in each voxel
            # Flatten 3D indices to 1D: idx = x + W*y + W*H*z
            mask = torch.bincount(
                kji[..., 0]
                + shape[2] * kji[..., 1]
                + shape[2] * shape[1] * kji[..., 2],
                minlength=shape[0] * shape[1] * shape[2],
            )

            # Reshape to 3D volume with batch/channel dims
            mask = mask.view((1, 1) + shape).float()  # (1, 1, D, H, W)

            # Compute adaptive threshold
            # Account for different resolutions: normalize by expected points per voxel
            mask_threshold = (
                self.mask_threshold
                * resolution_min**3  # Volume of fine voxel
                / self.resolution.log().mean().exp()
                ** 3  # Geometric mean of resolutions^3
            )
            # Also account for non-uniform coverage
            mask_threshold *= mask.sum() / (mask > 0).sum()

            # Apply Gaussian blur to smooth mask (fill small gaps, remove islands)
            assert len(mask.shape) == 5  # Verify shape (1, 1, D, H, W)
            mask = (
                gaussian_blur(
                    mask,
                    (resolution_max / resolution_min).item(),  # Blur sigma in voxels
                    3,  # Blur in 3D
                )
                > mask_threshold  # Threshold to binary
            )[0, 0]  # Remove batch/channel dims -> (D, H, W)

            # Compute center of volume in world coordinates
            xyz_c = xyz_min + (shape_xyz - 1) / 2 * resolution_min  # (3,)

            # Create Volume object
            # Transformation is just translation to center the volume
            return Volume(
                mask.float(),  # Data: binary mask as float
                mask,  # Mask: same as data (all True where mask=1)
                RigidTransform(
                    torch.cat([0 * xyz_c, xyz_c])[None],  # [0, 0, 0, tx, ty, tz]
                    True,  # trans_first
                ),
                resolution_min,  # x resolution
                resolution_min,  # y resolution
                resolution_min,  # z resolution
            )
