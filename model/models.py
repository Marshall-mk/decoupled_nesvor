"""
Neural Network Models for NeSVoR MRI Super-Resolution

This module implements the core neural network architectures for NeSVoR:
- INR: Implicit Neural Representation for 3D volume
- DeformNet: Deformable registration network
- NeSVoR: Complete model integrating INR, registration, bias field, and uncertainty

Key Concepts:
-----------
1. **Implicit Neural Representation (INR)**:
   - Represents 3D volume as a continuous function: f(x,y,z) = intensity
   - Hash grid encoding + MLP decoder
   - Enables super-resolution beyond input data resolution

2. **Joint Optimization**:
   - Simultaneously optimizes:
     a) Volume representation (INR weights)
     b) Slice transformations (rigid 6-DOF)
     c) Bias field (intensity correction)
     d) Uncertainty (heteroscedastic variance)
     e) Deformations (non-rigid motion, optional)

3. **Uncertainty Modeling**:
   - Per-slice variance: global noise level for each slice
   - Per-pixel variance: spatially-varying uncertainty
   - Enables robust reconstruction despite corrupted data

4. **Bias Field Correction**:
   - Learned multiplicative bias field
   - Corrects for scanner intensity inhomogeneity
   - Prevents bias from being baked into reconstruction

5. **Deformable Motion**:
   - Optional non-rigid deformation per slice
   - Hash grid + MLP predicting displacement field
   - Regularized to ensure smoothness

Architecture Overview:
--------------------
NeSVoR = {
    INR:
        encoding: HashGrid (multi-resolution positional encoding)
        density_net: MLP → (intensity, features)

    Per-Slice Parameters:
        transformation: 6-DOF rigid (optimized)
        slice_embedding: Slice-specific features
        logit_coef: Slice intensity scaling
        log_var_slice: Per-slice noise variance

    Optional Networks:
        sigma_net: MLP → per-pixel variance
        b_net: MLP → bias field
        deform_net: DeformNet → displacement field
}

Loss Components:
--------------
1. D_LOSS (Data): Weighted MSE between predicted and observed intensities
2. S_LOSS (Variance): Log-likelihood term for learned variances
3. T_REG (Transformation): Keep transforms close to initialization
4. B_REG (Bias): Prevent bias field from growing too large
5. I_REG (Image): Total Variation / Edge-preserving / L2 smoothness
6. D_REG (Deform): Smoothness of deformation field (Jacobian regularization)

Classes:
-------
INR: Basic implicit neural representation (hash encoding + MLP)
DeformNet: Deformable transformation network
NeSVoR: Complete model with all components

Functions:
---------
build_encoding: Create hash grid encoder (tinycudann or PyTorch)
build_network: Create MLP network (tinycudann or PyTorch)
compute_resolution_nlevel: Compute hash grid parameters from bounding box

Constants:
---------
D_LOSS, S_LOSS, DS_LOSS: Data fidelity loss keys
B_REG, T_REG, I_REG, D_REG: Regularization loss keys
"""

from argparse import Namespace
from math import log2
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, Tuple
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.hash_grid_torch import HashEmbedder
from utils import (
    resolution2sigma,
    RigidTransform,
    ax_transform_points,
    mat_transform_points,
)

# Try to use tinycudann (CUDA-accelerated) if available, fallback to PyTorch
USE_TORCH = False

if not USE_TORCH:
    try:
        import tinycudann as tcnn
    except:
        logging.warning("Fail to load tinycudann. Will use pytorch implementation.")
        USE_TORCH = True


# Loss and regularization component keys
D_LOSS = "MSE"  # Data fidelity loss (weighted MSE)
S_LOSS = "logVar"  # Slice variance loss (log-likelihood term)
DS_LOSS = "MSE+logVar"  # Combined data + variance loss
B_REG = "biasReg"  # Bias field regularization
T_REG = "transReg"  # Transformation regularization
I_REG = "imageReg"  # Image smoothness regularization (TV/edge/L2)
D_REG = "deformReg"  # Deformation field regularization


def build_encoding(**config):
    """
    Create hash grid encoding layer (tinycudann or PyTorch fallback).

    Args:
        **config: Configuration dictionary with keys:
            - n_input_dims: Input dimension (3 for 3D coordinates)
            - otype: Encoding type ("HashGrid")
            - n_levels: Number of resolution levels
            - n_features_per_level: Features per level (typically 2)
            - log2_hashmap_size: log2 of hash table size (typically 19)
            - base_resolution: Coarsest grid resolution
            - per_level_scale: Scale factor between levels
            - dtype: torch.float16 or torch.float32

    Returns:
        Encoding module (tcnn.Encoding or HashEmbedder)

    Note:
        - Prefers tinycudann for speed (10-100x faster)
        - Falls back to PyTorch implementation if tcnn unavailable
        - tinycudann requires CUDA and may not support half precision
    """
    if USE_TORCH:
        # PyTorch implementation (slower but always available)
        encoding = HashEmbedder(**config)
    else:
        # Extract parameters for tinycudann
        n_input_dims = config.pop("n_input_dims")
        dtype = config.pop("dtype")
        try:
            # Tinycudann implementation (fast CUDA kernels)
            encoding = tcnn.Encoding(
                n_input_dims=n_input_dims, encoding_config=config, dtype=dtype
            )
        except RuntimeError as e:
            if "TCNN was not compiled with half-precision support" in str(e):
                logging.error(
                    "TCNN was not compiled with half-precision support! "
                    "Try using --single-precision in the nesvor command! "
                )
            raise e
    return encoding


def build_network(**config):
    """
    Create MLP network (tinycudann or PyTorch fallback).

    Args:
        **config: Configuration dictionary with keys:
            - n_input_dims: Input feature dimension
            - n_output_dims: Output dimension
            - activation: Hidden layer activation ("ReLU", "Tanh", "None")
            - output_activation: Final layer activation ("None" typically)
            - n_neurons: Hidden layer width
            - n_hidden_layers: Number of hidden layers
            - dtype: torch.float16 or torch.float32

    Returns:
        MLP module (tcnn.Network or nn.Sequential)

    Note:
        - Uses CutlassMLP from tinycudann if available and FP16
        - Falls back to PyTorch Sequential for CPU or FP32
        - Activation can be None, ReLU, Tanh, etc.
    """
    dtype = config.pop("dtype")
    assert dtype == torch.float16 or dtype == torch.float32

    # Use tinycudann's fast CutlassMLP for FP16 on GPU
    if dtype == torch.float16 and not USE_TORCH:
        return tcnn.Network(
            n_input_dims=config["n_input_dims"],
            n_output_dims=config["n_output_dims"],
            network_config={
                "otype": "CutlassMLP",
                "activation": config["activation"],
                "output_activation": config["output_activation"],
                "n_neurons": config["n_neurons"],
                "n_hidden_layers": config["n_hidden_layers"],
            },
        )
    else:
        # PyTorch implementation
        # Get activation functions
        activation = (
            None
            if config["activation"] == "None"
            else getattr(nn, config["activation"])
        )
        output_activation = (
            None
            if config["output_activation"] == "None"
            else getattr(nn, config["output_activation"])
        )

        # Build Sequential model
        models = []
        if config["n_hidden_layers"] > 0:
            # Input layer
            models.append(nn.Linear(config["n_input_dims"], config["n_neurons"]))

            # Hidden layers
            for _ in range(config["n_hidden_layers"] - 1):
                if activation is not None:
                    models.append(activation())
                models.append(nn.Linear(config["n_neurons"], config["n_neurons"]))

            # Output layer
            if activation is not None:
                models.append(activation())
            models.append(nn.Linear(config["n_neurons"], config["n_output_dims"]))
        else:
            # No hidden layers: direct input → output
            models.append(nn.Linear(config["n_input_dims"], config["n_output_dims"]))

        # Final activation
        if output_activation is not None:
            models.append(output_activation())

        return nn.Sequential(*models)


def compute_resolution_nlevel(
    bounding_box: torch.Tensor,
    coarsest_resolution: float,
    finest_resolution: float,
    level_scale: float,
    spatial_scaling: float,
) -> Tuple[int, int]:
    """
    Compute hash grid parameters (base resolution and number of levels).

    Given desired coarsest and finest resolutions, computes:
    - base_resolution: Grid size at coarsest level
    - n_levels: Number of resolution levels

    The hash grid uses geometric progression:
        resolution_l = base_resolution * level_scale^l

    Args:
        bounding_box: (2, 3) bounding box [[x_min, y_min, z_min],
                                          [x_max, y_max, z_max]]
        coarsest_resolution: Coarsest resolution (mm) - largest voxel size
        finest_resolution: Finest resolution (mm) - smallest voxel size
        level_scale: Scale factor between levels (typically 1.39-2.0)
        spatial_scaling: Coordinate scaling factor (typically 30.0)

    Returns:
        (base_resolution, n_levels) tuple:
            - base_resolution: Grid size at level 0
            - n_levels: Number of levels

    Example:
        >>> bbox = torch.tensor([[-30, -30, -30], [30, 30, 30]])
        >>> base_res, n_levels = compute_resolution_nlevel(
        ...     bbox, coarsest_resolution=2.0, finest_resolution=0.5,
        ...     level_scale=1.5, spatial_scaling=30.0
        ... )
        >>> # base_res ≈ 30 (grid at coarsest level)
        >>> # n_levels ≈ 10 (enough levels to reach finest resolution)
        >>> # Finest grid: 30 * 1.5^9 ≈ 1152 (resolution = 60mm / 1152 ≈ 0.05mm)

    Note:
        - Computes based on longest dimension of bounding box
        - Both base_resolution and n_levels are rounded up
        - Actual finest resolution may be finer than requested
    """
    # Compute base resolution from coarsest setting
    # base_resolution = max_extent / coarsest_resolution
    base_resolution = (
        (
            (bounding_box[1] - bounding_box[0]).max()
            * spatial_scaling
            / coarsest_resolution
        )
        .ceil()
        .int()
        .item()
    )

    # Compute number of levels needed to reach finest resolution
    # finest_resolution = max_extent / (base_resolution * level_scale^(n_levels-1))
    # Solving for n_levels:
    # n_levels = log(max_extent / (finest_resolution * base_resolution)) / log(level_scale) + 1
    n_levels = (
        (
            torch.log2(
                (bounding_box[1] - bounding_box[0]).max()
                * spatial_scaling
                / finest_resolution
                / base_resolution
            )
            / log2(level_scale)
            + 1
        )
        .ceil()
        .int()
        .item()
    )

    return int(base_resolution), int(n_levels)


class INR(nn.Module):
    """
    Implicit Neural Representation for 3D medical volume.

    This is the core volume representation: a neural network that maps
    3D coordinates to intensity values. It uses:
    - Hash grid encoding for multi-resolution positional encoding
    - MLP decoder for mapping encoded features to intensity

    The network represents the volume as a continuous function:
        f(x, y, z) → intensity

    This enables:
    - Super-resolution (query at arbitrary resolution)
    - Smooth interpolation between voxels
    - Memory-efficient representation

    Attributes:
        bounding_box: (2, 3) tensor defining spatial extent
        encoding: Hash grid encoder (HashEmbedder or tcnn.Encoding)
        density_net: MLP decoder (nn.Sequential or tcnn.Network)

    Example:
        >>> bbox = torch.tensor([[-30, -30, -30], [30, 30, 30]])
        >>> inr = INR(bbox, args, spatial_scaling=30.0)
        >>>
        >>> # Query at random 3D points
        >>> xyz = torch.randn(1000, 3) * 20  # Points in [-20, 20]^3
        >>> intensities = inr(xyz)  # (1000,) intensity values
        >>>
        >>> # Query with PSF sampling
        >>> xyz_psf = inr.sample_batch(
        ...     xyz, transformation=None, psf_sigma=1.0, n_samples=128
        ... )
        >>> intensities_blurred = inr(xyz_psf).mean(-1)
    """

    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        """
        Initialize INR model.

        Args:
            bounding_box: (2, 3) spatial extent [[x_min, y_min, z_min],
                                                 [x_max, y_max, z_max]]
            args: Hyperparameters namespace with:
                - coarsest_resolution: Coarsest hash grid resolution (mm)
                - finest_resolution: Finest hash grid resolution (mm)
                - level_scale: Scale between hash grid levels
                - n_features_per_level: Features per level (typically 2)
                - log2_hashmap_size: Hash table size (19 → 512K entries)
                - width: MLP hidden layer width
                - depth: MLP number of hidden layers
                - n_features_z: Additional output features (for variance net)
                - dtype: torch.float16 or torch.float32
                - img_reg_autodiff: Use autodiff for image regularization
            spatial_scaling: Coordinate scaling factor (typically 30.0)

        Note:
            - Coordinates are normalized to [0, 1]^3 before encoding
            - Output density is softplus-activated (always positive)
            - In training mode, returns (density, encoding, features)
            - In eval mode, returns only density
        """
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        self.register_buffer("bounding_box", bounding_box)

        # Compute hash grid parameters
        base_resolution, n_levels = compute_resolution_nlevel(
            self.bounding_box,
            args.coarsest_resolution,
            args.finest_resolution,
            args.level_scale,
            spatial_scaling,
        )

        # Build hash grid encoding
        self.encoding = build_encoding(
            n_input_dims=3,
            otype="HashGrid",
            n_levels=n_levels,
            n_features_per_level=args.n_features_per_level,
            log2_hashmap_size=args.log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=args.level_scale,
            dtype=args.dtype,
        )

        # Build density MLP decoder
        # Output: 1 (density) + n_features_z (for variance prediction)
        self.density_net = build_network(
            n_input_dims=n_levels * args.n_features_per_level,
            n_output_dims=1 + args.n_features_z,
            activation="ReLU",
            output_activation="None",
            n_neurons=args.width,
            n_hidden_layers=args.depth,
            dtype=torch.float32 if args.img_reg_autodiff else args.dtype,
        )

        # Log hyperparameters
        logging.debug(
            "hyperparameters for hash grid encoding: "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * args.level_scale ** (n_levels - 1)),
            args.level_scale,
            n_levels,
        )
        logging.debug(
            "bounding box for reconstruction (mm): "
            + "x=[%f, %f], y=[%f, %f], z=[%f, %f]",
            self.bounding_box[0, 0],
            self.bounding_box[1, 0],
            self.bounding_box[0, 1],
            self.bounding_box[1, 1],
            self.bounding_box[0, 2],
            self.bounding_box[1, 2],
        )

    def forward(self, x: torch.Tensor):
        """
        Predict intensity at 3D coordinates.

        Args:
            x: (..., 3) 3D coordinates in world space (mm)

        Returns:
            Training mode: (density, encoding, features)
                - density: (...,) intensity values (softplus-activated, positive)
                - encoding: (..., n_levels * n_features_per_level) hash grid features
                - features: (..., 1 + n_features_z) MLP output before softplus

            Eval mode: density (...,) intensity values only

        Note:
            - Coordinates normalized to [0, 1]^3 before encoding
            - Uses softplus activation: density = softplus(z[0])
            - Softplus ensures positive intensities (physical constraint)
        """
        # Normalize coordinates to [0, 1]^3
        x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])

        # Store shape for reshaping output
        prefix_shape = x.shape[:-1]

        # Flatten for batch processing
        x = x.view(-1, x.shape[-1])

        # Hash grid encoding
        pe = self.encoding(x)

        # Convert to float32 if needed (for compatibility)
        if not self.training:
            pe = pe.to(dtype=x.dtype)

        # MLP decoder
        z = self.density_net(pe)

        # Apply softplus to get positive density
        density = F.softplus(z[..., 0].view(prefix_shape))

        if self.training:
            # Return all intermediate values for loss computation
            return density, pe, z
        else:
            # Return only density for inference
            return density

    def sample_batch(
        self,
        xyz: torch.Tensor,
        transformation: Optional[RigidTransform],
        psf_sigma: Union[float, torch.Tensor],
        n_samples: int,
    ) -> torch.Tensor:
        """
        Generate batch of coordinates with PSF sampling and transformation.

        This function:
        1. Adds Gaussian noise for PSF simulation (if n_samples > 1)
        2. Applies rigid transformation (if provided)

        Used for simulating thick-slice acquisition via Monte Carlo sampling.

        Args:
            xyz: (N, 3) coordinates in slice-local space
            transformation: Optional RigidTransform to apply
            psf_sigma: PSF standard deviation (scalar or (N, 3) anisotropic)
            n_samples: Number of PSF samples (1 = no PSF)

        Returns:
            (N, n_samples, 3) coordinates with PSF noise and transformation applied

        Note:
            - If n_samples = 1, returns (N, 1, 3) with no noise
            - If transformation is None, only PSF sampling is applied
            - PSF noise is Gaussian: xyz + N(0, psf_sigma^2)

        Example:
            >>> xyz = torch.randn(100, 3)  # 100 slice pixels
            >>> transform = RigidTransform(torch.randn(1, 6))
            >>> xyz_sampled = inr.sample_batch(
            ...     xyz, transform, psf_sigma=1.5, n_samples=128
            ... )
            >>> # xyz_sampled.shape = (100, 128, 3)
            >>> # Each pixel expanded to 128 PSF samples
        """
        # Add PSF sampling noise
        if n_samples > 1:
            # Ensure psf_sigma has correct shape
            if isinstance(psf_sigma, torch.Tensor):
                psf_sigma = psf_sigma.view(-1, 1, 3)

            # Generate Gaussian random offsets
            xyz_psf = torch.randn(
                xyz.shape[0], n_samples, 3, dtype=xyz.dtype, device=xyz.device
            )

            # Add noise to coordinates
            xyz = xyz[:, None] + xyz_psf * psf_sigma
        else:
            # No PSF: just add dimension
            xyz = xyz[:, None]

        # Apply rigid transformation if provided
        if transformation is not None:
            trans_first = transformation.trans_first
            mat = transformation.matrix(trans_first)
            xyz = mat_transform_points(mat[:, None], xyz, trans_first)

        return xyz


class DeformNet(nn.Module):
    """
    Deformable transformation network for non-rigid motion.

    This network predicts a displacement field that deforms 3D coordinates:
        x_deformed = x + DeformNet(x, embedding)

    Used to model non-rigid motion (breathing, fetal movement, etc.) that
    cannot be captured by rigid transformations alone.

    Architecture:
    - Hash grid encoding (typically coarser than INR)
    - Per-slice embedding (different deformation for each slice)
    - MLP decoder outputting 3D displacement
    - Residual connection: output = input + displacement

    Attributes:
        bounding_box: (2, 3) spatial extent
        encoding: Hash grid encoder for coordinates
        deform_net: MLP decoder for displacement

    Note:
        - Initialized with small weights (~1e-4) for stability
        - Regularized to ensure smoothness (Jacobian penalty)
        - Typically uses coarser grid than main INR
    """

    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        """
        Initialize deformable network.

        Args:
            bounding_box: (2, 3) spatial extent
            args: Hyperparameters with:
                - coarsest_resolution_deform: Coarsest grid resolution
                - finest_resolution_deform: Finest grid resolution
                - level_scale_deform: Scale between levels
                - n_features_per_level_deform: Features per level
                - n_features_deform: Per-slice embedding dimension
                - log2_hashmap_size: Hash table size
                - width: MLP hidden width
                - dtype: Data type
            spatial_scaling: Coordinate scaling factor

        Note:
            - Deform grid typically coarser than main INR
            - MLP has 2 hidden layers (shallower than main INR)
            - Tanh activation for bounded displacements
            - Small initialization for stability
        """
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        self.register_buffer("bounding_box", bounding_box)

        # Compute hash grid parameters for deformation
        base_resolution, n_levels = compute_resolution_nlevel(
            bounding_box,
            args.coarsest_resolution_deform,
            args.finest_resolution_deform,
            args.level_scale_deform,
            spatial_scaling,
        )
        level_scale = args.level_scale_deform

        # Build hash grid encoding (with smoothstep interpolation for smoothness)
        self.encoding = build_encoding(
            n_input_dims=3,
            otype="HashGrid",
            n_levels=n_levels,
            n_features_per_level=args.n_features_per_level_deform,
            log2_hashmap_size=args.log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=level_scale,
            dtype=args.dtype,
            interpolation="Smoothstep",
        )

        # Build MLP decoder
        # Input: hash grid features + per-slice embedding
        # Output: 3D displacement
        self.deform_net = build_network(
            n_input_dims=n_levels * args.n_features_per_level_deform
            + args.n_features_deform,
            n_output_dims=3,
            activation="Tanh",  # Bounded activation for stability
            output_activation="None",
            n_neurons=args.width,
            n_hidden_layers=2,  # Shallower than main INR
            dtype=torch.float32,
        )

        # Initialize with very small weights (important for stability!)
        for p in self.deform_net.parameters():
            torch.nn.init.uniform_(p, a=-1e-4, b=1e-4)

        # Log hyperparameters
        logging.debug(
            "hyperparameters for hash grid encoding (deform net): "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * level_scale ** (n_levels - 1)),
            level_scale,
            n_levels,
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor):
        """
        Predict deformed coordinates.

        Args:
            x: (..., 3) input coordinates in world space
            e: (..., n_features_deform) per-slice embedding

        Returns:
            (..., 3) deformed coordinates = x + displacement

        Note:
            - Residual connection ensures identity when weights are zero
            - Displacement is predicted, not absolute position
            - Embedding allows different deformation per slice
        """
        # Normalize coordinates to [0, 1]^3
        x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])

        # Store shape for reshaping
        x_shape = x.shape

        # Flatten for batch processing
        x = x.view(-1, x.shape[-1])

        # Hash grid encoding
        pe = self.encoding(x)

        # Concatenate encoding + per-slice embedding
        inputs = torch.cat((pe, e.reshape(-1, e.shape[-1])), -1)

        # Predict displacement (residual connection!)
        outputs = self.deform_net(inputs) + x

        # Denormalize back to world space
        outputs = (
            outputs * (self.bounding_box[1] - self.bounding_box[0])
            + self.bounding_box[0]
        )

        return outputs.view(x_shape)


class NeSVoR(nn.Module):
    """
    Complete NeSVoR model for MRI super-resolution reconstruction.

    This is the main model that integrates:
    - INR: 3D volume representation
    - Slice transformations: Optimizable 6-DOF rigid transforms
    - Bias field: Intensity inhomogeneity correction
    - Uncertainty: Heteroscedastic variance (per-slice and per-pixel)
    - Deformation (optional): Non-rigid motion modeling

    The model jointly optimizes all components to reconstruct a high-resolution
    3D volume from low-resolution, thick slices with:
    - Misalignment (optimizes transformations)
    - Intensity variations (learns bias field)
    - Noise and artifacts (models uncertainty)
    - Non-rigid motion (learns deformations, optional)

    Forward Pass:
    1. Sample PSF points around each pixel
    2. Apply slice transformation
    3. Apply deformation (if enabled)
    4. Query INR to get intensity
    5. Apply bias field correction
    6. Apply slice intensity scaling
    7. Average over PSF samples
    8. Compute losses (data + regularization)

    Attributes:
        inr: Implicit neural representation (volume)
        axisangle: (N_slices, 6) rigid transformations (optimizable)
        psf_sigma: (N_slices, 3) PSF standard deviations
        slice_embedding: Per-slice feature embeddings (optional)
        logit_coef: Per-slice intensity scaling logits (optional)
        log_var_slice: Per-slice log variance (optional)
        sigma_net: Per-pixel variance network (optional)
        b_net: Bias field network (optional)
        deform_net: Deformation network (optional)
        deform_embedding: Per-slice deformation embeddings (optional)

    Example:
        >>> # Create model
        >>> model = NeSVoR(
        ...     transformation, resolution, v_mean,
        ...     bounding_box, spatial_scaling, args
        ... )
        >>>
        >>> # Training forward pass
        >>> batch = dataset.get_batch(batch_size=4096)
        >>> losses = model(**batch)
        >>> # losses = {D_LOSS, S_LOSS, T_REG, I_REG, ...}
        >>>
        >>> # Weighted loss
        >>> loss = losses[D_LOSS] + losses[S_LOSS] + 0.1 * losses[T_REG] + ...
        >>> loss.backward()
    """

    def __init__(
        self,
        transformation: RigidTransform,
        resolution: torch.Tensor,
        v_mean: float,
        bounding_box: torch.Tensor,
        spatial_scaling: float,
        args: Namespace,
    ) -> None:
        """
        Initialize NeSVoR model.

        Args:
            transformation: (N_slices,) initial rigid transformations
            resolution: (N_slices, 3) resolution for each slice
            v_mean: Mean intensity (for normalization)
            bounding_box: (2, 3) spatial extent
            spatial_scaling: Coordinate scaling factor
            args: Hyperparameters (see train.py for full list)

        Note:
            - Transformations can be optimized or fixed (args.no_transformation_optimization)
            - PSF sigma computed from resolution
            - Delta for edge-preserving regularization from v_mean
            - Moves to GPU if available
        """
        super().__init__()

        # Force PyTorch implementation on CPU
        if "cpu" in str(args.device):
            global USE_TORCH
            USE_TORCH = True
        else:
            # Set default GPU for tinycudann
            # Extract device index from torch.device object
            device_idx = args.device.index if args.device.index is not None else 0
            torch.cuda.set_device(device_idx)

        self.spatial_scaling = spatial_scaling
        self.args = args
        self.n_slices = 0
        self.trans_first = True

        # Set transformations (creates self.axisangle parameter/buffer)
        self.transformation = transformation

        # Compute PSF sigma from resolution
        self.psf_sigma = resolution2sigma(resolution, isotropic=False)

        # Delta for edge-preserving regularization
        self.delta = args.delta * v_mean

        # Build network components
        self.build_network(bounding_box)

        # Move to device
        self.to(args.device)

    @property
    def transformation(self) -> RigidTransform:
        """Get current slice transformations (detached from computation graph)."""
        return RigidTransform(self.axisangle.detach(), self.trans_first)

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        """
        Set slice transformations.

        Args:
            value: RigidTransform with N_slices transformations

        Note:
            - First call: sets n_slices
            - Subsequent calls: must have same n_slices
            - Creates self.axisangle_init (frozen initial values)
            - Creates self.axisangle (parameter if optimizing, buffer if fixed)
        """
        if self.n_slices == 0:
            self.n_slices = len(value)
        else:
            assert self.n_slices == len(value)

        # Convert to axis-angle representation
        axisangle = value.axisangle(self.trans_first)

        if TYPE_CHECKING:
            self.axisangle_init: torch.Tensor

        # Store initial values (for regularization)
        self.register_buffer("axisangle_init", axisangle.detach().clone())

        # Create parameter or buffer
        if not self.args.no_transformation_optimization:
            # Optimizable transformations
            self.axisangle = nn.Parameter(axisangle.detach().clone())
        else:
            # Fixed transformations
            self.register_buffer("axisangle", axisangle.detach().clone())

    def build_network(self, bounding_box) -> None:
        """
        Build network components based on configuration.

        Creates:
        - slice_embedding: Per-slice features (if n_features_slice > 0)
        - logit_coef: Per-slice intensity scaling (if not no_slice_scale)
        - log_var_slice: Per-slice variance (if not no_slice_variance)
        - deform_embedding + deform_net: Deformation (if deformable)
        - inr: Main volume representation (always)
        - sigma_net: Per-pixel variance (if not no_pixel_variance)
        - b_net: Bias field (if n_levels_bias > 0)

        Args:
            bounding_box: (2, 3) spatial extent
        """
        # Per-slice embeddings (for slice-specific appearance)
        if self.args.n_features_slice:
            self.slice_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_slice
            )

        # Per-slice intensity scaling (handles intensity variations)
        if not self.args.no_slice_scale:
            self.logit_coef = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )

        # Per-slice variance (global noise level per slice)
        if not self.args.no_slice_variance:
            self.log_var_slice = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )

        # Deformation network (optional, for non-rigid motion)
        if self.args.deformable:
            self.deform_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_deform
            )
            self.deform_net = DeformNet(bounding_box, self.args, self.spatial_scaling)

        # Main INR (volume representation)
        self.inr = INR(bounding_box, self.args, self.spatial_scaling)

        # Per-pixel variance network (spatially-varying uncertainty)
        if not self.args.no_pixel_variance:
            self.sigma_net = build_network(
                n_input_dims=self.args.n_features_slice + self.args.n_features_z,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )

        # Bias field network (intensity inhomogeneity correction)
        if self.args.n_levels_bias:
            self.b_net = build_network(
                n_input_dims=self.args.n_levels_bias * self.args.n_features_per_level
                + self.args.n_features_slice,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )

    def forward(
        self,
        xyz: torch.Tensor,
        v: torch.Tensor,
        slice_idx: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass: predict intensities and compute losses.

        Args:
            xyz: (B, 3) coordinates in slice-local space
            v: (B,) observed intensities
            slice_idx: (B,) slice indices

        Returns:
            Dictionary of losses:
                - D_LOSS: Data fidelity (weighted MSE)
                - S_LOSS: Variance log-likelihood (if variance enabled)
                - DS_LOSS: Combined data + variance (if variance enabled)
                - T_REG: Transformation regularization (if optimizing transforms)
                - B_REG: Bias field regularization (if bias enabled)
                - I_REG: Image regularization (always)
                - D_REG: Deformation regularization (if deformable)

        Algorithm:
            1. Sample PSF points (Monte Carlo)
            2. Apply slice transformations
            3. Apply deformations (if enabled)
            4. Query INR → density
            5. Apply bias field → biased_density
            6. Average over PSF samples → predicted_intensity
            7. Apply slice scaling → final_prediction
            8. Compute weighted MSE with learned variance
            9. Compute all regularization losses
        """
        # ========== PSF Sampling ==========
        batch_size = xyz.shape[0]
        n_samples = self.args.n_samples

        # Generate Gaussian random offsets for PSF
        xyz_psf = torch.randn(
            batch_size, n_samples, 3, dtype=xyz.dtype, device=xyz.device
        )

        # Get PSF sigma for each point's slice
        psf_sigma = self.psf_sigma[slice_idx][:, None]  # (B, 1, 3)

        # ========== Transformation ==========
        # Apply rigid transformation to PSF-sampled points
        t = self.axisangle[slice_idx][:, None]  # (B, 1, 6)
        xyz = ax_transform_points(
            t, xyz[:, None] + xyz_psf * psf_sigma, self.trans_first
        )  # (B, n_samples, 3)

        # ========== Deformation (Optional) ==========
        xyz_ori = xyz  # Keep original for regularization
        if self.args.deformable:
            # Get per-slice deformation embedding
            de = self.deform_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
            # Apply deformation
            xyz = self.deform_net(xyz, de)

        # ========== Slice Embedding (Optional) ==========
        if self.args.n_features_slice:
            # Get per-slice appearance embedding
            se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
        else:
            se = None

        # ========== INR Forward ==========
        # Query INR and get density, bias, variance
        results = self.net_forward(xyz, se)

        # ========== Post-Processing ==========
        density = results["density"]  # (B, n_samples)

        # Bias field
        if "log_bias" in results:
            log_bias = results["log_bias"]
            bias = log_bias.exp()
            bias_detach = bias.detach()  # Don't backprop through bias for variance
        else:
            log_bias = 0
            bias = 1
            bias_detach = 1

        # Per-pixel variance
        if "log_var" in results:
            log_var = results["log_var"]
            var = log_var.exp()
        else:
            log_var = 0
            var = 1

        # ========== Imaging Forward Model ==========
        # Slice intensity scaling (handles inter-slice intensity variations)
        if not self.args.no_slice_scale:
            # Softmax ensures scales are positive and sum to n_slices
            c: Any = F.softmax(self.logit_coef, 0)[slice_idx] * self.n_slices
        else:
            c = 1

        # Apply bias and average over PSF samples
        v_out = (bias * density).mean(-1)  # (B,)

        # Apply slice scaling
        v_out = c * v_out

        # Update variance (average over PSF, scale by slice coef)
        if not self.args.no_pixel_variance:
            var = (bias_detach * var).mean(-1)
            var = c.detach() * var
            var = var**2

        # Add per-slice variance
        if not self.args.no_slice_variance:
            var = var + self.log_var_slice.exp()[slice_idx]

        # ========== Loss Computation ==========
        losses = {}

        # Data fidelity: weighted MSE
        losses[D_LOSS] = ((v_out - v) ** 2 / (2 * var)).mean()

        # Variance log-likelihood
        if not (self.args.no_pixel_variance and self.args.no_slice_variance):
            losses[S_LOSS] = 0.5 * var.log().mean()
            losses[DS_LOSS] = losses[D_LOSS] + losses[S_LOSS]

        # Transformation regularization
        if not self.args.no_transformation_optimization:
            losses[T_REG] = self.trans_loss(trans_first=self.trans_first)

        # Bias field regularization
        if self.args.n_levels_bias:
            losses[B_REG] = log_bias.mean() ** 2

        # Deformation regularization
        if self.args.deformable:
            losses[D_REG] = self.deform_reg(xyz, xyz_ori, de)

        # Image regularization (TV/edge/L2)
        losses[I_REG] = self.img_reg(density, xyz)

        return losses

    def net_forward(
        self,
        x: torch.Tensor,
        se: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Query INR and auxiliary networks.

        Args:
            x: (B, n_samples, 3) coordinates
            se: (B, n_samples, n_features_slice) slice embeddings (optional)

        Returns:
            Dictionary with:
                - density: (B, n_samples) INR output
                - log_bias: (B, n_samples) log bias field (if enabled)
                - log_var: (B, n_samples) log variance (if enabled)
        """
        # Query main INR
        density, pe, z = self.inr(x)  # (B, n_samples)
        prefix_shape = density.shape
        results = {"density": density}

        # Collect inputs for auxiliary networks
        zs = []
        if se is not None:
            zs.append(se.reshape(-1, se.shape[-1]))

        # Bias field network
        if self.args.n_levels_bias:
            # Use coarse levels of hash grid for bias
            pe_bias = pe[
                ..., : self.args.n_levels_bias * self.args.n_features_per_level
            ]
            results["log_bias"] = self.b_net(torch.cat(zs + [pe_bias], -1)).view(
                prefix_shape
            )

        # Per-pixel variance network
        if not self.args.no_pixel_variance:
            zs.append(z[..., 1:])  # Use extra features from density_net
            results["log_var"] = self.sigma_net(torch.cat(zs, -1)).view(prefix_shape)

        return results

    def trans_loss(self, trans_first: bool = True) -> torch.Tensor:
        """
        Transformation regularization: keep close to initialization.

        Penalizes deviation from initial transformations to prevent
        overfitting to noise or drifting to unrealistic alignments.

        Args:
            trans_first: Transformation convention

        Returns:
            Scalar regularization loss

        Note:
            - Computes relative transformation: T_init^{-1} ∘ T_current
            - Separate penalties for rotation (equal weight) and translation (1e-3x)
            - Translation scaled by spatial_scaling^2
        """
        # Current and initial transformations
        x = RigidTransform(self.axisangle, trans_first=trans_first)
        y = RigidTransform(self.axisangle_init, trans_first=trans_first)

        # Relative transformation
        err = y.inv().compose(x).axisangle(trans_first=trans_first)

        # Rotation penalty (equal weight for all 3 rotation components)
        loss_R = torch.mean(err[:, :3] ** 2)

        # Translation penalty (scaled down by 1e-3 and spatial_scaling^2)
        loss_T = torch.mean(err[:, 3:] ** 2)

        return loss_R + 1e-3 * self.spatial_scaling * self.spatial_scaling * loss_T

    def img_reg(self, density, xyz):
        """
        Image regularization: encourage smoothness.

        Supports three regularization types:
        - "TV": Total Variation (L1 gradient norm)
        - "edge": Edge-preserving (Charbonnier loss)
        - "L2": L2 gradient norm (Tikhonov)

        Args:
            density: (B, n_samples) predicted intensities
            xyz: (B, n_samples, 3) coordinates

        Returns:
            Scalar regularization loss

        Note:
            - Can use autodiff (slower, exact) or finite differences (faster, approx)
            - Edge-preserving uses delta parameter from initialization
            - Gradient computed in scaled coordinate space
        """
        if self.args.image_regularization == "none":
            return torch.zeros((1,), dtype=density.dtype, device=density.device)

        # Compute gradient
        if self.args.img_reg_autodiff:
            # Autodiff: exact gradient
            n_sample = 4
            xyz = xyz[:, :n_sample].flatten(0, 1).detach()
            xyz.requires_grad_()
            density, _, _ = self.inr(xyz)
            grad = (
                torch.autograd.grad((density.sum(),), (xyz,), create_graph=True)[0]
                / self.spatial_scaling
            )
            grad2 = grad.pow(2)
        else:
            # Finite differences: approximate gradient using PSF samples
            xyz = xyz * self.spatial_scaling
            d_density = density - torch.flip(density, (1,))  # Intensity difference
            dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6  # Distance^2
            grad2 = d_density**2 / dx2  # Gradient magnitude^2

        # Apply regularization function
        if self.args.image_regularization == "TV":
            # Total Variation: L1 norm of gradient
            return grad2.sqrt().mean()
        elif self.args.image_regularization == "edge":
            # Edge-preserving (Charbonnier): sqrt(1 + (grad/delta)^2) - 1
            return self.delta * (
                (1 + grad2 / (self.delta * self.delta)).sqrt().mean() - 1
            )
        elif self.args.image_regularization == "L2":
            # Tikhonov: L2 norm of gradient
            return grad2.mean()
        else:
            raise ValueError("unknown image regularization!")

    def deform_reg(self, out, xyz, e):
        """
        Deformation regularization: encourage smoothness and preserve volume.

        Penalizes deviation of Jacobian from identity matrix.
        This ensures:
        - Smooth deformation (gradual changes)
        - Volume preservation (det(J) ≈ 1)
        - No folding (J stays close to I)

        Args:
            out: Deformed coordinates (not used in autodiff version)
            xyz: (B, n_samples, 3) input coordinates
            e: (B, n_samples, n_features_deform) deformation embeddings

        Returns:
            Scalar regularization loss

        Note:
            - Computes Jacobian J = ∂(deform_net)/∂x via autodiff
            - Penalty: ||J^T J - I||_F^2 (Frobenius norm squared)
            - This is equivalent to penalizing ||J - I||_F^2 when J ≈ I
            - Uses only subset of samples (n_sample=4) for efficiency
        """
        if True:  # use autodiff
            # Sample subset for efficiency
            n_sample = 4
            x = xyz[:, :n_sample].flatten(0, 1).detach()
            e = e[:, :n_sample].flatten(0, 1).detach()

            # Enable gradient tracking for Jacobian computation
            x.requires_grad_()

            # Forward pass
            outputs = self.deform_net(x, e)

            # Compute Jacobian matrix via autodiff
            grads = []
            out_sum = []
            for i in range(3):
                out_sum.append(outputs[:, i].sum())
                grads.append(
                    torch.autograd.grad((out_sum[-1],), (x,), create_graph=True)[0]
                )
            jacobian = torch.stack(grads, -1)  # (N, 3, 3)

            # Compute J^T J
            jtj = torch.matmul(jacobian, jacobian.transpose(-1, -2))

            # Identity matrix
            I = torch.eye(3, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)

            # Penalty: ||J^T J - I||_F^2
            sq_residual = ((jtj - I) ** 2).sum((-2, -1))

            # Handle NaNs (can occur with extreme deformations)
            return torch.nan_to_num(sq_residual, 0.0, 0.0, 0.0).mean()
        else:
            # Alternative: finite difference approximation (not used)
            out = out - xyz
            d_out2 = ((out - torch.flip(out, (1,))) ** 2).sum(-1) + 1e-6
            dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
            dd_dx = d_out2.sqrt() / dx2.sqrt()
            return F.smooth_l1_loss(dd_dx, torch.zeros_like(dd_dx).detach(), beta=1e-3)
