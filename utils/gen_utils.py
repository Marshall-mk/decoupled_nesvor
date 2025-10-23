"""
General Utility Functions for NeSVoR2

This module provides general-purpose utility functions including:
- Logging utilities for training monitoring
- Loss functions (NCC, SSIM) for image similarity
- Image processing utilities (resampling, blurring, PSF generation)
- Helper functions for reproducibility and file operations

Key components:
- Logging: TrainLogger, LazyLog, setup_logger for experiment tracking
- Loss functions: ncc_loss, ssim_loss for comparing images
- Image processing: resample, gaussian_blur, meshgrid
- Training utilities: MovingAverage for tracking metrics
"""

from os import PathLike
import torch
from math import log, sqrt
import torch.nn.functional as F
import collections
from argparse import Namespace
import os
import random
import numpy as np
import logging
import warnings
import types
import traceback
import sys
import io
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Collection,
    Iterable,
    Sequence,
    Tuple,
    Callable,
    TypeVar,
    cast,
)


# ============================================================================
# CONSTANTS
# ============================================================================

# Full Width at Half Maximum (FWHM) for Gaussian distribution
# Used to convert between FWHM and standard deviation (sigma)
# Relationship: FWHM = 2 * sqrt(2 * ln(2)) * sigma
GAUSSIAN_FWHM = 1 / (2 * sqrt(2 * log(2)))

# FWHM for sinc function (approximation)
# Used for modeling slice selection profile in MRI
SINC_FWHM = 1.206709128803223 * GAUSSIAN_FWHM

# Type aliases for better code readability
PathType = Union[str, PathLike[str]]  # File path can be string or PathLike object
DeviceType = Union[
    torch.device, str, None
]  # Device can be torch.device, string, or None


# ============================================================================
# LOGGING UTILITIES
# ============================================================================
# These classes and functions provide structured logging for experiments


class LazyLog(object):
    """
    Lazy evaluation wrapper for log messages.

    This class delays the evaluation of logging messages until they're actually
    needed (i.e., when the log level requires them to be displayed). This is
    useful for expensive string formatting operations that shouldn't be performed
    if the message won't be logged.

    Example:
        Instead of:
            logging.info("Computed values: %s" % expensive_computation())

        Use:
            logging.info(LazyLog(expensive_computation))

        The expensive_computation() only runs if INFO level is enabled.
    """

    def __init__(self, func: Callable[..., Any], *args, **kwargs) -> None:
        """
        Initialize lazy log wrapper.

        Args:
            func: Function to call when string representation is needed
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        """
        Evaluate the function when string representation is requested.

        Returns:
            String result from calling the wrapped function
        """
        return self.func(*self.args, **self.kwargs)


class TrainLogger(object):
    """
    Formatted logger for training metrics.

    Provides a structured way to log training progress with aligned columns.
    The first call creates a header, subsequent calls log metrics with
    consistent formatting.

    Example:
        logger = TrainLogger("Epoch", "Loss", "Accuracy")
        logger.log(1, 0.5, 0.95)
        logger.log(2, 0.3, 0.97)

        Output:
        Epoch        Loss         Accuracy
        1            5.000e-01    9.500e-01
        2            3.000e-01    9.700e-01
    """

    def __init__(self, *args: str) -> None:
        """
        Initialize training logger with column headers.

        Args:
            *args: Column header names (e.g., "Epoch", "Loss", "LR")
        """
        self.headers = args
        # Each column is 12 characters wide with space separator
        self.template = "%12s " * len(self.headers)
        # Log the header row
        logging.info(LazyLog(self._log, self.template, args))

    def log(self, *args):
        """
        Log a row of training metrics.

        Args:
            *args: Values to log (must match number of headers)

        Raises:
            AssertionError: If number of values doesn't match number of headers
        """
        assert len(args) == len(self.headers), (
            "The length of inputs differ from the length of header!"
        )
        logging.info(LazyLog(self._log, self.template, args))

    def _log(self, template, args):
        """
        Format values for logging.

        Converts floats to scientific notation (3 decimal places).

        Args:
            template: Format string template
            args: Values to format

        Returns:
            Formatted string
        """
        args = list(args)
        for i in range(len(args)):
            # Format floats in scientific notation
            if isinstance(args[i], float):
                args[i] = "%.3e" % args[i]
        return template % tuple(args)


def _log_params(model: torch.nn.Module) -> str:
    """
    Create a formatted table of model parameters.

    Lists all trainable parameters with their names, shapes, and counts.

    Args:
        model: PyTorch model to inspect

    Returns:
        Formatted string table of parameters

    Example output:
        trainable parameters in MyModel
        ----------------------------------------
        Name                    Shape                # Param
        encoder.weight          [256, 128]           32768
        encoder.bias            [256]                256
        ----------------------------------------
    """
    # Calculate column widths based on longest parameter name
    name_len = max(len(name) for name, _ in model.named_parameters()) + 1
    shape_len = 20
    n_param_len = 20
    sep_len = name_len + shape_len + n_param_len + 3
    sep = "-" * sep_len

    # Create column format template
    template = f"%{name_len}s %{shape_len}s %{n_param_len}s\n"

    # Build parameter list: headers first, then all parameters
    args: List = ["Name", "Shape", "# Param"]
    for name, param in model.named_parameters():
        args.extend([name, list(param.shape), param.numel()])

    # Format complete table
    template = "trainable parameters in %s\n%s\n" + template * (len(args) // 3) + "%s"
    return template % (model.__class__.__name__, sep, *args, sep)


def log_params(model: torch.nn.Module) -> LazyLog:
    """
    Create lazy log wrapper for model parameters.

    Args:
        model: PyTorch model to log

    Returns:
        LazyLog object that will format parameters when needed

    Usage:
        logging.info(log_params(model))
    """
    return LazyLog(_log_params, model)


def log_args(args: Namespace) -> None:
    """
    Log all command-line arguments.

    Useful for recording experiment configuration.

    Args:
        args: Argparse Namespace containing parsed arguments

    Example output:
        input arguments
        ----------------------------------------
        learning_rate: 0.001
        batch_size: 32
        epochs: 100
        ----------------------------------------
    """
    d = vars(args)  # Convert Namespace to dictionary
    logging.debug(
        "input arguments\n"
        + "----------------------------------------\n"
        + "%s: %s\n" * len(d)
        + "----------------------------------------",
        *sum(d.items(), ()),  # Flatten dict items into flat list
    )


# Global flag to prevent multiple logger initializations
_initialized = False


def setup_logger(
    filename: Optional[str] = None,
    verbose: int = 1,
    level: Optional[int] = None,
) -> None:
    """
    Configure logging system for the application.

    Sets up both file and console logging with appropriate formatting and
    log levels. Also installs custom exception hook to log uncaught exceptions.

    Args:
        filename: Optional log file path. If None, only logs to console
        verbose: Verbosity level (ignored when `level` provided):
            0 = WARNING (only warnings and errors)
            1 = INFO (include informational messages)
            2 = DEBUG (include debug messages)
            3+ = NOTSET (log everything)
        level: Explicit logging level (e.g. logging.INFO). Takes precedence over
            `verbose` if provided.

    Note:
        Can only be called once. Subsequent calls are ignored.
        Adds custom "RESULT" log level above WARNING for final results.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Map verbosity to log level unless explicit level is supplied
    if level is None:
        if verbose == 0:
            level = logging.WARNING
        elif verbose == 1:
            level = logging.INFO
        elif verbose == 2:
            level = logging.DEBUG
        else:
            level = logging.NOTSET

    handlers: List[Any] = []
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add file handler if filename provided
    if filename:
        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setFormatter(log_formatter)
        handlers.append(file_handler)

    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        handlers=handlers,
        level=level,
    )

    # Install custom exception hook to log uncaught exceptions
    def log_except_hook(*exc_info):
        """Log uncaught exceptions instead of printing to stderr."""
        text = "".join(traceback.format_exception(*exc_info))
        logging.error("Unhandled exception:\n%s", text)

    sys.excepthook = log_except_hook

    # Add custom "RESULT" log level for final experiment results
    # This is between WARNING and ERROR in priority
    levelNum = logging.WARNING + 5
    levelName = "RESULT"
    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)


def log_result(message, *args, **kwargs):
    """
    Log a message at RESULT level.

    RESULT level is used for final experiment results that should always
    be visible, regardless of verbosity settings.

    Args:
        message: Log message (can contain format specifiers)
        *args: Arguments for message formatting
        **kwargs: Additional arguments passed to logging function
    """
    return logging.log(getattr(logging, "RESULT"), message, *args, **kwargs)


class LogIO(io.StringIO):
    """
    StringIO wrapper that forwards writes to a logging function.

    Useful for redirecting stdout/stderr to the logging system.

    Example:
        log_stream = LogIO(logging.info)
        print("This goes to logger", file=log_stream)
    """

    def __init__(self, fn, *args, **kwargs) -> None:
        """
        Initialize logging stream.

        Args:
            fn: Logging function to call (e.g., logging.info)
            *args: Additional args for StringIO
            **kwargs: Additional kwargs for StringIO
        """
        super().__init__(*args, **kwargs)
        self.fn = fn

    def write(self, __s: str) -> int:
        """
        Write string to both buffer and logger.

        Args:
            __s: String to write

        Returns:
            Number of characters written
        """
        __s = __s.strip()
        if __s:  # Only log non-empty strings
            self.fn(__s)
        return super().write(__s)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
# Image similarity metrics used as loss functions for registration/reconstruction


def ncc_loss(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    win: Optional[int] = 9,
    level: int = 0,
    eps: float = 1e-6,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Normalized Cross-Correlation (NCC) loss for image similarity.

    NCC measures linear relationship between two images. It's invariant to
    linear intensity transformations (brightness/contrast changes), making it
    robust for medical image registration.

    Formula: NCC = (E[IJ] - E[I]E[J]) / (std[I] * std[J])

    Args:
        I: First image, shape (N, C, *spatial_dims)
        J: Second image, same shape as I
        mask: Optional binary mask, same shape as I
        win: Window size for local NCC. If None, computes global NCC
        level: Pyramid level (for multi-scale registration). Adjusts window size
        eps: Small constant for numerical stability
        reduction: 'none', 'mean', or 'sum' - how to reduce the output

    Returns:
        NCC values (negated, so lower is better):
            - If reduction='none': shape matches input
            - If reduction='mean': scalar
            - If reduction='sum': scalar

    Note:
        - Supports 1D, 2D, and 3D images (3D/4D/5D tensors)
        - Local NCC (with win) is more sensitive to local misalignment
        - Global NCC (win=None) measures overall image similarity
    """
    spatial_dims = len(I.shape) - 2
    assert spatial_dims in (1, 2, 3), "ncc_loss only support 3D, 4D, and 5D data"

    # Apply mask if provided
    if mask is not None:
        I = I * mask
        J = J * mask

    c = I.shape[1]  # Number of channels

    # Global NCC: compute statistics over entire image
    if win is None:
        I = torch.flatten(I, 1)  # Flatten spatial dimensions
        J = torch.flatten(J, 1)

        if mask is not None:
            # Masked statistics
            mask = torch.flatten(mask, 1)
            N = mask.sum(-1) + eps  # Number of valid pixels
            I_mean = I.sum(-1) / N
            J_mean = J.sum(-1) / N
            I2_mean = (I * I).sum(-1) / N
            J2_mean = (J * J).sum(-1) / N
            IJ_mean = (I * J).sum(-1) / N
        else:
            # Unmasked statistics
            I_mean = I.mean(-1)
            J_mean = J.mean(-1)
            I2_mean = (I * I).mean(-1)
            J2_mean = (J * J).mean(-1)
            IJ_mean = (I * J).mean(-1)

    # Local NCC: compute statistics in sliding windows
    else:
        I = I.view(-1, 1, *I.shape[2:])  # Reshape for convolution
        J = J.view(-1, 1, *J.shape[2:])

        # Adjust window size based on pyramid level
        win = 2 * int(win / 2**level / 2) + 1  # Ensure odd window size

        # Create averaging filter (box filter)
        mean_filt = torch.ones([1, 1] + [win] * spatial_dims, device=I.device) / (
            win**spatial_dims
        )

        # Select appropriate convolution function
        conv_fn = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]

        # Compute local means and second moments via convolution
        I_mean = conv_fn(I, mean_filt, stride=1, padding=win // 2)
        J_mean = conv_fn(J, mean_filt, stride=1, padding=win // 2)
        I2_mean = conv_fn(I * I, mean_filt, stride=1, padding=win // 2)
        J2_mean = conv_fn(J * J, mean_filt, stride=1, padding=win // 2)
        IJ_mean = conv_fn(I * J, mean_filt, stride=1, padding=win // 2)

    # Compute NCC components
    cross = IJ_mean - I_mean * J_mean  # Cross-covariance
    I_var = I2_mean - I_mean * I_mean  # Variance of I
    J_var = J2_mean - J_mean * J_mean  # Variance of J

    # NCC = cross^2 / (var_I * var_J)
    # Squared to make it always positive
    cc = cross * cross / (I_var * J_var + eps)

    # Apply reduction
    if reduction == "mean":
        return -cc.mean()  # Negative because we minimize loss
    elif reduction == "sum":
        return -cc.sum()
    else:  # 'none'
        if win is None:
            return -cc.view(-1, c)
        else:
            return -cc.view(-1, c, *I.shape[2:])


def ssim_loss(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    win: int = 11,
    sigma: float = 1.5,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) loss.

    SSIM measures perceived image quality by comparing luminance, contrast,
    and structure. It's based on human visual perception and correlates well
    with perceptual image quality.

    Formula: SSIM = (2*μ_I*μ_J + C1) * (2*σ_IJ + C2) /
                    ((μ_I^2 + μ_J^2 + C1) * (σ_I^2 + σ_J^2 + C2))

    Args:
        I: First image, shape (N, C, *spatial_dims)
        J: Second image, same shape as I
        mask: Optional binary mask for valid regions
        win: Window size for Gaussian weighting (typically 11)
        sigma: Standard deviation of Gaussian window (typically 1.5)
        reduction: 'none', 'mean', or 'sum'

    Returns:
        SSIM values (negated, so lower is better)

    Note:
        - Images are normalized to [0, 1] range before computation
        - Uses Gaussian weighting for local statistics
        - C1 and C2 are stabilization constants
    """
    # Normalize images to [0, 1] range
    I_min = I.min()
    I_max = I.max()
    J_min = J.min()
    J_max = J.max()
    I = (I - I_min) / (I_max - I_min)
    J = (J - J_min) / (J_max - J_min)

    # SSIM parameters
    spatial_dims = len(I.shape) - 2
    C1 = 0.01**2  # Stabilization constant for luminance
    C2 = 0.03**2  # Stabilization constant for contrast
    truncated = win / 2 / sigma - 0.5  # Truncation point for Gaussian
    compensation = 1.0  # Compensation factor

    # Compute local means with Gaussian weighting
    mu1 = gaussian_blur(I, sigma, truncated)
    mu2 = gaussian_blur(J, sigma, truncated)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = compensation * (gaussian_blur(I * I, sigma, truncated) - mu1_sq)
    sigma2_sq = compensation * (gaussian_blur(J * J, sigma, truncated) - mu2_sq)
    sigma12 = compensation * (gaussian_blur(I * J, sigma, truncated) - mu1_mu2)

    # Compute SSIM components
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # Contrast-structure
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map  # Full SSIM

    # Apply mask if provided
    if mask is not None:
        ssim_map = ssim_map * mask

    # Apply reduction
    if reduction == "mean":
        return -ssim_map.mean()
    elif reduction == "sum":
        return -ssim_map.sum()
    else:
        return -ssim_map


# ============================================================================
# GENERAL UTILITIES
# ============================================================================


def set_seed(seed: Optional[int]) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for PyTorch, NumPy, and Python's random module to ensure
    reproducible results across runs.

    Args:
        seed: Random seed value. If None, does nothing.

    Note:
        For full reproducibility, you may also need to:
        - Set CUDA deterministic mode: torch.backends.cudnn.deterministic = True
        - Disable CUDA benchmarking: torch.backends.cudnn.benchmark = False
        - Use single-threaded operations
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def makedirs(path: Union[str, Iterable[str]]) -> None:
    """
    Create directory or directories, ignoring if they already exist.

    Safer alternative to os.makedirs that doesn't fail on existing directories.

    Args:
        path: Single path string or iterable of paths

    Note:
        Silently ignores FileExistsError but re-raises other exceptions
    """
    if isinstance(path, str):
        path = [path]
    for p in path:
        if p:  # Skip empty strings
            try:
                os.makedirs(p, exist_ok=False)
            except FileExistsError:
                pass  # Directory already exists, this is fine
            except Exception as e:
                raise e  # Re-raise other exceptions


def merge_args(args_old: Namespace, args_new: Namespace) -> Namespace:
    """
    Merge two argument namespaces, with new arguments overriding old ones.

    Useful for combining default arguments with user-specified overrides.

    Args:
        args_old: Original/default arguments
        args_new: New arguments to override with

    Returns:
        New Namespace with merged arguments

    Example:
        defaults = Namespace(lr=0.001, batch_size=32)
        overrides = Namespace(lr=0.01)
        merged = merge_args(defaults, overrides)
        # merged.lr == 0.01, merged.batch_size == 32
    """
    dict_old = vars(args_old)
    dict_new = vars(args_new)
    dict_old.update(dict_new)  # In-place update (args_old is modified)
    return Namespace(**dict_old)


def resample(
    x: torch.Tensor, res_xyz_old: Sequence, res_xyz_new: Sequence
) -> torch.Tensor:
    """
    Resample image to a new resolution using interpolation.

    Changes the physical sampling rate of an image while preserving content.
    Uses trilinear/bilinear interpolation for smooth resampling.

    Args:
        x: Input image, shape (N, C, *spatial_dims)
        res_xyz_old: Current resolution for each dimension [rx, ry, rz]
        res_xyz_new: Target resolution for each dimension

    Returns:
        Resampled image with new shape based on resolution ratio

    Example:
        # Downsample from 0.5mm to 1mm resolution
        x_orig = torch.randn(1, 1, 100, 100, 100)  # 0.5mm resolution
        x_down = resample(x_orig, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        # x_down.shape = (1, 1, 50, 50, 50)

    Note:
        - If resolutions are identical, returns input unchanged
        - New size = old_size * (old_res / new_res)
        - Uses PyTorch grid_sample with align_corners=True
    """
    ndim = x.ndim - 2  # Number of spatial dimensions
    assert len(res_xyz_new) == len(res_xyz_old) == ndim

    # Short-circuit if no resampling needed
    if all(r_new == r_old for (r_new, r_old) in zip(res_xyz_new, res_xyz_old)):
        return x

    # Create sampling grids for each dimension
    grids = []
    for i in range(ndim):
        fac = res_xyz_old[i] / res_xyz_new[i]  # Resolution ratio
        size_new = int(x.shape[-i - 1] * fac)  # New size in this dimension

        # Compute normalized grid coordinates [-grid_max, grid_max]
        # grid_max ensures proper alignment at boundaries
        grid_max = (size_new - 1) / fac / (x.shape[-i - 1] - 1)

        grids.append(
            torch.linspace(
                -grid_max, grid_max, size_new, dtype=x.dtype, device=x.device
            )
        )

    # Create meshgrid of sampling coordinates
    grid = torch.stack(torch.meshgrid(*grids[::-1], indexing="ij")[::-1], -1)

    # Sample using grid_sample
    y = F.grid_sample(
        x, grid[None].expand((x.shape[0],) + (-1,) * (ndim + 1)), align_corners=True
    )
    return y


def meshgrid(
    shape_xyz: Collection,
    resolution_xyz: Collection,
    min_xyz: Optional[Collection] = None,
    device: DeviceType = None,
    stack_output: bool = True,
):
    """
    Create a meshgrid of 3D coordinates in physical space.

    Generates coordinates for every voxel in a volume, useful for
    sampling or coordinate-based networks.

    Args:
        shape_xyz: Size in each dimension [nx, ny, nz]
        resolution_xyz: Voxel spacing in each dimension [rx, ry, rz] (in mm)
        min_xyz: Minimum coordinate for each dimension. If None, centers at origin
        device: Device to create tensors on
        stack_output: If True, returns stacked tensor; otherwise tuple of grids

    Returns:
        If stack_output=True: Tensor of shape (*shape_xyz, 3) with coordinates
        If stack_output=False: Tuple of 3 tensors, each with shape (*shape_xyz)

    Example:
        # Create coordinates for 10x10x10 volume with 1mm spacing
        coords = meshgrid([10, 10, 10], [1.0, 1.0, 1.0])
        # coords.shape = (10, 10, 10, 3)
        # coords[0, 0, 0] = [-4.5, -4.5, -4.5]  (corner)
        # coords[4, 4, 4] = [0, 0, 0]  (center)
    """
    assert len(shape_xyz) == len(resolution_xyz)

    # Default: center coordinate system at origin
    if min_xyz is None:
        min_xyz = tuple(-(s - 1) * r / 2 for s, r in zip(shape_xyz, resolution_xyz))
    else:
        assert len(shape_xyz) == len(min_xyz)

    # Infer device from input tensors if not specified
    if device is None:
        if isinstance(shape_xyz, torch.Tensor):
            device = shape_xyz.device
        elif isinstance(resolution_xyz, torch.Tensor):
            device = resolution_xyz.device
        else:
            device = torch.device("cpu")
    dtype = torch.float32

    # Create 1D coordinate arrays for each dimension
    arr_xyz = [
        torch.arange(s, dtype=dtype, device=device) * r + m
        for s, r, m in zip(shape_xyz, resolution_xyz, min_xyz)
    ]

    # Create meshgrid (reverse for indexing='ij' convention)
    grid_xyz = torch.meshgrid(arr_xyz[::-1], indexing="ij")[::-1]

    if stack_output:
        return torch.stack(grid_xyz, -1)
    else:
        return grid_xyz


def gaussian_blur(
    x: torch.Tensor, sigma: Union[float, collections.abc.Iterable], truncated: float
) -> torch.Tensor:
    """
    Apply Gaussian blur to image via separable convolution.

    Efficiently blurs images by applying 1D Gaussian filters sequentially
    along each dimension (separable filtering). This is much faster than
    using a full nD Gaussian kernel.

    Args:
        x: Input image, shape (N, C, *spatial_dims)
        sigma: Standard deviation of Gaussian. Can be:
            - Single float: same sigma for all dimensions
            - List/tuple: different sigma per dimension
        truncated: Truncation factor (kernel extends to truncated * sigma)

    Returns:
        Blurred image, same shape as input

    Example:
        # Blur with sigma=2.0 in all directions
        x_blur = gaussian_blur(x, 2.0, truncated=3.0)

        # Different blur per axis
        x_blur = gaussian_blur(x, [1.0, 1.0, 3.0], truncated=3.0)

    Note:
        - Separable filtering: O(N*D*K) instead of O(N*K^D)
        - Uses grouped convolution to process all channels independently
        - Kernel size = 2 * int(sigma * truncated + 0.5) + 1
    """
    spatial_dims = len(x.shape) - 2

    # Convert scalar sigma to list
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = [sigma] * spatial_dims

    # Create 1D Gaussian kernels for each dimension
    kernels = [gaussian_1d_kernel(s, truncated, x.device) for s in sigma]

    c = x.shape[1]  # Number of channels
    conv_fn = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]

    # Apply 1D convolutions sequentially along each axis
    for d in range(spatial_dims):
        # Reshape kernel for dimension d
        s = [1] * len(x.shape)
        s[d + 2] = -1  # Extend along dimension d
        k = kernels[d].reshape(s).repeat(*([c, 1] + [1] * spatial_dims))

        # Set padding for this dimension
        padding = [0] * spatial_dims
        padding[d] = (k.shape[d + 2] - 1) // 2

        # Apply convolution (grouped by channel)
        x = conv_fn(x, k, padding=padding, groups=c)

    return x


def gaussian_1d_kernel(
    sigma: float, truncated: float, device: DeviceType
) -> torch.Tensor:
    """
    Generate 1D Gaussian kernel using error function.

    Creates a discrete approximation of Gaussian by integrating over each
    pixel width. This is more accurate than sampling the Gaussian at pixel
    centers, especially for small sigmas.

    Formula: kernel[i] = 0.5 * (erf((x+0.5)/σ) - erf((x-0.5)/σ))

    Args:
        sigma: Standard deviation of Gaussian
        truncated: Truncation factor (kernel extends to truncated * sigma)
        device: Device to create kernel on

    Returns:
        1D Gaussian kernel, normalized and clamped to [0, inf)

    Note:
        Implementation from MONAI library
        Kernel size = 2 * tail + 1, where tail = int(sigma * truncated + 0.5)
    """
    # Compute kernel half-width
    tail = int(max(sigma * truncated, 0.5) + 0.5)

    # Create coordinate array
    x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)

    # Normalization factor for erf
    t = 0.70710678 / sigma  # sqrt(2) / sigma

    # Integrate Gaussian over pixel width using error function
    kernel = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())

    # Clamp small negative values to zero
    return kernel.clamp(min=0)


class MovingAverage:
    """
    Compute exponential or simple moving average of metrics.

    Tracks multiple named metrics and computes their moving averages.
    Supports both exponential moving average (EMA) and simple average.

    Example:
        # Exponential moving average with alpha=0.9
        ma = MovingAverage(alpha=0.9)
        for i in range(100):
            loss = compute_loss()
            ma('loss', loss)
            print(f"Average loss: {ma['loss']:.4f}")

        # Simple average (alpha=0)
        ma = MovingAverage(alpha=0)
        # Now ma['loss'] is the arithmetic mean of all values

    Note:
        - alpha=0: Simple average (sum / count)
        - alpha>0: EMA with bias correction
        - Bias correction: divide by (1 - alpha^n) for first n values
    """

    def __init__(self, alpha: float) -> None:
        """
        Initialize moving average tracker.

        Args:
            alpha: Decay factor for EMA (0 <= alpha < 1)
                - 0: Simple average
                - Close to 1: More weight on recent values

        Raises:
            AssertionError: If alpha not in [0, 1)
        """
        assert 0 <= alpha < 1
        self.alpha = alpha
        self._value: Dict[str, Any] = dict()  # Stores (count, value) tuples

    def to_dict(self) -> Dict[str, Any]:
        """Export state for checkpointing."""
        return {"alpha": self.alpha, "value": self._value}

    def from_dict(self, d: Dict) -> None:
        """Restore state from checkpoint."""
        self.alpha = d["alpha"]
        self._value = d["value"]

    def __getitem__(self, key: str) -> Any:
        """
        Get current average for a metric.

        Args:
            key: Metric name

        Returns:
            Current average value (with bias correction if EMA)
        """
        if key not in self._value:
            return 0
        num, v = self._value[key]
        if self.alpha:
            # EMA with bias correction
            return v / (1 - self.alpha**num)
        else:
            # Simple average
            return v / num

    def __call__(self, key: str, value) -> None:
        """
        Update moving average with new value.

        Args:
            key: Metric name
            value: New value to incorporate
        """
        if key not in self._value:
            self._value[key] = (0, 0)

        num, v = self._value[key]
        num += 1

        if self.alpha:
            # Exponential moving average
            v = v * self.alpha + value * (1 - self.alpha)
        else:
            # Simple accumulation
            v += value

        self._value[key] = (num, v)

    def __str__(self) -> str:
        """Format all metrics as string."""
        s = ""
        for key in self._value:
            s += "%s = %.3e  " % (key, self[key])
        if len(self._value) > 0:
            return ("iter = %d  " % self._value[key][0]) + s
        else:
            return s

    @property
    def header(self) -> str:
        """Get CSV header with all metric names."""
        return "iter," + ",".join(self._value.keys())

    @property
    def value(self) -> List:
        """Get list of all current values (for CSV export)."""
        values = []
        for key in self._value:
            values.append(self[key])
        if len(self._value) > 0:
            return [self._value[key][0]] + values
        else:
            return values


def resolution2sigma(rx, ry=None, rz=None, /, isotropic=False):
    """
    Convert voxel resolution to Gaussian sigma for point spread function (PSF).

    In MRI, the imaging process acts like a blur (convolution with PSF).
    This function computes the appropriate Gaussian sigma to model this blur.

    Different blur models:
    - In-plane (x, y): Sinc function (approximated as Gaussian with SINC_FWHM)
    - Through-plane (z): Gaussian with GAUSSIAN_FWHM

    Args:
        rx: Resolution in x (or all directions if ry, rz not given)
        ry: Resolution in y (optional)
        rz: Resolution in z (optional)
        isotropic: If True, use Gaussian in all directions

    Returns:
        Sigma value(s) corresponding to the resolution(s)
        - Single float: if single resolution given and isotropic=True
        - Tuple of 3 floats: if 3 resolutions given
        - Tensor: if input is tensor

    Example:
        # Anisotropic acquisition: 0.5mm in-plane, 2mm through-plane
        sigma_x, sigma_y, sigma_z = resolution2sigma(0.5, 0.5, 2.0)

        # Isotropic (e.g., for synthetic data)
        sigma = resolution2sigma(1.0, isotropic=True)

    Note:
        - FWHM (Full Width Half Maximum) relates to sigma by: FWHM ≈ 2.355 * sigma
        - Sinc approximation is better for in-plane MRI physics
    """
    # Select FWHM factors based on mode
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM  # In-plane: sinc-like PSF
        fz = GAUSSIAN_FWHM  # Through-plane: Gaussian PSF

    assert not ((ry is None) ^ (rz is None))  # Either both or neither

    # Handle different input types
    if ry is None:
        if isinstance(rx, float) or isinstance(rx, int):
            # Single scalar resolution
            if isotropic:
                return fx * rx
            else:
                return fx * rx, fy * rx, fz * rx
        elif isinstance(rx, torch.Tensor):
            # Tensor input
            if isotropic:
                return fx * rx
            else:
                assert rx.shape[-1] == 3
                return rx * torch.tensor([fx, fy, fz], dtype=rx.dtype, device=rx.device)
        elif isinstance(rx, List) or isinstance(rx, Tuple):
            # List/tuple input
            assert len(rx) == 3
            return resolution2sigma(rx[0], rx[1], rx[2], isotropic=isotropic)
        else:
            raise Exception(str(type(rx)))
    else:
        # Three separate values
        return fx * rx, fy * ry, fz * rz


def get_PSF(
    r_max: Optional[int] = None,
    res_ratio: Tuple[float, float, float] = (1, 1, 3),
    threshold: float = 1e-3,
    device: DeviceType = torch.device("cpu"),
    psf_type: str = "gaussian",
) -> torch.Tensor:
    """
    Generate 3D Point Spread Function (PSF) kernel.

    The PSF models the blurring that occurs during image acquisition.
    For MRI, this is typically:
    - Gaussian blur in slice direction
    - Sinc-like blur in-plane (due to k-space sampling)

    Args:
        r_max: Maximum kernel radius in voxels. If None, auto-computed
        res_ratio: Resolution anisotropy [rx/rmin, ry/rmin, rz/rmin]
            Example: (1, 1, 3) means z-resolution is 3x coarser
        threshold: Values below this are set to zero (for sparsity)
        device: Device to create kernel on
        psf_type: 'gaussian' or 'sinc'

    Returns:
        3D PSF kernel, normalized to sum to 1

    Example:
        # PSF for anisotropic MRI (0.5mm in-plane, 1.5mm through-plane)
        # Ratio = (1, 1, 3)
        psf = get_PSF(res_ratio=(1, 1, 3), psf_type='sinc')

        # Use for convolution-based forward model
        blurred = F.conv3d(image, psf[None, None], padding=psf.shape[0]//2)

    Note:
        - 'gaussian': Isotropic Gaussian (simpler, faster)
        - 'sinc': Sinc in-plane, Gaussian through-plane (more realistic)
        - Kernel is automatically cropped to remove zeros
    """
    # Convert resolution ratios to sigmas
    sigma_x, sigma_y, sigma_z = resolution2sigma(res_ratio, isotropic=False)

    # Auto-compute kernel radius if not specified
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)  # Minimum size

    # Create 3D coordinate grid
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")

    # Generate PSF based on type
    if psf_type == "gaussian":
        # Anisotropic 3D Gaussian
        psf = torch.exp(
            -0.5
            * (grid_x**2 / sigma_x**2 + grid_y**2 / sigma_y**2 + grid_z**2 / sigma_z**2)
        )
    elif psf_type == "sinc":
        # Sinc in-plane (radially symmetric) + Gaussian through-plane
        # This better models MRI slice selection profile
        psf = torch.sinc(
            torch.sqrt((grid_x / res_ratio[0]) ** 2 + (grid_y / res_ratio[1]) ** 2)
        ) ** 2 * torch.exp(-0.5 * grid_z**2 / sigma_z**2)
    else:
        raise TypeError(f"Unknown PSF type: <{psf_type}>!")

    # Threshold small values to zero (for sparsity)
    psf[psf.abs() < threshold] = 0

    # Crop to minimal bounding box
    rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item())
    ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
    rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()

    # Normalize to sum to 1
    psf = psf / psf.sum()

    return psf


NOT_DOC = True


def doc_mode():
    global NOT_DOC
    NOT_DOC = False


def not_doc():
    return NOT_DOC


class RST:
    def __init__(self, rst) -> None:
        self.rst = rst

    def __str__(self) -> str:
        return rst2txt(self.rst)


def rst(source: str):
    if not NOT_DOC:
        return source
    else:
        return RST(source)


def rst2txt(source: str) -> str:
    """
    adapted from https://stackoverflow.com/questions/57119361/convert-restructuredtext-to-plain-text-programmatically-in-python
    """
    try:
        import docutils.nodes
        import docutils.parsers.rst
        import docutils.utils
        import sphinx.writers.text
        import sphinx.builders.text
        import sphinx.util.osutil
        from sphinx.application import Sphinx

        # parser rst
        parser = docutils.parsers.rst.Parser()
        components = (docutils.parsers.rst.Parser,)
        settings = docutils.frontend.OptionParser(
            components=components
        ).get_default_values()
        document = docutils.utils.new_document("<rst-doc>", settings=settings)
        parser.parse(source, document)

        # sphinx
        _app = types.SimpleNamespace(
            srcdir=None,
            confdir=None,
            outdir=None,
            doctreedir="/",
            events=None,
            config=types.SimpleNamespace(
                text_newlines="native",
                text_sectionchars="=",
                text_add_secnumbers=False,
                text_secnumber_suffix=".",
            ),
            tags=set(),
            registry=types.SimpleNamespace(
                create_translator=lambda self,
                something,
                new_builder: sphinx.writers.text.TextTranslator(document, new_builder)
            ),
        )
        app = cast(Sphinx, _app)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builder = sphinx.builders.text.TextBuilder(app)
        translator = sphinx.writers.text.TextTranslator(document, builder)
        document.walkabout(translator)
        return str(translator.body)
    except Exception as e:
        logging.warning("Got the following error during rst conversion: %s", e)
        return source


def show_link(text: str, link: str) -> str:
    if NOT_DOC:
        return link
    else:
        return f"`{text} <{link}>`_"
