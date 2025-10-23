"""
Training Loop for NeSVoR MRI Super-Resolution

This module implements the main training function for the NeSVoR model.
It handles the complete training pipeline from dataset creation to final output.

Key Concepts:
-----------
1. **Coordinate Normalization**:
   - Centers and scales coordinates to improve optimization
   - Typical scaling: 30mm → 1.0 (makes gradients more balanced)
   - Undone after training to restore original coordinate system

2. **Mixed Precision Training**:
   - Uses FP16 (half precision) for speed and memory efficiency
   - Automatic gradient scaling to prevent underflow
   - Falls back to FP32 if numerical issues detected

3. **Adaptive Learning Rate**:
   - Starts with high LR for fast convergence
   - Decays at milestones for fine-tuning
   - Separate LR for encoding (hash grid) and network (MLP)

4. **Weight Decay Regularization**:
   - Applied only to network parameters, not encoding
   - Prevents overfitting to noise in input slices
   - Encourages smoother reconstructions

5. **Loss Components**:
   - Data fidelity: MSE between predicted and observed intensities
   - Slice variance: Learned per-slice noise level
   - Pixel variance: Learned per-pixel uncertainty
   - Image regularization: Total Variation / Edge-preserving / L2
   - Transformation regularization: Keep close to initial alignment
   - Bias regularization: Prevent bias field from dominating
   - Deformation regularization: Encourage smooth deformations

Training Procedure:
-----------------
1. Create point dataset from input slices
2. Apply coordinate centering and scaling
3. Initialize NeSVoR model with optimized transformations
4. Setup optimizer (AdamW) and learning rate scheduler
5. Training loop:
   - Sample random batch of points
   - Forward pass: predict intensities
   - Compute weighted loss (data + regularization)
   - Backward pass and parameter update
   - Log progress at milestones
6. Undo centering and scaling
7. Return trained model, updated transformations, and mask

Functions:
---------
train: Main training function for NeSVoR model
"""

from argparse import Namespace
from typing import List, Tuple
import time
import datetime
import torch
import torch.optim as optim
import logging
from utils import (
    MovingAverage,
    log_params,
    TrainLogger,
    Volume,
    Slice,
    PointDataset,
    RigidTransform,
)
from model.models import (
    INR,
    NeSVoR,
    D_LOSS,
    S_LOSS,
    DS_LOSS,
    I_REG,
    B_REG,
    T_REG,
    D_REG,
)


def train(slices: List[Slice], args: Namespace) -> Tuple[INR, List[Slice], Volume]:
    """
    Train NeSVoR model for MRI super-resolution reconstruction.

    This is the main training function that:
    1. Creates dataset from input slices
    2. Normalizes coordinates for stable optimization
    3. Initializes and trains the NeSVoR model
    4. Returns trained INR, updated slice transformations, and mask

    Args:
        slices: List of input Slice objects with low-resolution, thick slices
        args: Namespace with hyperparameters:
            - n_epochs or n_iter: Training duration
            - batch_size: Points per batch (typically 4096-16384)
            - learning_rate: Initial LR (typically 0.01)
            - milestones: LR decay schedule [0.5, 0.75, 0.9]
            - gamma: LR decay factor (typically 0.33)
            - single_precision: Use FP32 instead of FP16
            - weight_transformation: Weight for transformation regularization
            - weight_bias: Weight for bias field regularization
            - weight_image: Weight for image regularization (TV/edge/L2)
            - weight_deform: Weight for deformation regularization
            - device: 'cuda' or 'cpu'
            - And many more architecture/regularization parameters

    Returns:
        Tuple of (trained_inr, output_slices, mask):
            - trained_inr: INR model (can be sampled to generate volumes)
            - output_slices: Input slices with updated transformations
            - mask: Volume mask indicating reconstruction region

    Note:
        - Uses coordinate centering and scaling (undone before returning)
        - Mixed precision training for speed (FP16 by default)
        - Monitors gradient scaler for numerical stability
        - Logs progress at LR decay milestones
        - Can run on CPU (slow) or GPU (recommended)

    Example:
        >>> from argparse import Namespace
        >>> from utils import load_slices
        >>>
        >>> # Load input slices
        >>> slices = load_slices('input_folder/')
        >>>
        >>> # Setup hyperparameters
        >>> args = Namespace(
        ...     n_epochs=100,
        ...     batch_size=8192,
        ...     learning_rate=0.01,
        ...     milestones=[0.5, 0.75, 0.9],
        ...     gamma=0.33,
        ...     single_precision=False,  # Use FP16
        ...     weight_transformation=0.1,
        ...     weight_bias=0.1,
        ...     weight_image=0.01,
        ...     weight_deform=0.01,
        ...     image_regularization='TV',  # Total Variation
        ...     device='cuda',
        ...     # ... many more parameters
        ... )
        >>>
        >>> # Train model
        >>> model, slices_out, mask = train(slices, args)
        >>>
        >>> # Sample high-resolution volume
        >>> from model.sample import sample_volume
        >>> volume = sample_volume(model, mask.resample(0.5), 0.5)
    """
    # ========== Dataset Creation ==========
    # Create point-based dataset from slices
    dataset = PointDataset(slices)

    # Convert n_epochs to n_iter if specified
    if args.n_epochs is not None:
        args.n_iter = args.n_epochs * (dataset.v.numel() // args.batch_size)

    # ========== Coordinate Normalization ==========
    # Apply centering and scaling for stable optimization
    use_scaling = True  # Scale coordinates to ~[-1, 1] range
    use_centering = True  # Center at origin

    # Scaling factor: map ~60mm extent to ~2.0 range
    spatial_scaling = 30.0 if use_scaling else 1

    # Compute bounding box and center
    bb = dataset.bounding_box  # (2, 3) [min, max]
    center = (bb[0] + bb[-1]) / 2 if use_centering else torch.zeros_like(bb[0])

    # Apply centering and scaling to transformations
    # 1. Translate to center at origin: T = [-center]
    # 2. Scale translations: T' = T / spatial_scaling
    # 3. Compose with original transformations
    ax = (
        RigidTransform(torch.cat([torch.zeros_like(center), -center])[None])
        .compose(dataset.transformation)
        .axisangle()
    )
    ax[:, -3:] /= spatial_scaling  # Scale translation components only
    transformation = RigidTransform(ax)

    # Scale point coordinates
    dataset.xyz /= spatial_scaling

    # ========== Model Initialization ==========
    model = NeSVoR(
        transformation,
        dataset.resolution / spatial_scaling,  # Scaled resolution
        dataset.mean,  # Mean intensity for normalization
        (bb - center) / spatial_scaling,  # Scaled bounding box
        spatial_scaling,  # Store for regularization losses
        args,
    )

    # ========== Optimizer Setup ==========
    # Separate parameters into encoding (hash grid) and network (MLPs)
    params_net = []
    params_encoding = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            if "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)

    # Log parameter counts
    logging.debug(log_params(model))

    # AdamW optimizer with different settings for encoding and network
    optimizer = torch.optim.AdamW(
        params=[
            {"name": "encoding", "params": params_encoding},
            {
                "name": "net",
                "params": params_net,
                "weight_decay": 1e-2,  # Regularize network only
            },
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.99),  # Momentum parameters
        eps=1e-15,  # Small epsilon for numerical stability
    )

    # ========== Learning Rate Scheduler ==========
    # MultiStepLR: decay by gamma at each milestone
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,  # LR *= gamma at each milestone
    )

    # Convert relative milestones (0.5, 0.75, 0.9) to iteration numbers
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]

    # ========== Mixed Precision Setup ==========
    # Use FP16 unless single_precision flag is set and device supports it
    device_type = (
        args.device.type if isinstance(args.device, torch.device) else str(args.device)
    )
    fp16 = not args.single_precision and device_type in {"cuda", "mps"}

    # Gradient scaler for automatic mixed precision
    scaler = torch.amp.GradScaler(
        init_scale=1.0,  # Start with 1x scaling
        enabled=fp16 and device_type == "cuda",  # GradScaler only supported on CUDA
        growth_factor=2.0,  # Increase scale if no overflow
        backoff_factor=0.5,  # Decrease scale if overflow
        growth_interval=2000,  # Check every 2000 iterations
    )

    # ========== Training Loop ==========
    model.train()  # Set to training mode

    # Loss weights for different components
    loss_weights = {
        D_LOSS: 1,  # Data fidelity (MSE)
        S_LOSS: 1,  # Slice variance (log-likelihood)
        T_REG: args.weight_transformation,  # Transformation regularization
        B_REG: args.weight_bias,  # Bias field regularization
        I_REG: args.weight_image,  # Image regularization (TV/edge/L2)
        D_REG: args.weight_deform,  # Deformation regularization
    }

    # Moving average tracker for losses
    average = MovingAverage(1 - 0.001)  # Decay factor 0.999

    # Logging setup
    logging_header = False
    logging.info("NeSVoR training starts.")
    train_time = 0.0

    # Main training loop
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()

        # ========== Forward Pass ==========
        # Get random batch of points
        batch = dataset.get_batch(args.batch_size, args.device)

        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type=device_type, enabled=fp16):
            # Compute all loss components
            losses = model(**batch)

            # Weighted sum of losses
            loss = 0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    loss = loss + loss_weights[k] * losses[k]

        # ========== Backward Pass ==========
        # Scale loss and backward (for FP16 stability)
        scaler.scale(loss).backward()

        # Debug: check for NaN gradients
        if args.debug:
            for _name, _p in model.named_parameters():
                if _p.grad is not None and not _p.grad.isfinite().all():
                    logging.warning("iter %d: Found NaNs in the grad of %s", i, _name)

        # Update parameters with scaled gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Track time
        train_time += time.time() - train_step_start

        # Update moving averages
        for k in losses:
            average(k, losses[k].item())

        # ========== Logging and LR Decay ==========
        if (decay_milestones and i >= decay_milestones[0]) or i == args.n_iter:
            # Log current progress
            if not logging_header:
                # Create logger on first log
                train_logger = TrainLogger(
                    "time",
                    "epoch",
                    "iter",
                    *list(losses.keys()),
                    "lr",
                )
                logging_header = True

            # Log metrics
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)),
                dataset.epoch,
                i,
                *[average[k] for k in losses],
                optimizer.param_groups[0]["lr"],
            )

            # Decay learning rate at milestone
            if i < args.n_iter:
                decay_milestones.pop(0)
                scheduler.step()

            # Check gradient scaler health
            if scaler.is_enabled():
                current_scaler = scaler.get_scale()
                if current_scaler < 1 / (2**5):
                    # Very low scale indicates numerical issues
                    logging.warning(
                        "Numerical instability detected! "
                        "The scale of GradScaler is %f, which is too small. "
                        "The results might be suboptimal. "
                        "Try to set --single-precision or run the command again with a different random seed."
                    )
                if i == args.n_iter:
                    logging.debug("Final scale of GradScaler = %f" % current_scaler)

    # ========== Post-Processing ==========
    # Extract optimized transformations
    transformation = model.transformation

    # Undo centering and scaling
    # 1. Scale translations back: T' = T * spatial_scaling
    # 2. Translate back to original center: T'' = [center] ∘ T'
    ax = transformation.axisangle()
    ax[:, -3:] *= spatial_scaling  # Undo translation scaling
    transformation = RigidTransform(ax)
    transformation = RigidTransform(
        torch.cat([torch.zeros_like(center), center])[None]
    ).compose(transformation)

    # Restore bounding box and point coordinates
    model.inr.bounding_box.copy_(bb)
    dataset.xyz *= spatial_scaling

    # Update dataset with final transformations
    dataset.transformation = transformation
    mask = dataset.mask

    # Create output slices with updated transformations
    output_slices = []
    for i in range(len(slices)):
        output_slice = slices[i].clone()
        output_slice.transformation = transformation[i]
        output_slices.append(output_slice)

    return model.inr, output_slices, mask
