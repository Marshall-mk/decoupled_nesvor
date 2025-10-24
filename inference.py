#!/usr/bin/env python
"""
NeSVoR Inference Script

Perform inference (sampling) on a pre-trained NeSVoR model.
This script loads a trained model and samples a high-resolution volume.

Usage:
    python inference.py --input-model model.pt --output-volume output.nii.gz

    python inference.py \
        --input-model trained_model.pt \
        --output-volume hr_volume.nii.gz \
        --output-resolution 0.5 \
        --n-inference-samples 256
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

from model.models import INR
from model.sample import sample_volume
from utils import Volume

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


def load_trained_model(model_path: str, device: torch.device):
    """
    Load a trained NeSVoR model from disk.

    Args:
        model_path: Path to the .pt model file
        device: Device to load model on (cuda or cpu)

    Returns:
        Tuple of (model, mask, args)
    """
    logging.info(f"Loading model from {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract components
    model_state = checkpoint["model"]
    mask = checkpoint["mask"]
    args = checkpoint["args"]

    # Create model instance
    model = INR(model_state["bounding_box"], args)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    logging.info("✓ Model loaded successfully")
    logging.info(f"  Bounding box: {model.bounding_box.cpu().numpy()}")

    return model, mask, args


def run_inference(
    model_path: str,
    output_volume_path: str,
    output_resolution: float = 0.8,
    n_inference_samples: int = 128,
    inference_batch_size: int = 1024,
    output_intensity_mean: float = None,
    with_background: bool = False,
    device: str = "cuda",
):
    """
    Run inference on a trained model.

    Args:
        model_path: Path to trained model (.pt file)
        output_volume_path: Where to save output volume (.nii.gz)
        output_resolution: Output resolution in mm (default: 0.8)
        n_inference_samples: Number of PSF samples (default: 128)
        inference_batch_size: Batch size for sampling (default: 1024)
        output_intensity_mean: Optional intensity rescaling
        with_background: Include background in output (default: False)
        device: Device to use ('cuda', 'cuda:0', 'cpu', etc.)
    """
    # Setup device
    if isinstance(device, str):
        device = torch.device(device)

    logging.info("="*80)
    logging.info("NeSVoR Inference")
    logging.info("="*80)
    logging.info(f"Model: {model_path}")
    logging.info(f"Output: {output_volume_path}")
    logging.info(f"Resolution: {output_resolution} mm")
    logging.info(f"Device: {device}")
    logging.info(f"PSF samples: {n_inference_samples}")
    logging.info("="*80)

    # Load model
    model, mask, args = load_trained_model(model_path, device)

    # Sample volume
    logging.info("\nSampling high-resolution volume...")
    logging.info(f"  Resolution: {output_resolution} mm")
    logging.info(f"  PSF samples: {n_inference_samples}")
    logging.info(f"  Batch size: {inference_batch_size}")

    output_volume = sample_volume(
        model,
        mask,
        psf_resolution=output_resolution,
        batch_size=inference_batch_size,
        n_samples=n_inference_samples,
    )

    logging.info(f"✓ Volume sampled: {output_volume.image.shape}")
    logging.info(f"  Intensity range: [{output_volume.image.min().item():.2f}, {output_volume.image.max().item():.2f}]")

    # Rescale intensity if requested
    if output_intensity_mean is not None:
        logging.info(f"\nRescaling intensity to mean={output_intensity_mean}")
        output_volume.rescale(output_intensity_mean)

    # Save output
    logging.info(f"\nSaving volume to {output_volume_path}")
    output_volume.save(output_volume_path, masked=not with_background)
    logging.info("✓ Volume saved successfully")

    logging.info("\n" + "="*80)
    logging.info("INFERENCE COMPLETED")
    logging.info("="*80)

    return output_volume


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="NeSVoR Inference - Sample high-resolution volume from trained model"
    )

    # Required arguments
    parser.add_argument(
        "--input-model",
        required=True,
        help="Path to trained model file (.pt)"
    )
    parser.add_argument(
        "--output-volume",
        required=True,
        help="Path to output volume file (.nii.gz)"
    )

    # Sampling parameters
    parser.add_argument(
        "--output-resolution",
        type=float,
        default=0.8,
        help="Output volume resolution in mm (default: 0.8)"
    )
    parser.add_argument(
        "--n-inference-samples",
        type=int,
        default=128,
        help="Number of PSF samples for inference (default: 128)"
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1024,
        help="Batch size for inference (default: 1024)"
    )

    # Output options
    parser.add_argument(
        "--output-intensity-mean",
        type=float,
        default=None,
        help="Rescale output to this mean intensity (optional)"
    )
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Include background in output (default: masked)"
    )

    # System options
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda', 'cuda:0', 'cpu', etc. (default: auto)"
    )

    args = parser.parse_args()

    # Run inference
    try:
        run_inference(
            model_path=args.input_model,
            output_volume_path=args.output_volume,
            output_resolution=args.output_resolution,
            n_inference_samples=args.n_inference_samples,
            inference_batch_size=args.inference_batch_size,
            output_intensity_mean=args.output_intensity_mean,
            with_background=args.with_background,
            device=args.device,
        )
        return 0
    except Exception as e:
        logging.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
