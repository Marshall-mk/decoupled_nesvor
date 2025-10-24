"""
NeSVoR2 Main Entry Point

This is the main entry point for running the NeSVoR2 MRI super-resolution pipeline.
The pipeline consists of the following phases:
1. Load inputs (stacks, masks, models)
2. Segmentation (2D fetal brain masking)
3. Bias field correction (N4 algorithm)
4. Assessment (stack quality metrics)
5. Registration (motion correction) - PLACEHOLDER
6. Reconstruction (NeSVoR training)
7. Sampling (generate high-resolution volume)
8. Save outputs

Each phase can be run independently or as part of the full pipeline.
Benchmarking metrics (time, memory, FLOPs) are tracked for each phase.

Usage:
    python main.py --config config.json
    python main.py --input-stacks stack1.nii.gz stack2.nii.gz --output-volume output.nii.gz

Example:
    python main.py \\
        --input-stacks stack1.nii.gz stack2.nii.gz stack3.nii.gz \\
        --stack-masks mask1.nii.gz mask2.nii.gz mask3.nii.gz \\
        --thicknesses 3.0 3.0 3.0 \\
        --segmentation \\
        --bias-field-correction \\
        --output-volume fetal_brain.nii.gz \\
        --output-resolution 0.8 \\
        --n-iter 5000 \\
        --batch-size 4096 \\
        --benchmark
"""

from argparse import Namespace, ArgumentParser
import logging
import time
import json
import os
import psutil
import gc
from typing import Any, Dict, Optional, Tuple, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from tqdm import tqdm

# Use absolute imports to allow running as script
from utils import (
    load_stack,
    load_slices,
    save_slices,
    merge_args,
    load_mask,
    Volume,
    Stack,
    Slice,
    rst,
)
from model.models import INR
from model.train import train
from model.sample import sample_volume, sample_slices
from preprocess import (
    stack_intersect,
    otsu_thresholding,
    thresholding,
    n4_bias_field_correction,
    brain_segmentation,
    assess,
)


# ============================================================================
# Benchmarking Infrastructure
# ============================================================================


@dataclass
class PhaseMetrics:
    """Metrics for a single pipeline phase."""

    phase_name: str
    start_time: float
    end_time: float
    duration: float  # seconds
    peak_memory_mb: float  # MB
    memory_allocated_mb: float  # MB (GPU)
    memory_reserved_mb: float  # MB (GPU)
    flops: Optional[int] = None  # FLOPs (if available)
    status: str = "success"  # success, failed, skipped
    error: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Overall benchmark results for the entire pipeline."""

    total_duration: float  # seconds
    peak_memory_mb: float  # MB
    phases: List[PhaseMetrics]

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "total_duration": self.total_duration,
            "peak_memory_mb": self.peak_memory_mb,
            "phases": [p.to_dict() for p in self.phases],
        }

    def save(self, path: str):
        """Save benchmark results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Benchmark results saved to {path}")


class BenchmarkTracker:
    """Tracks benchmarking metrics across pipeline phases."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.phases: List[PhaseMetrics] = []
        self.pipeline_start_time = time.time()
        self.peak_memory = 0.0

    @contextmanager
    def track_phase(self, phase_name: str):
        """Context manager for tracking a single phase."""
        if not self.enabled:
            yield
            return

        # Record start metrics
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            start_gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        else:
            start_gpu_allocated = 0
            start_gpu_reserved = 0

        # Run the phase
        error = None
        status = "success"
        try:
            logging.info(f"[BENCHMARK] Starting phase: {phase_name}")
            yield
        except Exception as e:
            error = str(e)
            status = "failed"
            logging.error(f"[BENCHMARK] Phase {phase_name} failed: {error}")
            raise
        finally:
            # Record end metrics
            end_time = time.time()
            duration = end_time - start_time

            end_mem = process.memory_info().rss / 1024 / 1024  # MB
            peak_mem = end_mem
            self.peak_memory = max(self.peak_memory, peak_mem)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_gpu_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
                peak_gpu_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
            else:
                peak_gpu_allocated = 0
                peak_gpu_reserved = 0

            # Create metrics record
            metrics = PhaseMetrics(
                phase_name=phase_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                peak_memory_mb=peak_mem,
                memory_allocated_mb=peak_gpu_allocated,
                memory_reserved_mb=peak_gpu_reserved,
                status=status,
                error=error,
            )

            self.phases.append(metrics)

            logging.info(
                f"[BENCHMARK] Phase {phase_name} completed: "
                f"duration={duration:.2f}s, "
                f"memory={peak_mem:.1f}MB, "
                f"GPU={peak_gpu_allocated:.1f}MB"
            )

            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def finalize(self) -> BenchmarkResults:
        """Finalize and return benchmark results."""
        total_duration = time.time() - self.pipeline_start_time
        return BenchmarkResults(
            total_duration=total_duration,
            peak_memory_mb=self.peak_memory,
            phases=self.phases,
        )


# ============================================================================
# Pipeline Phases
# ============================================================================


def load_inputs(args: Namespace, benchmark: BenchmarkTracker) -> Tuple[Dict, Namespace]:
    """
    Phase 1: Load input stacks, masks, and models.

    Args:
        args: Command-line arguments
        benchmark: Benchmark tracker

    Returns:
        input_dict: Dictionary containing loaded inputs
        args: Updated arguments (merged with model args if loading model)
    """
    with benchmark.track_phase("load_inputs"):
        input_dict: Dict[str, Any] = {}

        # Load input stacks
        if getattr(args, "input_stacks", None) is not None:
            input_stacks = []
            logging.info(f"Loading {len(args.input_stacks)} input stacks")

            for i, f in enumerate(args.input_stacks):
                stack = load_stack(
                    f,
                    args.stack_masks[i]
                    if getattr(args, "stack_masks", None) is not None
                    else None,
                    device=args.device,
                )

                # Set thickness if provided
                if getattr(args, "thicknesses", None) is not None:
                    stack.thickness = args.thicknesses[i]

                input_stacks.append(stack)
                logging.info(
                    f"  Stack {i + 1}: shape={stack.slices.shape}, "
                    f"resolution={stack.resolution_x:.2f}mm, "
                    f"thickness={stack.thickness:.2f}mm"
                )

            # Background thresholding
            if getattr(args, "background_threshold", 0.0) > 0:
                logging.info("Applying background thresholding")
                input_stacks = thresholding(input_stacks, args.background_threshold)

            # Otsu thresholding
            if getattr(args, "otsu_thresholding", False):
                logging.info("Applying Otsu thresholding")
                input_stacks = otsu_thresholding(input_stacks)

            # Load or create volume mask
            volume_mask: Optional[Volume] = None
            if getattr(args, "volume_mask", None):
                logging.info("Loading volume mask")
                volume_mask = load_mask(args.volume_mask, device=args.device)
            elif getattr(args, "stacks_intersection", False):
                logging.info("Creating volume mask from stack intersection")
                volume_mask = stack_intersect(input_stacks, box=True)

            # Apply volume mask
            if volume_mask is not None:
                logging.info("Applying volume mask to stacks")
                for stack in input_stacks:
                    stack.apply_volume_mask(volume_mask)

            input_dict["input_stacks"] = input_stacks
            input_dict["volume_mask"] = volume_mask

        # Load input slices (if provided directly)
        if getattr(args, "input_slices", None) is not None:
            logging.info("Loading input slices")
            input_slices = load_slices(args.input_slices, args.device)
            input_dict["input_slices"] = input_slices

        # Load pre-trained model (if resuming or sampling only)
        if getattr(args, "input_model", None) is not None:
            logging.info(f"Loading pre-trained model from {args.input_model}")
            cp = torch.load(args.input_model, map_location=args.device)
            input_dict["model"] = INR(cp["model"]["bounding_box"], cp["args"])
            input_dict["model"].load_state_dict(cp["model"])
            input_dict["mask"] = cp["mask"]
            args = merge_args(cp["args"], args)
            logging.info("Model loaded successfully")

        return input_dict, args


def run_segmentation(
    input_dict: Dict, args: Namespace, benchmark: BenchmarkTracker
) -> Dict:
    """
    Phase 2: Segment brain regions (2D fetal brain masking).

    Args:
        input_dict: Dictionary containing input_stacks
        args: Command-line arguments
        benchmark: Benchmark tracker

    Returns:
        Updated input_dict with segmented masks
    """
    if not getattr(args, "segmentation", False):
        logging.info("Skipping segmentation (not requested)")
        return input_dict

    with benchmark.track_phase("segmentation"):
        if "input_stacks" not in input_dict:
            logging.warning("No input stacks to segment")
            return input_dict

        input_stacks = input_dict["input_stacks"]
        logging.info(f"Running brain segmentation on {len(input_stacks)} stacks")

        # Run segmentation on each stack
        segmented_stacks = brain_segmentation(input_stacks, args)

        input_dict["input_stacks"] = segmented_stacks
        logging.info("Segmentation completed")

        return input_dict


def run_bias_correction(
    input_dict: Dict, args: Namespace, benchmark: BenchmarkTracker
) -> Dict:
    """
    Phase 3: Bias field correction using N4 algorithm.

    Args:
        input_dict: Dictionary containing input_stacks
        args: Command-line arguments
        benchmark: Benchmark tracker

    Returns:
        Updated input_dict with bias-corrected stacks
    """
    if not getattr(args, "bias_field_correction", False):
        logging.info("Skipping bias field correction (not requested)")
        return input_dict

    with benchmark.track_phase("bias_correction"):
        if "input_stacks" not in input_dict:
            logging.warning("No input stacks for bias correction")
            return input_dict

        input_stacks = input_dict["input_stacks"]
        logging.info(f"Running N4 bias field correction on {len(input_stacks)} stacks")

        # Run N4 on each stack
        corrected_stacks = []
        for i, stack in enumerate(input_stacks):
            logging.info(f"  Correcting stack {i + 1}/{len(input_stacks)}")
            corrected_stack = n4_bias_field_correction(stack)
            corrected_stacks.append(corrected_stack)

        input_dict["input_stacks"] = corrected_stacks
        input_dict["output_corrected_stacks"] = corrected_stacks
        logging.info("Bias field correction completed")

        return input_dict


def run_assessment(
    input_dict: Dict, args: Namespace, benchmark: BenchmarkTracker
) -> Dict:
    """
    Phase 4: Assess stack quality (motion metrics, SNR, etc.).

    Args:
        input_dict: Dictionary containing input_stacks
        args: Command-line arguments
        benchmark: Benchmark tracker

    Returns:
        Updated input_dict with assessment results
    """
    if getattr(args, "skip_assessment", False):
        logging.info("Skipping assessment (disabled)")
        return input_dict

    with benchmark.track_phase("assessment"):
        if "input_stacks" not in input_dict:
            logging.warning("No input stacks to assess")
            return input_dict

        input_stacks = input_dict["input_stacks"]
        logging.info(f"Assessing quality of {len(input_stacks)} stacks")

        # Run assessment with configured metric/filtering options
        augmentation = not getattr(args, "no_augmentation_assess", False)
        filtered_stacks, assessment_results = assess(
            input_stacks,
            getattr(args, "metric", "none"),
            getattr(args, "filter_method", "none"),
            getattr(args, "cutoff", 0.0) or 0.0,
            getattr(args, "batch_size_assess", 8),
            augmentation,
            args.device,
        )

        input_dict["input_stacks"] = filtered_stacks
        input_dict["assessment_results"] = assessment_results
        logging.info("Assessment completed")

        # Log key metrics
        for i, result in enumerate(assessment_results):
            logging.info(f"  Stack {i + 1} metrics: {result}")

        return input_dict


def run_registration(
    input_dict: Dict, args: Namespace, benchmark: BenchmarkTracker
) -> Dict:
    """
    Phase 5: Registration / motion correction using SVoRT.

    Applies motion correction to input stacks using SVoRT (Slice-to-Volume Registration
    Transformer) if enabled, otherwise converts stacks to slices without registration.

    Args:
        input_dict: Dictionary containing input_stacks
        args: Command-line arguments (registration, svort_version, svort_use_vvr, etc.)
        benchmark: Benchmark tracker

    Returns:
        Updated input_dict with registered slices in input_slices

    Registration Options:
        - registration=False: Skip registration, just flatten stacks to slices
        - registration=True, svort=True: Use SVoRT transformer-based registration
        - registration=True, svort=True, use_vvr=True: Use SVoRT + traditional stack registration
    """
    if not getattr(args, "registration", False):
        logging.info("Skipping registration (not requested)")
        # Convert stacks to slices without registration
        if "input_stacks" in input_dict and "input_slices" not in input_dict:
            slices = []
            for stack in input_dict["input_stacks"]:
                for i in range(stack.slices.shape[0]):
                    slice_img = stack.slices[i]
                    slice_mask = stack.mask[i] if stack.mask is not None else None
                    slice_obj = Slice(
                        slice_img,
                        slice_mask,
                        stack.transformation[i],
                        stack.resolution_x,
                        stack.resolution_y,
                        stack.thickness,
                    )
                    slices.append(slice_obj)
            input_dict["input_slices"] = slices
            logging.info(f"Created {len(slices)} slices from stacks (no registration)")
        return input_dict

    with benchmark.track_phase("registration"):
        if "input_stacks" not in input_dict:
            logging.warning("No input stacks for registration")
            return input_dict

        input_stacks = input_dict["input_stacks"]
        logging.info(f"Running registration on {len(input_stacks)} stacks")

        # Get registration parameters
        use_svort = getattr(args, "svort", True)
        svort_version = getattr(args, "svort_version", "v2")
        use_vvr = getattr(args, "use_vvr", False)
        force_vvr = getattr(args, "force_vvr", False)
        force_scanner = getattr(args, "force_scanner", False)

        if use_svort:
            # Import SVoRT registration
            from preprocess.svort import svort_predict

            logging.info(f"Running SVoRT {svort_version} registration")
            logging.info(
                f"  use_vvr={use_vvr}, force_vvr={force_vvr}, force_scanner={force_scanner}"
            )

            # Run SVoRT registration
            registered_slices = svort_predict(
                dataset=input_stacks,
                device=args.device,
                svort_version=svort_version,
                svort=True,
                vvr=use_vvr,
                force_vvr=force_vvr,
                force_scanner=force_scanner,
            )

            input_dict["input_slices"] = registered_slices
            logging.info(
                f"SVoRT registration completed: {len(registered_slices)} slices"
            )

        else:
            # No SVoRT: just convert stacks to slices
            logging.info(
                "SVoRT disabled, converting stacks to slices without registration"
            )
            slices = []
            for stack in input_stacks:
                for i in range(stack.slices.shape[0]):
                    slice_img = stack.slices[i]
                    slice_mask = stack.mask[i] if stack.mask is not None else None
                    slice_obj = Slice(
                        slice_img,
                        slice_mask,
                        stack.transformation[i],
                        stack.resolution_x,
                        stack.resolution_y,
                        stack.thickness,
                    )
                    slices.append(slice_obj)
            input_dict["input_slices"] = slices
            logging.info(f"Created {len(slices)} slices from stacks")

        return input_dict


def run_reconstruction(
    input_dict: Dict, args: Namespace, benchmark: BenchmarkTracker
) -> Dict:
    """
    Phase 6: Train NeSVoR model (reconstruction).

    Args:
        input_dict: Dictionary containing input_slices
        args: Command-line arguments
        benchmark: Benchmark tracker

    Returns:
        Updated input_dict with trained model
    """
    if getattr(args, "skip_reconstruction", False):
        logging.info("Skipping reconstruction (disabled)")
        return input_dict

    with benchmark.track_phase("reconstruction"):
        if "input_slices" not in input_dict:
            logging.error("No input slices for reconstruction")
            raise ValueError("input_slices required for reconstruction")

        input_slices = input_dict["input_slices"]
        logging.info(f"Training NeSVoR model on {len(input_slices)} slices")
        logging.info(
            f"Training parameters: n_iter={args.n_iter}, batch_size={args.batch_size}"
        )

        # Create a progress bar for training iterations
        if not getattr(args, "debug", False):
            pbar = tqdm(
                total=args.n_iter,
                desc="NeSVoR Training",
                unit="iter",
                ncols=100,
                dynamic_ncols=True
            )
            args.progress_bar = pbar
        else:
            args.progress_bar = None

        try:
            # Train model
            model, output_slices, mask = train(input_slices, args)
        finally:
            # Clean up progress bar
            if hasattr(args, 'progress_bar') and args.progress_bar is not None:
                args.progress_bar.close()
                delattr(args, 'progress_bar')

        input_dict["output_model"] = model
        input_dict["output_slices"] = output_slices
        input_dict["mask"] = mask

        logging.info("Reconstruction completed")

        return input_dict


def run_sampling(
    input_dict: Dict, args: Namespace, benchmark: BenchmarkTracker
) -> Dict:
    """
    Phase 7: Sample high-resolution volume from trained INR.

    Args:
        input_dict: Dictionary containing output_model and mask
        args: Command-line arguments
        benchmark: Benchmark tracker

    Returns:
        Updated input_dict with sampled volume
    """
    if not getattr(args, "output_volume", None):
        logging.info("Skipping sampling (no output volume requested)")
        return input_dict

    with benchmark.track_phase("sampling"):
        if "output_model" not in input_dict or "mask" not in input_dict:
            logging.error("No trained model or mask for sampling")
            raise ValueError("output_model and mask required for sampling")

        model = input_dict["output_model"]
        mask = input_dict["mask"]

        # Get sampling parameters
        output_resolution = getattr(args, "output_resolution", 0.8)
        n_samples = getattr(args, "n_inference_samples", 128)
        batch_size = getattr(args, "inference_batch_size", 1024)

        logging.info(
            f"Sampling volume at {output_resolution}mm resolution "
            f"with {n_samples} PSF samples"
        )

        # Sample volume
        output_volume = sample_volume(
            model,
            mask,
            psf_resolution=output_resolution,
            batch_size=batch_size,
            n_samples=n_samples,
        )

        input_dict["output_volume"] = output_volume

        logging.info(f"Volume sampled: shape={output_volume.image.shape}")

        # Sample simulated slices if requested
        if getattr(args, "simulated_slices", None) and "output_slices" in input_dict:
            logging.info("Sampling simulated slices")
            output_slices = input_dict["output_slices"]

            simulated = sample_slices(
                model,
                output_slices,
                mask,
                output_psf_factor=1.0,
                n_samples=n_samples,
            )

            input_dict["simulated_slices"] = simulated
            logging.info(f"Sampled {len(simulated)} simulated slices")

        return input_dict


def save_outputs(
    input_dict: Dict, args: Namespace, benchmark: BenchmarkTracker
) -> None:
    """
    Phase 8: Save all output files.

    Args:
        input_dict: Dictionary containing all outputs
        args: Command-line arguments
        benchmark: Benchmark tracker
    """
    with benchmark.track_phase("save_outputs"):
        # Save output volume
        if getattr(args, "output_volume", None) and "output_volume" in input_dict:
            logging.info(f"Saving output volume to {args.output_volume}")
            volume = input_dict["output_volume"]

            # Rescale if requested
            if getattr(args, "output_intensity_mean", None):
                volume.rescale(args.output_intensity_mean)

            # Save (with or without background)
            volume.save(
                args.output_volume, masked=not getattr(args, "with_background", False)
            )

        # Save trained model
        if getattr(args, "output_model", None) and "output_model" in input_dict:
            logging.info(f"Saving model to {args.output_model}")
            torch.save(
                {
                    "model": input_dict["output_model"].state_dict(),
                    "mask": input_dict["mask"],
                    "args": args,
                },
                args.output_model,
            )

        # Save output slices
        if getattr(args, "output_slices", None) and "output_slices" in input_dict:
            logging.info(f"Saving output slices to {args.output_slices}")
            save_slices(args.output_slices, input_dict["output_slices"], sep=True)

        # Save simulated slices
        if getattr(args, "simulated_slices", None) and "simulated_slices" in input_dict:
            logging.info(f"Saving simulated slices to {args.simulated_slices}")
            save_slices(
                args.simulated_slices, input_dict["simulated_slices"], sep=False
            )

        # Save corrected stacks
        if (
            getattr(args, "output_corrected_stacks", None)
            and "output_corrected_stacks" in input_dict
        ):
            logging.info("Saving bias-corrected stacks")
            for stack, path in zip(
                input_dict["output_corrected_stacks"], args.output_corrected_stacks
            ):
                stack.save(path)

        # Save stack masks
        if getattr(args, "output_stack_masks", None) and "input_stacks" in input_dict:
            logging.info("Saving stack masks")
            for stack, path in zip(input_dict["input_stacks"], args.output_stack_masks):
                if stack.mask is not None:
                    # Create volume from mask
                    mask_vol = Volume(
                        stack.mask.float(),
                        stack.mask,
                        stack.transformation,
                        stack.resolution_x,
                        stack.resolution_y,
                        stack.thickness,
                    )
                    mask_vol.save(path)

        # Save assessment results
        if getattr(args, "output_json", None):
            logging.info(f"Saving configuration and results to {args.output_json}")
            output_data = vars(args).copy()

            # Convert device to serializable format
            if "device" in output_data:
                output_data["device"] = str(output_data["device"])

            # Add assessment results if available
            if "assessment_results" in input_dict:
                output_data["assessment_results"] = input_dict["assessment_results"]

            with open(args.output_json, "w") as f:
                json.dump(output_data, f, indent=2)


# ============================================================================
# Main Pipeline
# ============================================================================


def run_pipeline(args: Namespace) -> BenchmarkResults:
    """
    Run the complete NeSVoR2 pipeline.

    Args:
        args: Command-line arguments

    Returns:
        BenchmarkResults containing metrics for all phases
    """
    # Initialize benchmark tracker
    benchmark = BenchmarkTracker(enabled=getattr(args, "benchmark", False))

    # Initialize data dictionary
    input_dict: Dict[str, Any] = {}

    try:
        # Phase 1: Load inputs
        input_dict, args = load_inputs(args, benchmark)
        logging.info("Inputs loaded successfully.")

        # Phase 2: Segmentation
        input_dict = run_segmentation(input_dict, args, benchmark)
        logging.info("Segmentation completed successfully.")

        # Phase 3: Bias correction
        input_dict = run_bias_correction(input_dict, args, benchmark)

        # Phase 4: Assessment
        input_dict = run_assessment(input_dict, args, benchmark)

        # Phase 5: Registration 
        input_dict = run_registration(input_dict, args, benchmark)

        # Phase 6: Reconstruction
        input_dict = run_reconstruction(input_dict, args, benchmark)
        logging.info("Reconstruction completed successfully.")

        # Phase 7: Sampling
        input_dict = run_sampling(input_dict, args, benchmark)

        # Phase 8: Save outputs
        save_outputs(input_dict, args, benchmark)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise
    finally:
        # Finalize benchmark
        results = benchmark.finalize()

        # Save benchmark results if requested
        if getattr(args, "benchmark", False):
            benchmark_path = getattr(args, "benchmark_output", "benchmark_results.json")
            results.save(benchmark_path)

            # Print summary
            logging.info("=" * 80)
            logging.info("BENCHMARK SUMMARY")
            logging.info("=" * 80)
            logging.info(f"Total duration: {results.total_duration:.2f}s")
            logging.info(f"Peak memory: {results.peak_memory_mb:.1f}MB")
            logging.info("")
            logging.info("Phase breakdown:")
            for phase in results.phases:
                logging.info(
                    f"  {phase.phase_name:20s}: "
                    f"{phase.duration:6.2f}s  "
                    f"({phase.duration / results.total_duration * 100:5.1f}%)  "
                    f"[{phase.status}]"
                )
            logging.info("=" * 80)

    return results


def create_default_args() -> Namespace:
    """Create default arguments with sensible defaults."""
    parser = ArgumentParser(
        description="NeSVoR2: MRI Super-Resolution via Implicit Neural Representations"
    )

    # Input options
    parser.add_argument("--input-stacks", nargs="+", help="Input stack NIfTI files")
    parser.add_argument("--stack-masks", nargs="+", help="Stack mask NIfTI files")
    parser.add_argument(
        "--thicknesses", nargs="+", type=float, help="Slice thicknesses (mm)"
    )
    parser.add_argument("--input-slices", help="Input slices directory")
    parser.add_argument("--input-model", help="Pre-trained model file (.pt)")
    parser.add_argument("--volume-mask", help="Volume mask NIfTI file")
    parser.add_argument(
        "--stacks-intersection",
        action="store_true",
        help="Create volume mask from stack intersection",
    )

    # Preprocessing options - Segmentation
    parser.add_argument(
        "--segmentation", action="store_true", help="Run brain segmentation"
    )
    parser.add_argument(
        "--batch-size-seg",
        type=int,
        default=16,
        help="Batch size for segmentation",
    )
    parser.add_argument(
        "--no-augmentation-seg",
        action="store_true",
        help="Disable inference data augmentation in segmentation",
    )
    parser.add_argument(
        "--dilation-radius-seg",
        type=float,
        default=1.0,
        help="Dilation radius for segmentation mask in millimeter.",
    )
    parser.add_argument(
        "--threshold-small-seg",
        type=float,
        default=0.1,
        help=(
            "Threshold for removing small segmentation mask (between 0 and 1). "
            "A mask will be removed if its area < threshold * max area in the stack."
        ),
    )

    # Preprocessing options - N4 Bias Field Correction
    parser.add_argument(
        "--bias-field-correction",
        action="store_true",
        help="Run N4 bias field correction",
    )
    parser.add_argument(
        "--n-levels-bias",
        default=0,
        type=int,
        help="Number of levels used for bias field estimation.",
    )
    parser.add_argument(
        "--n-proc-n4",
        type=int,
        default=8,
        help="Number of workers for the N4 algorithm.",
    )
    parser.add_argument(
        "--shrink-factor-n4",
        type=int,
        default=2,
        help="The shrink factor used to reduce the size and complexity of the image.",
    )
    parser.add_argument(
        "--tol-n4",
        type=float,
        default=0.001,
        help="The convergence threshold in N4.",
    )
    parser.add_argument(
        "--spline-order-n4",
        type=int,
        default=3,
        help="The order of B-spline.",
    )
    parser.add_argument(
        "--noise-n4",
        type=float,
        default=0.01,
        help="The noise estimate defining the Wiener filter.",
    )
    parser.add_argument(
        "--n-iter-n4",
        type=int,
        default=50,
        help="The maximum number of iterations specified at each fitting level.",
    )
    parser.add_argument(
        "--n-levels-n4",
        type=int,
        default=4,
        help="The number of fitting levels.",
    )
    parser.add_argument(
        "--n-control-points-n4",
        type=int,
        default=4,
        help=(
            "The control point grid size in each dimension. "
            "The B-spline mesh size is equal to the number of control points in that dimension minus the spline order."
        ),
    )
    parser.add_argument(
        "--n-bins-n4",
        type=int,
        default=200,
        help="The number of bins in the log input intensity histogram.",
    )
    parser.add_argument(
        "--skip-assessment", action="store_true", help="Skip quality assessment"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["ncc", "matrix-rank", "volume", "iqa2d", "iqa3d", "none"],
        default="none",
        help=rst(
            "Metric for assessing input stacks. \n\n"
            "1. ``ncc`` (\u2191): cross correlaiton between adjacent slices; \n"
            "2. ``matrix-rank`` (\u2193): motion metric based on the rank of the data matrix; \n"
            "3. ``volume`` (\u2191): volume of the masked ROI; \n"
            "4. ``iqa2d`` (\u2191): image quality score generated by a `2D CNN <https://arxiv.org/abs/2006.12704>`_ (only for fetal brain), the score of a stack is the average score of the images in it; \n"
            "5. ``iqa3d`` (\u2191): image quality score generated by a `3D CNN <https://github.com/FNNDSC/pl-fetal-brain-assessment>`_ (only for fetal brain); \n"
            "6. ``none``: no metric used for assessment. \n\n"
            "**Note**: (\u2191) means a stack with a higher score will have a better rank.\n"
        ),
    )
    parser.add_argument(
        "--filter-method",
        type=str,
        choices=["top", "bottom", "threshold", "percentage", "none"],
        default="none",
        help=rst(
            "Method to remove low-quality stacks. \n\n"
            "1. ``top``: keep the top ``C`` stacks; \n"
            "2. ``bottom``: remove the bottom ``C`` stacks; \n"
            "3. ``threshold``: remove a stack if the metric is worse than ``C``; \n"
            "4. ``percentatge``: remove the bottom (``num_stack * C``) stacks; \n"
            "5. ``none``: no filtering. \n\n"
            "The value of ``C`` is specified by `--cutoff <#cutoff>`__. \n"
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help=rst(
            "The cutoff value for filtering, i.e., the value ``C`` in `--filter-method <#filter-method>`__"
        ),
    )
    parser.add_argument(
        "--batch-size-assess", type=int, default=8, help="Batch size for IQA network"
    )
    parser.add_argument(
        "--no-augmentation-assess",
        action="store_true",
        help="Disable inference data augmentation in IQA network",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.0,
        help="Background intensity threshold",
    )
    parser.add_argument(
        "--otsu-thresholding", action="store_true", help="Apply Otsu thresholding"
    )

    # Registration options (SVoRT)
    parser.add_argument(
        "--registration", action="store_true", help="Run registration/motion correction"
    )
    parser.add_argument(
        "--svort",
        action="store_true",
        default=True,
        help="Use SVoRT for registration (default: True)",
    )
    parser.add_argument(
        "--no-svort",
        action="store_false",
        dest="svort",
        help="Disable SVoRT registration",
    )
    parser.add_argument(
        "--svort-version",
        default="v2",
        choices=["v1", "v2"],
        help="SVoRT model version (default: v2)",
    )
    parser.add_argument(
        "--use-vvr",
        action="store_true",
        help="Use traditional stack-to-stack registration (VVR) with SVoRT",
    )
    parser.add_argument(
        "--force-vvr", action="store_true", help="Force use of VVR results over SVoRT"
    )
    parser.add_argument(
        "--force-scanner",
        action="store_true",
        help="Map results back to scanner coordinate system",
    )

    # Training options
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Skip reconstruction (training)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=6000, help="Number of training iterations"
    )
    parser.add_argument("--n-epochs", type=int, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-3, help="Learning rate"
    )
    parser.add_argument(
        "--single-precision", action="store_true", help="Use FP32 instead of FP16"
    )
    parser.add_argument(
        "--gamma",
        default=0.33,
        type=float,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 0.9],
        help="List of milestones of learning rate decay. Must be in (0, 1) and increasing.",
    )
    parser.add_argument(
        "--n-samples",
        default=128 * 2,
        type=int,
        help="Number of sample for PSF during training.",
    )

    # Model architecture options
    parser.add_argument(
        "--coarsest-resolution",
        type=float,
        default=16.0,
        help="Coarsest hash grid resolution (mm)",
    )
    parser.add_argument(
        "--finest-resolution",
        type=float,
        default=0.5,
        help="Finest hash grid resolution (mm)",
    )
    parser.add_argument(
        "--level-scale", type=float, default=1.3819, help="Hash grid level scale factor"
    )
    parser.add_argument(
        "--n-features-per-level",
        type=int,
        default=2,
        help="Features per hash grid level",
    )
    parser.add_argument(
        "--log2-hashmap-size", type=int, default=19, help="log2 of hash table size"
    )
    parser.add_argument("--width", type=int, default=64, help="MLP hidden layer width")
    parser.add_argument(
        "--depth", type=int, default=1, help="MLP depth (hidden layers)"
    )
    parser.add_argument(
        "--n-features-z",
        default=15,
        type=int,
        help="Length of the intermediate feature vector z.",
    )
    parser.add_argument(
        "--n-features-slice",
        default=16,
        type=int,
        help="Length of the slice embedding vector e.",
    )
    parser.add_argument(
        "--no-transformation-optimization",
        action="store_true",
        help="Disable optimization for rigid slice transfromation, i.e., the slice transformations are fixed",
    )
    parser.add_argument(
        "--no-slice-scale",
        action="store_true",
        help="Disable adaptive scaling for slices.",
    )
    parser.add_argument(
        "--no-pixel-variance",
        action="store_true",
        help="Disable pixel-level variance.",
    )
    parser.add_argument(
        "--no-slice-variance",
        action="store_true",
        help="Disable slice-level variance.",
    )

    # Regularization options
    parser.add_argument(
        "--weight-transformation",
        type=float,
        default=0.1,
        help="Transformation regularization weight",
    )
    parser.add_argument(
        "--weight-bias",
        type=float,
        default=100.0,
        help="Bias field regularization weight",
    )
    parser.add_argument(
        "--weight-image", type=float, default=1.0, help="Image regularization weight"
    )
    parser.add_argument(
        "--image-regularization",
        default="edge",
        choices=["none", "TV", "edge", "L2"],
        help=rst(
            "Type of image regularization. \n\n"
            "1. ``TV``: total variation (L1 regularization of image gradient); \n"
            "2. ``edge``: edge-preserving regularization, see `--delta <#delta>`__\ . \n"
            "3. ``L2``: L2 regularization of image gradient; \n"
            "4. ``none``: no image regularization. \n\n"
        ),
    )
    parser.add_argument(
        "--weight-deform",
        default=0.1,
        type=float,
        help="Weight of deformation regularization ",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help=rst(
            "Parameter to define intensity of an edge in edge-preserving regularization. \n"
            "See `--image-regularization <#image-regularization>`__\n\n"
            "The edge-preserving regularization becomes L1 when ``delta`` goes to 0. \n"
        ),
    )
    parser.add_argument(
        "--img-reg-autodiff",
        action="store_true",
        help=(
            "Use auto diff to compute the image graident in the image regularization. "
            "By default, the finite difference is used."
        ),
    )

    # deformation options
    parser.add_argument(
        "--deformable",
        action="store_true",
        help="Enable implicit deformation field.",
    )
    parser.add_argument(
        "--n-features-deform",
        default=8,
        type=int,
        help="Length of the deformation embedding vector.",
    )
    parser.add_argument(
        "--n-features-per-level-deform",
        default=4,
        type=int,
        help="Length of the feature vector at each level (deformation field).",
    )
    parser.add_argument(
        "--level-scale-deform",
        default=1.3819,
        type=float,
        help="Scaling factor between two levels (deformation field).",
    )
    parser.add_argument(
        "--coarsest-resolution-deform",
        default=32.0,
        type=float,
        help="Resolution of the coarsest grid in millimeter (deformation field).",
    )
    parser.add_argument(
        "--finest-resolution-deform",
        default=8.0,
        type=float,
        help="Resolution of the finest grid in millimeter (deformation field).",
    )

    # Sampling options
    parser.add_argument(
        "--output-resolution",
        type=float,
        default=0.8,
        help="Output volume resolution (mm)",
    )
    parser.add_argument(
        "--n-inference-samples", type=int, default=128, help="PSF samples for inference"
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=1024, help="Inference batch size"
    )
    parser.add_argument(
        "--output-psf-factor",
        type=float,
        default=1.0,
        help="Determine the PSF for generating output volume: FWHM = output-resolution * output-psf-factor",
    )
    parser.add_argument(
        "--output-intensity-mean",
        type=float,
        default=700.0,
        help="Mean intensity of the output volume",
    )
    parser.add_argument(
        "--with-background", action="store_true", help="Include background in output"
    )
    parser.add_argument(
        "--sample-mask",
        type=str,
        help="3D Mask for sampling INR. If not provided, will use a mask estimated from the input data.",
    )
    parser.add_argument(
        "--sample-orientation",
        type=str,
        help="Path to a NIfTI file. The sampled volume will be reoriented according to the transformation in this file.",
    )

    # Output options
    parser.add_argument("--output-volume", help="Output volume NIfTI file")
    parser.add_argument("--output-model", help="Output model file (.pt)")
    parser.add_argument("--output-slices", help="Output slices directory")
    parser.add_argument("--simulated-slices", help="Simulated slices directory")
    parser.add_argument(
        "--output-corrected-stacks", nargs="+", help="Bias-corrected stack output files"
    )
    parser.add_argument(
        "--output-stack-masks", nargs="+", help="Stack mask output files"
    )
    parser.add_argument("--output-json", help="Output JSON file for results")

    # Benchmark options
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking (time, memory, FLOPs)",
    )
    parser.add_argument(
        "--benchmark-output",
        default="benchmark_results.json",
        help="Benchmark results output file",
    )

    # System options
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (disables progress bars)"
    )

    return parser


def main():
    """Main entry point."""
    # Parse arguments
    parser = create_default_args()
    args = parser.parse_args()

    # Setup logging FIRST - before doing anything else
    import sys
    log_level = getattr(logging, args.log_level, logging.INFO)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure fresh logging with forced console output
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    
    # # Test logging immediately
    # print("Testing console output...")
    # print(f"Logging level set to: {log_level}")
    # print(f"Number of handlers: {len(logging.root.handlers)}")
    logging.info("Logging system initialized successfully")
    sys.stdout.flush()  # Force flush

    # Load config file if provided
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

        # Merge with command-line args (CLI takes precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    logging.info("=" * 80)
    logging.info("NeSVoR2: MRI Super-Resolution Pipeline")
    logging.info("=" * 80)
    logging.info(f"Device: {args.device}")
    logging.info(f"Benchmark: {args.benchmark}")
    sys.stdout.flush()  # Force flush

    # Convert device string to torch.device
    if isinstance(args.device, str):
        args.device = torch.device(args.device)
        logging.info(f"Using device: {args.device}")

    # Add default values for model parameters
    if not hasattr(args, "n_features_z"):
        args.n_features_z = 8
    if not hasattr(args, "n_features_slice"):
        args.n_features_slice = 0
    if not hasattr(args, "n_levels_bias"):
        args.n_levels_bias = 0
    if not hasattr(args, "no_slice_scale"):
        args.no_slice_scale = False
    if not hasattr(args, "no_slice_variance"):
        args.no_slice_variance = False
    if not hasattr(args, "no_pixel_variance"):
        args.no_pixel_variance = False
    if not hasattr(args, "no_transformation_optimization"):
        args.no_transformation_optimization = False
    if not hasattr(args, "deformable"):
        args.deformable = False
    if not hasattr(args, "delta"):
        args.delta = 1.0
    if not hasattr(args, "img_reg_autodiff"):
        args.img_reg_autodiff = False
    if not hasattr(args, "n_samples"):
        args.n_samples = 8
    if not hasattr(args, "milestones"):
        args.milestones = [0.5, 0.75, 0.9]
    if not hasattr(args, "gamma"):
        args.gamma = 0.33
    if not hasattr(args, "debug"):
        args.debug = False
    if not hasattr(args, "dtype"):
        args.dtype = torch.float32 if args.single_precision else torch.float16

    # Run pipeline
    try:
        results = run_pipeline(args)
        logging.info("Pipeline completed successfully!")
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
