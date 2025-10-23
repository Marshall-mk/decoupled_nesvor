# NeSVoR2: Neural Slice-to-Volume Reconstruction

MRI super-resolution reconstruction using Implicit Neural Representations (INR) for fetal brain imaging.

## Installation

### 1. Install Dependencies

```bash
# Install PyTorch (choose the appropriate CUDA version)
# For CPU only:
pip install torch torchvision

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
cd /Users/cleancoder/Downloads/PhD/code
pip install -r NeSVoR2/requirements.txt
```

### 2. Verify Installation

```bash
cd /Users/cleancoder/Downloads/PhD/code
python -m NeSVoR2.main --help
```

## Usage

### Running as a Python Module (Recommended)

The code must be run as a module from the parent directory to handle relative imports correctly:

```bash
cd /Users/cleancoder/Downloads/PhD/code

python -m NeSVoR2.main \
    --input-stacks ../../data/simpleData/sub4/T2WCorBHSENSEs2801a1028.nii.gz \
                    ../../data/simpleData/sub4/T2WsagBHSENSEs3001a1030.nii.gz \
                    ../../data/simpleData/sub4/T2WtraBHSENSEs2901a1029.nii.gz \
    --stack-masks ../../data/simpleData/sub4/T2WCorBHSENSEs2801a1028_mask.nii.gz \
                   ../../data/simpleData/sub4/T2WsagBHSENSEs3001a1030_mask.nii.gz \
                   ../../data/simpleData/sub4/T2WtraBHSENSEs2901a1029_mask.nii.gz \
    --stacks-intersection \
    --thicknesses 5.0 5.0 5.0 \
    --output-volume ../../data/simpleData/sub4/output_nesvor/result.nii.gz \
    --output-model ../../data/simpleData/sub4/output_nesvor/model.pt \
    --output-slices ../../data/simpleData/sub4/output_nesvor_slices \
    --simulated-slices ../../data/simpleData/sub4/output_sim_nesvor_slices \
    --output-json ../../data/simpleData/sub4/output_nesvor.json \
    --benchmark \
    --benchmark-output ../../data/simpleData/sub4/benchmark.json \
    --device mps
```

### Key Arguments

**Input/Output:**
- `--input-stacks`: Input NIfTI files (multiple stacks)
- `--stack-masks`: Corresponding mask files
- `--thicknesses`: Slice thickness for each stack (mm)
- `--output-volume`: Output high-resolution volume
- `--output-model`: Save trained model
- `--benchmark`: Enable benchmarking (time, memory)

**Preprocessing:**
- `--stacks-intersection`: Create volume mask from stack intersection
- `--segmentation`: Run brain segmentation
- `--bias-field-correction`: Run N4 bias field correction

**Registration:**
- `--registration`: Enable motion correction
- `--svort`: Use SVoRT transformer-based registration (default: True)
- `--svort-version v2`: SVoRT version (v1 or v2)
- `--use-vvr`: Include traditional stack-to-stack registration

**Training:**
- `--n-iter 5000`: Number of training iterations
- `--batch-size 4096`: Training batch size
- `--output-resolution 0.8`: Output resolution (mm)

**Device:**
- `--device cuda`: Use CUDA GPU
- `--device cpu`: Use CPU
- `--device mps`: Use Apple Silicon GPU (macOS)

### Pipeline Phases

The pipeline consists of 8 phases (can be run selectively):

1. **Load Inputs**: Load stacks, masks, models
2. **Segmentation**: 2D fetal brain masking (optional)
3. **Bias Field Correction**: N4 algorithm (optional)
4. **Assessment**: Stack quality metrics
5. **Registration**: SVoRT motion correction (optional)
6. **Reconstruction**: NeSVoR training
7. **Sampling**: Generate high-resolution volume
8. **Save Outputs**: Write results to disk

## Project Structure

```
NeSVoR2/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore patterns
├── __init__.py            # Package initialization
│
├── model/                 # Neural network models
│   ├── models.py          # INR, NeSVoR architectures
│   ├── train.py           # Training loop
│   ├── sample.py          # Volume sampling
│   └── hash_grid_torch.py # Hash grid encoding
│
├── preprocess/            # Preprocessing modules
│   ├── brain_segmentation.py  # Brain masking
│   ├── bias_field.py      # N4 correction
│   ├── assessment.py      # Quality metrics
│   ├── svort/            # SVoRT registration
│   └── svr/              # Traditional SVR
│
├── data/                  # Data structures
│   ├── data.py           # PointDataset
│   └── slice_acq.py      # Slice acquisition model
│
└── utils/                 # Utility functions
    ├── data_utils.py     # I/O, transformations
    └── gen_utils.py      # General utilities
```

## Important Notes

### Running the Code

**✅ DO THIS:**
```bash
cd /Users/cleancoder/Downloads/PhD/code
python -m NeSVoR2.main [args]
```

**❌ DON'T DO THIS:**
```bash
cd /Users/cleancoder/Downloads/PhD/code/NeSVoR2
python main.py [args]  # Will fail with import errors!
```

The codebase uses relative imports (e.g., `from ..utils import ...`) which only work when run as a module from the parent directory.

### Device Selection

- **CUDA GPU** (`--device cuda`): Fastest, requires NVIDIA GPU
- **Apple Silicon GPU** (`--device mps`): Fast on M1/M2/M3 Macs
- **CPU** (`--device cpu`): Slowest, but works everywhere

### Optional Dependencies

- **SimpleITK**: Required for `--bias-field-correction`. Install: `pip install SimpleITK`
- **MONAI**: Optional for enhanced brain segmentation. Install: `pip install monai`

## Troubleshooting

### Import Errors

If you get `ImportError: attempted relative import beyond top-level package`:
- Make sure you're running from `/Users/cleancoder/Downloads/PhD/code` (parent directory)
- Use `python -m NeSVoR2.main` not `python main.py`

### Missing Dependencies

If you get `ModuleNotFoundError`:
```bash
pip install -r NeSVoR2/requirements.txt
```

### CUDA Out of Memory

Reduce batch size:
```bash
python -m NeSVoR2.main ... --batch-size 2048
```

## Citation

If you use this code, please cite:

```bibtex
@article{xu2023nesvor,
  title={NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI},
  author={Xu, Junshen and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2023}
}
```

## License

See LICENSE file for details.
