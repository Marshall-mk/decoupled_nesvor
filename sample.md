Great — here's a concrete, end-to-end fetal-brain example (commands + what happens at each stage, with realistic numbers, tensor datatypes/shapes and the exact variables used in the repo). I walk through the pipeline you gave: load inputs -> segmentation -> bias_field_correction -> assessment -> registration -> reconstruction (train & sample). I use concrete example inputs (3 fetal stacks) and show the in-memory objects the code creates/uses and the files written.

Example CLI (fetal brain)
- Command (single reconstruct run that does segmentation, N4 and registration then trains NeSVoR and samples):
  nesvor reconstruct \
    --input-stacks stack1.nii.gz stack2.nii.gz stack3.nii.gz \
    --stack-masks mask1.nii.gz mask2.nii.gz mask3.nii.gz \
    --thicknesses 3.0 3.0 3.0 \
    --segmentation \
    --bias-field-correction \
    --registration svort \
    --output-volume fetal_brain.nii.gz \
    --output-resolution 0.8 \
    --output-model fetal_model.pt \
    --simulated-slices simulated_slices_folder \
    --n-samples 128 \
    --n-iter 5000 \
    --batch-size 4096

(You can change numbers like --output-resolution or --n-iter as needed.)

Concrete example inputs (assumptions)
- 3 input stacks (NIfTI)
  - stack1.nii.gz, stack2.nii.gz, stack3.nii.gz
  - each stack: 20 slices
  - raw per-slice image size: 256 x 256 (H x W)
  - original in-plane resolution: 1.25 mm (resolution_x = resolution_y = 1.25)
  - slice thickness: 3.0 mm (thickness = gap = 3.0)
  - intensities: typical MRI intensities (read as float)
  - stack masks provided: mask1.nii.gz etc (same dims/resolution as the stacks)

Stage-by-stage — inputs, types, shapes, outputs, code variables

1) Load inputs
- Code entry: nesvor/cli/io.py -> inputs(args)
- On-disk -> in-memory:
  - load_stack(...) produces a Stack for each file:
    - Stack.slices: torch.Tensor, dtype torch.float32, shape [N, 1, H, W]
      - Example: for each stack: shape [20, 1, 256, 256]
    - Stack.mask: torch.Tensor (bool), shape [20, 1, 256, 256]  (loaded from mask files)
    - Stack.transformation: RigidTransform with length 20 (one transform per slice; initialized from affine/header)
    - Stack.resolution_x = 1.25 (float), resolution_y = 1.25, thickness = 3.0, gap = 3.0
    - Stack.name = str(path)
  - These tensors are moved to args.device (CPU or CUDA)
- CLI-level result:
  - input_dict["input_stacks"] = [Stack1, Stack2, Stack3]
  - input_dict["volume_mask"] possibly set if user provided or stacks_intersection used
- Typical memory usage: 3 stacks × 20 slices × 256 × 256 × 4 bytes ≈ ~15 MB per stack image tensor (float32) (plus masks & transforms).

2) Segmentation (2D fetal brain masking)
- Trigger: --segmentation
- Code: nesvor/preprocessing/masking/brain_segmentation.py (build_monaifbs_net & batch_infer)
- Input:
  - Each Stack.slices: torch.Tensor [20,1,256,256] on device
- Operation:
  - Resampling internally to the segmentation network resolution (the wrapper uses RESOLUTION = 0.8 mm for inference; patches extracted and batched).
  - Batched forward passes through pre-trained MONAIfbs DynUNet (torch model) generate per-slice probability maps or logits.
- Output:
  - For each stack: updated Stack.mask: torch.Tensor (bool), shape [20,1,H_seg, W_seg]  
    - The code will convert and then place the mask back into the Stack (typically same H/W as original stack after restoring).
  - In-memory type: torch.bool (mask).
- Saved (if invoked by segment-stack cmd or outputs flags):
  - NIfTI files mask1.nii.gz, mask2.nii.gz, mask3.nii.gz (via outputs())
- Downstream uses:
  - Registration and bias correction will use masks to limit background and cropping.

3) Bias field correction (N4)
- Trigger: --bias-field-correction
- Code: nesvor/preprocessing/bias_field.py (n4_bias_field_correction wrapper using SimpleITK)
- Input:
  - Stack.slices [20,1,256,256], Stack.mask [20,1,256,256]
  - These are converted to numpy and passed to SimpleITK's N4 algorithm.
- Operation:
  - N4 estimates multiplicative low-frequency field and corrects intensities.
- Output:
  - Stack.slices replaced by corrected images (same shape [20,1,256,256] and dtype restored to torch.float32).
- Saved (if CLI flag for corrected stacks):
  - corrected_stack1.nii.gz etc.
- Notes: shape and metadata remain the same (resolution, transforms do not change).

4) Assessment (stack quality / motion assessment)
- Trigger: (--metric != "none") or explicit nesvor assess
- Code: nesvor/preprocessing/assessment.py (assess)
- Input:
  - List[Stack] with slices & masks
- Operation:
  - Computes per-stack metrics (e.g., NCC-based metrics, intensity statistics, motion score).
- Output:
  - Python dict (results) containing per-stack numeric scores (serializable to JSON)
  - In CLI flow:
    - input_dict["input_stacks"], results = _assess(...)
    - If requested: writes results JSON (--output-json)
- Types: numeric scalars (floats/ints), lists; all in Python objects (not tensors unless stored temporarily)

5) Registration (motion correction) — using SVoRT (rigid) for fetal brain
- Trigger: --registration svort (common for fetal)
- Code paths:
  - nesvor/svort/inference.py -> parse_data(...) -> svort_predict(...) -> correct_svort(...) -> get_transforms_full(...) -> _register(...) called from commands.py
- parse_data behavior (important datatype/shape changes):
  - Resampling for SVoRT:
    - res_s (resampled slice res) = 1.0 mm
    - res_r (recon resolution used for registration / template) = 0.8 mm
  - resample slices for registration:
    - slices = resample(data.slices * data.mask, (data.resolution_x, data.resolution_y), (res_s, res_s))
    - If original resolution was 1.25, resample to res_s=1.0
    - Example: each slice becomes roughly same pixel dimensions but resampled grid; code then crops to ROI
  - Cropping & padding to SVoRT input:
    - The wrapper finds bounding box, pads, then crops to 128x128 patches:
      slices = slices[:, :, i-64:i+64, j-64:j+64]
    - So for SVoRT input slices shape becomes [N_nonzero_slices, 1, 128, 128]
      - Example: each stack had 20 slices, maybe after masking 18 nonzero; for each stack slices shape could be [18, 1, 128, 128]
  - Stacks for SVoRT input are normalized: stacks.append(slices / torch.quantile(slices[slices > 0], 0.99))
- parse_data returns (among others):
  - stacks_svort_in: List[Stack]  (resampled, cropped, normalized) — each Stack.slices shape [N',1,128,128]
  - stacks_resampled: List[Stack] (resampled but not cropped)
  - stacks_resampled_reset: List[Stack] (reset transforms)
  - crop_idx (per-stack boolean index for z slices kept)
  - volume: Volume.zeros((200,200,200), res_r=0.8) — template volume used by SVoRT (in-memory Volume.image dtype torch.float32 of shape [200,200,200] initially zeros)
  - psf: torch.Tensor representing point-spread-function (get_PSF)
- SVoRT model (svort_predict) takes data dict:
  - "slices": torch.cat([s.slices for s in stacks], dim=0) => tensor shape [total_slices, 1, 128,128]
  - "transforms": RigidTransform.cat([s.transformation for s in stacks]).matrix() => transform matrices for each slice
  - returns corrected transforms and volume prediction
- After SVoRT:
  - stacks_out: List[Stack] where each returned Stack is a clone of input stack but with updated Stack.transformation (per-slice RigidTransform) — transformations are updated to the new motion-corrected transforms
- correct_svort(...) and get_transforms_full(...):
  - They compare stack-based transforms vs SVoRT transforms and optionally replace per-slice transforms where beneficial (NCC-based decision).
  - They construct stacks suitable to write motion-corrected slices (or to pass to reconstruction).
- Final registration output used by reconstruct:
  - input_dict["input_slices"] = _register(self.args, input_dict["input_stacks"])
    - Effectively produces a set of motion-corrected slices (List[Slice]) or a concatenated Stack (depending on implementation).
    - Typical returned slices shapes: slices are motion-corrected and kept at resampled sizes (commonly [1,H_reg,W_reg] per Slice); for reconstruction the pipeline will resample to match dataset resolution. For SVoRT path, slices were cropped to 128x128 for SVoRT but reconstruct uses the stack_resampled_reset variants to reconstruct at larger field — train() will build a PointDataset from the returned slices.
  - Code variable in CLI: input_dict["input_slices"] is passed to train(...).

Files that can be written at this stage
- Per-slice NIfTI output if --output-slices given (save_slices writes per-slice NIfTI).
- Optional corrected stacks saved by CLI if requested.

6) Reconstruction (NeSVoR training)
- Code: nesvor/inr/train.py -> train(input_dict["input_slices"], args)
- Input to train:
  - slices: List[Slice] (motion-corrected) — each Slice.image is torch.Tensor [H, W] (2D) where H/W depend on how registration output is returned, often resampled to consistent spacing before training. For this fetal example, reconstruct sampling is set to output-resolution 0.8 mm; the dataset inside train will compute bounding box, dataset.resolution etc.
  - args: Namespace of training hyper-parameters (n_iter=5000, batch_size=4096 etc.)
- Internals and datatypes:
  - dataset = PointDataset(slices) -> dataset contains:
    - xyz (torch.Tensor) : point coordinates sampled from all slices, shape [V, 3] where V = number of sampled points (depends on sampling strategy and masks)
    - dataset.v: volume sampling tensor with number of voxels/points (dataset.v.numel() used to compute epochs)
    - dataset.mask: Volume object representing mask estimated from slices
    - dataset.resolution : torch.tensor or float representing sampling resolution
    - mean, bounding_box (bb), transformation, etc. (all torch tensors)
  - model = NeSVoR(...) which produces model.inr (an INR) — torch.nn.Module with parameters (float32 or fp16 depending on args)
  - Training loop uses torch.autocast and GradScaler if mixed precision enabled.
  - At each training step:
    - batch = dataset.get_batch(args.batch_size, args.device)
    - model(**batch) returns dict of losses (D_LOSS, S_LOSS, regularization terms)
    - scaler.scale(loss).backward(), scaler.step(optimizer), scaler.update()
- Example sizes (approximate):
  - Suppose the bounding box leads to ~ (150 x 150 x 120) voxels at output resolution 0.8 mm -> mask region voxels ~2.7M. But train uses point-sampling and args.batch_size tunes how many points per iteration (e.g., batch_size=4096).
  - dataset.xyz shape could be [N_points_total, 3] where N_points_total is potentially millions before batching.
- Outputs from train():
  - Returns (model.inr, output_slices, mask)
    - model.inr : INR instance (torch.nn.Module); this is the continuous neural implicit representation of the 3D volume
    - output_slices : List[Slice] — updated slices with final optimized transformations applied (each output_slice.transformation = transformation[i] inside train)
    - mask : Volume object (torch.Tensor image and mask) used by training and saved in output model
  - CLI then calls _sample_inr(...) to sample a discretized Volume and simulated slices.

7) Sampling INR (sample volume and simulated slices)
- Code: nesvor/inr/sample.py (sample_volume, sample_slices) and CLI wrapper _sample_inr in commands.py
- Inputs:
  - model (INR), mask (Volume), output_slices (List[Slice]), sampling args (output_resolution=0.8, output_psf_factor, inference_batch_size, n_inference_samples)
- Outputs:
  - output_volume: Volume object (3D) with:
    - Volume.image: torch.Tensor, shape [D, H, W], dtype torch.float32
      - Example: if bounding box -> D x H x W = 150 x 150 x 120, Volume.image shape [150,150,120]
    - Volume.mask: torch.bool same shape
    - Volume.resolution_x/y/z = 0.8 mm
    - Volume.transformation = RigidTransform (world orientation)
  - simulated_slices: List[Slice] or Stack of slices simulated by sampling the INR at the slices' transformed coordinates — shapes match the slices they are simulating (e.g., [1,H_slice,W_slice] per simulated slice)
- CLI outputs:
  - outputs() saves:
    - output_volume -> saved as fetal_brain.nii.gz (NIfTI)
    - output_model -> fetal_model.pt (saved as torch.save({"model": state_dict, "mask": mask, "args": args}))
    - output_slices and simulated_slices -> saved into specified folders as per save_slices
  - Example file artifacts:
    - fetal_brain.nii.gz (3D reconstruction)
    - fetal_model.pt (PyTorch state dict of INR + mask + args)
    - simulated_slices_folder/0.nii.gz, mask_0.nii.gz, 1.nii.gz, …
    - optional registration outputs if specified earlier (--output-slices or --output-corrected-stacks)

Recap / mapping to repo variables and types
- After loading: input_dict["input_stacks"] -> List[Stack]; Stack.slices: torch.Tensor [N,1,H,W] (float32)
- After segmentation: same List[Stack], but Stack.mask updated to torch.bool [N,1,H,W]
- After bias correction: same List[Stack], but Stack.slices corrected (same shapes)
- After assessment: input_dict["input_stacks"] may be unchanged but you have an assessments dict (Python) you can save
- After registration:
  - input_dict["input_slices"] set by _register(...) — a List[Slice] (each Slice.image type torch.Tensor [H_slice, W_slice])
  - Note: for SVoRT internals SVoRT works on resampled/cropped stacks with tensors shaped [n_slices_total, 1, 128, 128]
- Training:
  - train(input_dict["input_slices"], args) -> returns (INR model, output_slices (List[Slice]), mask (Volume))
  - model: torch.nn.Module (INR) -> saved as .pt via outputs()
- Sampling:
  - sample_volume or sample_slices produce:
    - output_volume: Volume (image torch.Tensor [D,H,W]), saved as NIfTI
    - simulated_slices: List[Slice], saved as NIfTI per-slice (and masks)

Short concrete numeric example (one full run, numbers chained)
- Inputs:
  - 3 stacks, each [20,1,256,256], res=1.25mm, thickness=3.0mm
- Segmentation:
  - masks shape remain [20,1,256,256], dtype torch.bool
- Bias-correction:
  - slices remain [20,1,256,256], dtype torch.float32
- parse_data for SVoRT:
  - each stack slices -> resampled to res_s=1.0mm
  - crop/pad to 128x128 -> SVoRT sees per-stack slices [N_nonzero, 1, 128, 128], e.g., [18,1,128,128]
  - total_slices for SVoRT = sum over stacks N_nonzero e.g., 50
- SVoRT outputs:
  - stacks_out: per-stack Stack clone with updated Stack.transformation (RigidTransform per slice)
  - stacks_svort_full: full-size stacks with transforms updated to reflect SVoRT corrections
- Reconstruct preparation:
  - input_dict["input_slices"] receives the motion-corrected slices (List[Slice]) — each Slice.image typically still small (depending on resampling) but train will integrate coordinates and bounding box
- Training:
  - PointDataset uses these slices -> dataset.xyz may be millions of points (sampled)
  - train runs args.n_iter steps (e.g., 5000) with batch_size 4096
  - returns model.inr (INR), output_slices (len = number of slices), mask (Volume)
- Sampling:
  - Sampling at output_resolution 0.8mm, bounding box yields example Volume.image shape around [150,150,120] (D,H,W) as torch.float32
  - File outputs: fetal_brain.nii.gz, fetal_model.pt, simulated_slices_folder/...

If you’d like, I can:
- Produce a concrete small toy run (with exact tensor shapes at every intermediate variable name used in the code, e.g., exact shapes for stacks_svort_in, stacks_resampled_reset, dataset.xyz, etc.) using the example numbers above.
- Show the minimal Python snippet (using the library functions) that loads three example NIfTIs, runs the same preprocessing steps programmatically (instead of CLI), and prints the shapes/types of the major variables at each step. (This would be a runnable snippet you can drop into a script in the repo environment.)

Which follow-up would you prefer — (A) toy runnable script showing the shapes and prints, or (B) a JSON-like summary table of every variable and its shape/dtype used in the fetal example?