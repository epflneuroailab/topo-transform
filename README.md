# TopoTransform

This repository provides the **source code** to reproduce the results
presented in our [paper](). It depends on the
[TDANN](https://github.com/neuroailab/TDANN) repository.

## Setup

Please use **Python 3.11** for this repository. Any operating system
should be compatible.

Install the required packages (this typically takes \~20 minutes):

``` bash
# install core dependencies
pip install numpy scipy pandas scikit-learn torch torchvision h5py matplotlib opencv-python
pip install brainscore-core brainio nilearn
pip install git+https://github.com/YingtianDt/vision.git
pip install git+https://github.com/YingtianDt/neuroparc.git
```

Set the following environment variables to directories on your machine
where intermediate results, stimuli, and model checkpoints will be
stored:

    RESULTCACHING_HOME
    BRAINIO_HOME
    BRAINSCORE_HOME
    TORCH_HOME
    HF_HOME
    MMAP_HOME

See `config.py` for example configurations.

## Reproducibility

All paths are rooted at the repository by default. The important cache
directories are:

    cache/checkpoints
    cache/debug
    cache/plots
    cache/positions
    cache/brainio2

The plotting scripts reuse `cache/debug` aggressively. If the cache is
present, most figures are regenerated without rerunning feature
extraction.

### One-command Plot Reproduction

The easiest cached reproduction path is:

```bash
python -m scripts.reproduce --stage plots --plot_group core
```

This runs the core figure scripts:

```text
scripts.plot_localizer_decode
scripts.plot_localizers
scripts.plot_localizers_hierarchy
scripts.plot_motion_object
scripts.plot_neural_alignment
scripts.plot_smoothness
scripts.plot_wiring_cost
scripts.plot_localizers_cortex
scripts.plot_hierarchy_alignment
```

Outputs are written to `cache/plots`. To see the commands without
executing them:

```bash
python -m scripts.reproduce --stage plots --plot_group core --dry_run
```

Slower or GPU-heavy plots are separated:

```bash
python -m scripts.reproduce --stage plots --plot_group slow
```

The slow group currently includes:

```text
scripts.plot_localizer_motion
scripts.plot_localizers_seed
scripts.plot_wiring_cost_fmri
```

`scripts.plot_wiring_cost_fmri` can require substantial GPU memory
during smoothing. Run it alone if the GPU is shared.

### Training To Plots

To train the default TopoTransform VJEPA model, evaluate fLoc, and then
run the core plot suite:

```bash
python -m scripts.reproduce --stage all --plot_group core --seed 42
```

This expands to:

```bash
python train.py --data_name kinetics400 --lr 0.0001 --num_epochs 10 --batch_size 32 --samples_per_batch 16384 --seed 42 --layer_indices 14 18 22
python eval.py --checkpoint_name best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt --dataset_names vpnl biomotion kanwisher pitzalis --batch_size 32 --fwhm_mm 1.0 --resolution_mm 1.0
python -m scripts.plot_...
```

Use `--resume_training` to continue from an existing checkpoint. Use
`--use_wandb` only when your WandB credentials are configured.

To train the CLIP TopoTransform variant, switch the backbone:

```bash
python -m scripts.reproduce --stage train --model_name clip --seed 42
```

This uses `openai/clip-vit-large-patch14`, treats videos as per-frame
image batches, restores a singleton time axis for each frame feature, and
then follows the same TopoTransform setup as VJEPA.

To train the VideoMAE TopoTransform variant:

```bash
python -m scripts.reproduce --stage train --model_name videomae --seed 42
```

This uses `MCG-NJU/videomae-large-finetuned-kinetics`, resamples each
input video to the model's configured number of frames, restores VideoMAE
patch tokens to `(B, T, C, H, W)`, and then follows the same
TopoTransform setup as VJEPA.

To train a lower-layer small-tissue variant, use the stage-based tissue
configuration. For layer `4`, this uses V1 tissue/neighborhood geometry
and defaults to `rf_overlap=0.1`:

```bash
python train.py --layer_indices 4 --tissue_config small --seed 42
```

To compare sensitivity to the training data subset, train deterministic
halves with the same split seed:

```bash
python train.py --train_split half_a --train_split_seed 42 --seed 42
python train.py --train_split half_b --train_split_seed 42 --seed 42
```

### SOM Baseline

The SOM baseline uses the same training videos, averages last-layer
VJEPA features over time and spatial positions, and maps one 1024-D
video-level vector onto a single 1 mm grid. For the default layer `22`
the grid is `70 x 70`, matching the TDANN-sized sheet rather than the
three-layer TopoTransform sheet.

Train one SOM checkpoint:

```bash
python train_som.py --data_name kinetics400 --batch_size 32 --seed 42 --layer_indices 22
```

For quick feasibility tests, downsample the training run:

```bash
python train_som.py --data_name kinetics400 --batch_size 32 --seed 42 --max_batches 20 --samples_per_batch 2048
```

The checkpoint is saved as
`cache/checkpoints/som_vjepa_22_single_1mm_kinetics400_bs32_sd42.pt`.
When this file exists, the shared plot registry automatically includes a
`SOM` method group. To train SOM before cached plot reproduction:

```bash
python -m scripts.reproduce --stage plots --plot_group core --train_som --seed 42
```

### Individual Commands

Training only:

```bash
python train.py --data_name kinetics400 --lr 0.0001 --num_epochs 10 --seed 42
```

Evaluate a checkpoint:

```bash
python eval.py --checkpoint_name best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt
```

Run one plot:

```bash
python -m scripts.plot_smoothness
```

Run an extra plot module through the orchestrator:

```bash
python -m scripts.reproduce --stage plots --plot_group none --plots scripts.plot_robert_distribution
```

### Notes

Some scripts are exploratory or intentionally legacy/debug-oriented and
are not part of the main reproduction path:

```text
scripts.plot_smoothness_legacy
scripts.plot_smoothness_legacy2
scripts.test
scripts.test_autocorr
```
