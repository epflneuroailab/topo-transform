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

To reproduce the cached VJEPA layer-to-Glasser-ROI decoding summary
from `brainscore_analysis`:

```bash
python -m scripts.plot_vjepa_layer_roi_variance
```

This plots VJEPA layer index against Pearson R alignment for V1, V4,
and FFC using the cached `test_vjepa_layers_decoding/full` scores. The
script still uses cached joint ceilings to match the decoded voxel axis.

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


### Individual Commands

Training only:

```bash
python train.py --data_name kinetics400 --lr 0.0001 --num_epochs 10 --seed 42
```

Run one plot:

```bash
python -m scripts.plot_smoothness
```

Run an extra plot module through the orchestrator:

```bash
python -m scripts.reproduce --stage plots --plot_group none --plots scripts.plot_robert_distribution
```