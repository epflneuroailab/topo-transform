# TopoTransform

Code and cached-data workflows for **A Topographic Video Model Predicts the
Spatio-Functional Organization of the Human Visual System**.

The project builds on [TDANN](https://github.com/neuroailab/TDANN) and uses
BrainScore/BrainIO integrations for several neural evaluations.

## Setup

Python 3.11 is recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchcodec tables opencv-python seaborn
python -m pip install brainscore-core brainio nilearn
python -m pip install git+https://github.com/YingtianDt/vision.git
python -m pip install git+https://github.com/YingtianDt/neuroparc.git
```

Runtime paths default to directories under `cache/`; see `config.py`. They can
be overridden with these environment variables:

```text
RESULTCACHING_HOME
BRAINIO_HOME
BRAINSCORE_HOME
TORCH_HOME
HF_HOME
MMAP_HOME
```

The main cache directories are:

```text
cache/checkpoints   model checkpoints
cache/debug         cached analysis values
cache/plots         standard plot outputs
cache/positions     cortical-sheet positions
cache/brainio2      neural datasets
```

## Precomputed Cache

Most figure scripts read `cache/debug` directly. The current archive download
is approximately 28.5 GB; extraction requires additional free space. Download
and extract it with:

```bash
mkdir -p cache
curl -fL \
  https://epfl-neuroailab-public.s3.amazonaws.com/david/topo-transform/debug.zip \
  -o cache/debug.zip
bash download_cache.sh
```

`download_cache.sh` uses `7zz`, `7z`, `7za`, `unzip`, or `bsdtar`, in that
order, and extracts the archive to `cache/debug`.

To retrieve the complete public prefix instead, install and configure the AWS
CLI, then run:

```bash
bash download_cache.sh --full
```

Public files can also be browsed at
[epfl-neuroailab-public](https://epfl-neuroailab-public.s3.amazonaws.com/?prefix=david/topo-transform/).

## Reproduce Figures

The main entry point is `scripts/reproduce.py`. Cached reproduction does not
retrain the models.

Run the core analysis plots:

```bash
python -m scripts.reproduce \
  --stage plots \
  --plot_group core \
  --fail_fast
```

Run every cached plot group:

```bash
python -m scripts.reproduce \
  --stage plots \
  --plot_group all \
  --fail_fast
```

Standard outputs are written under `cache/plots`. The plot groups are:

| Group | Contents |
| --- | --- |
| `core` | Main evaluation plots |
| `slow` | Additional cache-intensive plots |
| `extra` | Extended and diagnostic analyses |
| `publication` | Publication-specific main and supplementary panels |
| `all` | All groups above |
| `none` | Only modules explicitly supplied with `--plots` |

Generate publication panels as editable SVG files with PNG previews:

```bash
python -m scripts.reproduce \
  --stage plots \
  --plot_group publication \
  --editable \
  --fail_fast
```

Editable outputs are written under `cache/publication_editable`. Text remains
SVG text and bar-chart rectangles are normalized to native SVG elements. This
workflow does not generate PDFs.

Run one existing plot module directly:

```bash
python -m scripts.plot_smoothness
```

Or run selected modules through the orchestrator:

```bash
python -m scripts.reproduce \
  --stage plots \
  --plot_group none \
  --plots scripts.plot_robert_distribution \
  --fail_fast
```

Use `--dry_run` to print the commands without executing them.

## Reproduce Source Data

Generate the currently automated Source Data worksheets from cached analysis
values:

```bash
python -m scripts.export_source_data --heavy
```

The workbook is written to `source_data/Source Data.xlsx`. The directory is
ignored by Git because the workbook is generated output.

`--heavy` includes Fig. 2b-d, which require feature loading and, for Fig. 2d,
model inference. Without `--heavy`, only the less expensive worksheets are
rebuilt. A subset can be selected by worksheet name, for example:

```bash
python -m scripts.export_source_data --only "Fig 3d left" "Fig S9"
```

The exporter currently generates these 21 worksheets:

```text
Fig 2b, Fig 2c, Fig 2d, Fig 2e
Fig 3d left, Fig 3d right, Fig 3e
Fig 4c, Fig 4d, Fig 4e
Fig S1, Fig S2, Fig S3, Fig S4a, Fig S4b, Fig S4c
Fig S6, Fig S8a, Fig S8b, Fig S9, Fig S10
```

## Train and Evaluate

Train the default TopoTransform V-JEPA model, evaluate the resulting checkpoint,
and run the core plots:

```bash
python -m scripts.reproduce \
  --stage all \
  --plot_group core \
  --seed 42 \
  --fail_fast
```

Inspect the expanded commands first when running in a new environment:

```bash
python -m scripts.reproduce \
  --stage all \
  --plot_group core \
  --seed 42 \
  --dry_run
```

Training can also be run directly:

```bash
python train.py \
  --data_name kinetics400 \
  --lr 0.0001 \
  --num_epochs 10 \
  --seed 42
```

Use `--resume_training` to continue from an existing checkpoint. Enable
`--use_wandb` only when Weights & Biases credentials are configured.

Run `python -m scripts.reproduce --help` for all training, evaluation, SOM, and
plotting options.
