import os
from pathlib import Path

DEBUG = True
RERUN = False
YASH = False

HOME_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    env_path = HOME_DIR / ".env"
    if not env_path.exists():
        return
    from dotenv import load_dotenv

    load_dotenv(env_path, override=True)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _set_default_env(name: str, value: Path | str) -> str:
    return os.environ.setdefault(name, str(value))


_load_dotenv()

CACHE_DIR = _ensure_dir(HOME_DIR / "cache")
DEBUG_DIR = _ensure_dir(CACHE_DIR / "debug")
PLOTS_DIR = _ensure_dir(CACHE_DIR / "plots")
POSITION_DIR = _ensure_dir(CACHE_DIR / "positions")

RESULTCACHING_HOME = _set_default_env("RESULTCACHING_HOME", CACHE_DIR / "resultcaching")
MMAP_HOME = _set_default_env("MMAP_HOME", CACHE_DIR / "mmap")
BRAINIO_HOME = _set_default_env("BRAINIO_HOME", CACHE_DIR / "brainio2")
BRAINSCORE_HOME = _set_default_env("BRAINSCORE_HOME", CACHE_DIR / "brain-score")
TORCH_HOME = _set_default_env("TORCH_HOME", CACHE_DIR / "torch")
HF_HOME = _set_default_env("HF_HOME", CACHE_DIR / "hf")

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("RESULTCACHING_DISABLE", "0")

ROOT_KINETICS400 = "/mnt/scratch/fkolly/datasets/kinetics-dataset/k400"
ROOT_IMAGENETVID = "/mnt/scratch/akgokce/datasets/imagenet"
ROOT_AFD101 = "/mnt/scratch/fkolly/datasets/AFD101"
ROOT_SSV2 = "/mnt/scratch/fkolly/datasets/smthsmthv2"
ROOT_AFRAZ2006 = "/mnt/scratch/ytang/datasets/afraz2006"
VPNL = "/mnt/scratch/ytang/datasets/fLoc_stimuli"
KANWISHER = "/mnt/scratch/ytang/datasets/lahner/stimulus_set/stimuli/localizer"
FLOW = "/mnt/scratch/ytang/datasets/flow_fields"
BIOLOGICAL_MOTION = "/mnt/scratch/ytang/datasets/biological-motion"
ROBERT_STATS = "/mnt/scratch/ytang/datasets/fsaverage_surfaces_robert"
ROBERT = "/mnt/scratch/ytang/datasets/robert2023"

PRETRAINED_DIR = "/mnt/scratch/fkolly/brainmo/pretrained"

if YASH:
    ROOT_KINETICS400 = "/data2/ynshah/Kinetics400/k400"
    ROOT_IMAGENETVID = "/data2/ynshah/imagenet-vid"
    PRETRAINED_DIR = "/data2/ynshah/tdann-transform/cache/checkpoints"
    VPNL = "/ccn2/u/ynshah/tdann-transform/cache/datasets/fLoc_stimuli"
    KANWISHER = "/ccn2/u/ynshah/spacetimetorch/datasets/lahner"
    FLOW = "/ccn2/u/ynshah/spacetimetorch/datasets/flow_fields"
    BIOLOGICAL_MOTION = "/ccn2/u/ynshah/spacetimetorch/datasets/biological-motion"
    from spacetorch.paths import POSITION_DIR


__all__ = [
    "BIOLOGICAL_MOTION",
    "BRAINIO_HOME",
    "BRAINSCORE_HOME",
    "CACHE_DIR",
    "DEBUG",
    "DEBUG_DIR",
    "FLOW",
    "HF_HOME",
    "HOME_DIR",
    "KANWISHER",
    "MMAP_HOME",
    "POSITION_DIR",
    "PLOTS_DIR",
    "PRETRAINED_DIR",
    "RERUN",
    "RESULTCACHING_HOME",
    "ROBERT",
    "ROBERT_STATS",
    "ROOT_AFD101",
    "ROOT_AFRAZ2006",
    "ROOT_IMAGENETVID",
    "ROOT_KINETICS400",
    "ROOT_SSV2",
    "TORCH_HOME",
    "VPNL",
    "YASH",
]
