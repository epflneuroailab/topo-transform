DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _vjepa_ckpt(seed: int, layers: str = "14_18_22", prefix: str = "") -> str:
    return (
        f"{prefix}best_transformed_model_global_vjepa_{layers}"
        f"_single_neighbInf_kinetics400_lr1e-4_bs32_sd{seed}.pt"
    )


def _clip_ckpt(seed: int, layers: str = "14_18_22", prefix: str = "") -> str:
    return (
        f"{prefix}best_transformed_model_global_clip_{layers}"
        f"_single_neighbInf_kinetics400_lr1e-4_bs32_sd{seed}.pt"
    )


def _videomae_ckpt(seed: int, layers: str = "14_18_22", prefix: str = "") -> str:
    return (
        f"{prefix}best_transformed_model_global_videomae_{layers}"
        f"_single_neighbInf_kinetics400_lr1e-4_bs32_sd{seed}.pt"
    )


def _som_ckpt(seed: int, layers: str = "22") -> str:
    return f"som_vjepa_{layers}_single_1mm_kinetics400_bs32_sd{seed}.pt"


MODEL_CKPT = _vjepa_ckpt(45)

MODEL_CKPTS = [_vjepa_ckpt(seed) for seed in DEFAULT_SEEDS]
CLIP_CKPTS = [_clip_ckpt(seed) for seed in DEFAULT_SEEDS]
VIDEOMAE_CKPTS = [_videomae_ckpt(seed) for seed in DEFAULT_SEEDS]
SOM_CKPTS = [_som_ckpt(seed) for seed in DEFAULT_SEEDS]
UNOPTIMIZED_CKPTS = [_vjepa_ckpt(seed, prefix="unoptimized.") for seed in DEFAULT_SEEDS]
TDANN_CKPTS = [f"tdann.model_final_checkpoint_phase199_seed{seed}.torch" for seed in range(5)]
SWAPOPT_CKPTS = [f"swapopt_single_sheet_seed{seed}" for seed in range(5)]
SWAPOPT_ONELAYER_CKPTS = [f"swapopt_seed{seed}" for seed in range(5)]
ONELAYER_CKPTS = [_vjepa_ckpt(seed, layers="18") for seed in DEFAULT_SEEDS]

HUMAN_C = "#2E7D32"
MODEL_C = "#7C8DB0"
DEFAULT_C = "gray"

LOCALIZER_P_THRESHOLD = 1e-3
LOCALIZER_T_THRESHOLD = 8
LOCALIZER_BIOMOTION_T_THRESHOLD = 8
LOCALIZER_FLOW_T_THRESHOLD = 16

all_roi_colors = {
    "Faces_moving_localizer": ("moving-face", (0.91, 0.30, 0.24)),
    "Bodies_moving_localizer": ("moving-body", (0.20, 0.63, 0.55)),
    "Scenes_moving_localizer": ("moving-place", (0.95, 0.77, 0.06)),
    "Faces_static_localizer": ("static-face", (1.00, 0.80, 0.80)),
    "Bodies_static_localizer": ("static-body", (0.80, 1.00, 0.80)),
    "Scenes_static_localizer": ("static-place", (1.00, 0.90, 0.70)),
    "Faces_static": ("static-face", (0.75, 0.00, 0.00)),
    "Bodies_static": ("static-body", (0.00, 0.45, 0.00)),
    "Scenes_static": ("static-place", (0.80, 0.35, 0.00)),
    "Faces_moving": ("dynamic-face", (1.00, 0.80, 0.80)),
    "Bodies_moving": ("dynamic-body", (0.80, 1.00, 0.80)),
    "Scenes_moving": ("dynamic-place", (1.00, 0.90, 0.70)),
    "object": ("object", (0.20, 0.20, 0.80)),
    "V6": ("V6", (0.00, 0.78, 0.88)),
    "MT-Huk": ("MT", (0.90, 0.25, 0.65)),
    "pSTS": ("pSTS", (0.55, 0.45, 0.95)),
    "V6-enhanced": ("V6-enhanced", (0.00, 0.78, 0.88)),
    "pSTS-enhanced": ("pSTS-enhanced", (0.55, 0.45, 0.95)),
}

ROI_COLOR_ALIASES = {
    "face": "Faces_moving_localizer",
    "body": "Bodies_moving_localizer",
    "place": "Scenes_moving_localizer",
    "mt": "MT-Huk",
    "v6": "V6",
    "psts": "pSTS",
    "v6-enhanced": "V6-enhanced",
    "psts-enhanced": "pSTS-enhanced",
    "face-detailed": "Faces_moving_localizer",
    "body-detailed": "Bodies_moving_localizer",
    "place-detailed": "Scenes_moving_localizer",
    "car-detailed": "MT-Huk",
    "instrument-detailed": "V6",
}
all_roi_colors.update({alias: all_roi_colors[target] for alias, target in ROI_COLOR_ALIASES.items()})

roi_groups = {
    "face-response": ["Faces_static", "Faces_moving"],
    "body-response": ["Bodies_static", "Bodies_moving"],
    "place-response": ["Scenes_static", "Scenes_moving"],
    "face": ["Faces_static_localizer", "Faces_moving_localizer"],
    "body": ["Bodies_static_localizer", "Bodies_moving_localizer"],
    "place": ["Scenes_static_localizer", "Scenes_moving_localizer"],
    "motion": ["V6", "pSTS", "MT-Huk"],
    "motion2": ["V6", "pSTS"],
    "motion3": ["V6-enhanced", "pSTS-enhanced"],
    "motion4": ["MT-Huk", "V6-enhanced", "pSTS-enhanced"],
    "V6": ["V6"],
    "MT": ["MT-Huk"],
    "pSTS": ["pSTS"],
    "categorical": [
        "Faces_static_localizer",
        "Bodies_static_localizer",
        "Scenes_static_localizer",
        "Faces_moving_localizer",
        "Bodies_moving_localizer",
        "Scenes_moving_localizer",
    ],
    "categorical2": [
        "Faces_moving_localizer",
        "Bodies_moving_localizer",
        "Scenes_moving_localizer",
    ],
    "fLoc": ["face", "body", "place", "object"],
    "fLoc2": ["face", "body", "place"],
    "fLoc3": ["face-detailed", "car-detailed", "instrument-detailed"],
    "all": list(all_roi_colors.keys()),
}
