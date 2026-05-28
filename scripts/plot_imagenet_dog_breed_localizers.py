import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from config import PLOTS_DIR
from models import clip_transform
from models import llcnn_transform
from models import vit_transform
from scripts.common import LOCALIZER_P_THRESHOLD
from scripts.common import LOCALIZER_T_THRESHOLD
from scripts.common import MODEL_CKPT
from scripts.get_localizers import localizers
from scripts.plot_localizers import _infer_tile_size
from scripts.plot_utils import ensure_dir
from scripts.plot_utils import savefig
from scripts.plot_utils import to_numpy
from validate import load_transformed_model
from validate.correction import fwe
from validate.floc.categories import functional_localization_one_vs_rest


DEFAULT_IMAGENET_ROOT = Path("/mnt/scratch/akgokce/datasets/imagenet/imagenet_val/train")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DOG_BREEDS = [
    ("n02085620", "Chihuahua"),
    ("n02085936", "Maltese"),
    ("n02099601", "Golden retriever"),
    ("n02099712", "Labrador retriever"),
    ("n02106662", "German shepherd"),
]
DOG_COLORS = {
    "Chihuahua": "#D55E00",
    "Maltese": "#E69F00",
    "Golden retriever": "#009E73",
    "Labrador retriever": "#0072B2",
    "German shepherd": "#CC79A7",
}
ANIMAL_CLASSES = [
    ("n02123045", "Tabby cat"),
    ("n02129165", "Lion"),
    ("n02391049", "Zebra"),
    ("n02504458", "African elephant"),
    ("n02481823", "Chimpanzee"),
]
ANIMAL_COLORS = {
    "Tabby cat": "#D55E00",
    "Lion": "#E69F00",
    "Zebra": "#009E73",
    "African elephant": "#0072B2",
    "Chimpanzee": "#CC79A7",
}
PRESETS = {
    "dog_breed": {
        "classes": DOG_BREEDS,
        "colors": DOG_COLORS,
        "label": "dog breeds",
        "file_prefix": "imagenet_dog_breed",
        "name_suffix": "imagenet_dog_breed",
    },
    "animal": {
        "classes": ANIMAL_CLASSES,
        "colors": ANIMAL_COLORS,
        "label": "animals",
        "file_prefix": "imagenet_animal",
        "name_suffix": "imagenet_animal",
    },
}
FLOC_CLASSES = [
    ("face", "Face"),
    ("body", "Body"),
    ("place", "Place"),
    ("object", "Object"),
]


def _get_input_transform(model):
    if model.__class__.__name__ == "TopoTransformedLLCNN":
        return llcnn_transform
    if model.__class__.__name__ == "TopoTransformedCLIP":
        return clip_transform
    return vit_transform


def _get_layer_positions(model):
    if model.smoothing:
        return [lp.coordinates.cpu() for lp in model.smoothed_layer_positions]
    return [lp.coordinates.cpu() for lp in model.layer_positions]


def _rects_for_positions(positions, tile_size):
    dx, dy = tile_size
    return [
        Rectangle((x - dx / 2, y - dy / 2), dx, dy)
        for x, y in positions
    ]


def _style_sheet_axis(ax, positions, tile_size):
    dx, dy = tile_size
    ax.set_aspect("equal", "box")
    ax.set_xlim(float(np.min(positions[:, 0])) - dx / 2, float(np.max(positions[:, 0])) + dx / 2)
    ax.set_ylim(float(np.min(positions[:, 1])) - dy / 2, float(np.max(positions[:, 1])) + dy / 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def collect_dog_breed_dataset(
    imagenet_root,
    classes=DOG_BREEDS,
    images_per_breed=50,
    seed=0,
):
    rng = np.random.default_rng(seed)
    datasets = {}
    for wnid, breed_name in classes:
        breed_dir = Path(imagenet_root) / wnid
        if not breed_dir.exists():
            raise FileNotFoundError(f"Missing ImageNet breed directory: {breed_dir}")
        paths = sorted(
            path
            for path in breed_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if len(paths) < images_per_breed:
            raise ValueError(
                f"{breed_name} has only {len(paths)} images in {breed_dir}; "
                f"need {images_per_breed}."
            )
        if len(paths) > images_per_breed:
            selected = sorted(rng.choice(paths, size=images_per_breed, replace=False))
        else:
            selected = paths
        datasets[breed_name] = [(str(path), breed_name) for path in selected]
    return datasets


def _significant_mask(t_vals, p_vals, p_threshold, t_threshold):
    return (np.asarray(p_vals).reshape(-1) < p_threshold) & (
        np.asarray(t_vals).reshape(-1) > t_threshold
    )


def _correct_p_values(p_vals_dict):
    corrected = {}
    for key, p_vals in p_vals_dict.items():
        shapes = [np.asarray(p_val).shape for p_val in p_vals]
        corrected[key] = [
            fwe(p_val).reshape(shape)
            for p_val, shape in zip(p_vals, shapes)
        ]
    return corrected


def plot_dog_breed_overlay(
    t_vals_dict,
    p_vals_dict,
    layer_positions,
    out_dir,
    classes=DOG_BREEDS,
    colors=DOG_COLORS,
    file_prefix="imagenet_dog_breed",
    p_threshold=LOCALIZER_P_THRESHOLD,
    t_threshold=LOCALIZER_T_THRESHOLD,
    dpi=300,
):
    out_dir = ensure_dir(out_dir)
    breed_names = [name for _, name in classes if name in t_vals_dict]
    breed_names += sorted(set(t_vals_dict) - set(breed_names))
    positions_by_layer = [to_numpy(pos) for pos in layer_positions]
    tile_sizes = [_infer_tile_size(pos) for pos in positions_by_layer]

    fig, axes = plt.subplots(
        1,
        len(layer_positions),
        figsize=(3.2 * len(layer_positions), 3.4),
        constrained_layout=True,
    )
    if len(layer_positions) == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        positions = positions_by_layer[layer_idx]
        tile_size = tile_sizes[layer_idx]
        ax.add_collection(
            PatchCollection(
                _rects_for_positions(positions, tile_size),
                facecolor="#EFEFEF",
                edgecolor="none",
                alpha=0.8,
                rasterized=True,
            )
        )
        for breed_name in breed_names:
            mask = _significant_mask(
                t_vals_dict[breed_name][layer_idx],
                p_vals_dict[breed_name][layer_idx],
                p_threshold,
                t_threshold,
            )
            if not np.any(mask):
                continue
            ax.add_collection(
                PatchCollection(
                    _rects_for_positions(positions[mask], tile_size),
                    facecolor=colors.get(breed_name, "#666666"),
                    edgecolor="none",
                    alpha=0.95,
                    rasterized=True,
                )
            )
        _style_sheet_axis(ax, positions, tile_size)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=9)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=colors.get(breed_name, "#666666"),
            markeredgecolor="none",
            markersize=7,
        )
        for breed_name in breed_names
    ]
    axes[0].legend(
        handles,
        breed_names,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(-0.55, 0.5),
        borderaxespad=0,
    )
    path = out_dir / f"{file_prefix}_overlay.svg"
    savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay: {path}")


def plot_dog_breed_tvals(
    t_vals_dict,
    layer_positions,
    out_dir,
    classes=DOG_BREEDS,
    file_prefix="imagenet_dog_breed",
    dpi=300,
):
    out_dir = ensure_dir(out_dir)
    breed_names = [name for _, name in classes if name in t_vals_dict]
    breed_names += sorted(set(t_vals_dict) - set(breed_names))
    positions_by_layer = [to_numpy(pos) for pos in layer_positions]
    tile_sizes = [_infer_tile_size(pos) for pos in positions_by_layer]
    vmax = max(
        float(np.nanmax(np.abs(to_numpy(t_vals_dict[breed_name][layer_idx]))))
        for breed_name in breed_names
        for layer_idx in range(len(layer_positions))
    )
    norm = Normalize(vmin=-vmax, vmax=vmax)
    tick_limit = int(np.floor(vmax))
    t_ticks = [-tick_limit, 0, tick_limit] if tick_limit > 0 else [-vmax, 0, vmax]

    for breed_name in breed_names:
        fig, axes = plt.subplots(
            1,
            len(layer_positions),
            figsize=(3.1 * len(layer_positions), 3.1),
            constrained_layout=True,
        )
        if len(layer_positions) == 1:
            axes = [axes]
        collection = None
        for layer_idx, ax in enumerate(axes):
            positions = positions_by_layer[layer_idx]
            tile_size = tile_sizes[layer_idx]
            vals = to_numpy(t_vals_dict[breed_name][layer_idx]).reshape(-1)
            collection = PatchCollection(
                _rects_for_positions(positions, tile_size),
                array=vals,
                cmap="RdBu_r",
                norm=norm,
                edgecolor="none",
                rasterized=True,
            )
            ax.add_collection(collection)
            _style_sheet_axis(ax, positions, tile_size)
            ax.set_title(f"Layer {layer_idx + 1}", fontsize=9)
        fig.suptitle(f"{breed_name} t-values", fontsize=11, fontweight="bold")
        cbar = fig.colorbar(
            collection,
            ax=axes,
            orientation="horizontal",
            shrink=0.78,
            pad=0.06,
            ticks=t_ticks,
            label="t-value",
        )
        cbar.ax.set_xticklabels([f"{tick:g}" for tick in t_ticks])
        safe_name = breed_name.lower().replace(" ", "_")
        path = out_dir / f"{file_prefix}_tvals_{safe_name}.svg"
        savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved t-value map: {path}")

    print(f"Shared t-value color scale: [{-vmax:.3f}, {vmax:.3f}]")


def plot_dog_breed_pvals(
    p_vals_dict,
    layer_positions,
    out_dir,
    classes=DOG_BREEDS,
    file_prefix="imagenet_dog_breed",
    plot_label=None,
    p_threshold=LOCALIZER_P_THRESHOLD,
    p_colorbar_min=1e-12,
    dpi=300,
):
    out_dir = ensure_dir(out_dir)
    breed_names = [name for _, name in classes if name in p_vals_dict]
    breed_names += sorted(set(p_vals_dict) - set(breed_names))
    positions_by_layer = [to_numpy(pos) for pos in layer_positions]
    tile_sizes = [_infer_tile_size(pos) for pos in positions_by_layer]
    vmax = -np.log10(p_colorbar_min)
    norm = Normalize(vmin=0.0, vmax=vmax)
    p_ticks = [1.0, p_threshold, 0.001, p_colorbar_min]
    p_tick_values = [-np.log10(p) for p in p_ticks]
    p_tick_labels = [
        "1",
        f"{p_threshold:g}",
        "0.001",
        f"{p_colorbar_min:.0e}",
    ]
    tick_pairs = [
        (tick, label)
        for tick, label in zip(p_tick_values, p_tick_labels)
        if tick <= vmax + 1e-9
    ]
    plot_label = plot_label or file_prefix.replace("_", " ")

    for breed_name in breed_names:
        fig, axes = plt.subplots(
            1,
            len(layer_positions),
            figsize=(3.2 * len(layer_positions), 3.35),
            constrained_layout=True,
        )
        if len(layer_positions) == 1:
            axes = [axes]
        collection = None
        for layer_idx, ax in enumerate(axes):
            positions = positions_by_layer[layer_idx]
            tile_size = tile_sizes[layer_idx]
            vals = -np.log10(np.clip(to_numpy(p_vals_dict[breed_name][layer_idx]), 1e-12, 1.0)).reshape(-1)
            collection = PatchCollection(
                _rects_for_positions(positions, tile_size),
                array=vals,
                cmap="magma",
                norm=norm,
                edgecolor="none",
                rasterized=True,
            )
            ax.add_collection(collection)
            _style_sheet_axis(ax, positions, tile_size)
            ax.set_title(f"Layer {layer_idx + 1}", fontsize=9)
        fig.suptitle(f"{breed_name} corrected p-values", fontsize=11, fontweight="bold")
        cbar = fig.colorbar(
            collection,
            ax=axes,
            orientation="horizontal",
            shrink=0.88,
            pad=0.08,
            aspect=34,
            ticks=[tick for tick, _ in tick_pairs],
            label="FWE-corrected p",
        )
        cbar.ax.set_xticklabels([label for _, label in tick_pairs], fontsize=8)
        safe_name = breed_name.lower().replace(" ", "_")
        path = out_dir / f"{file_prefix}_pvals_{safe_name}.svg"
        savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved corrected p-value map: {path}")

    print(f"{plot_label}: fixed corrected p-value color scale [1, {p_colorbar_min:.2e}]")


def plot_floc_pvals(
    ckpt,
    out_dir,
    device="cuda",
    fwhm_mm=2.0,
    resolution_mm=1.0,
    p_threshold=0.05,
    p_colorbar_min=1e-12,
    dpi=300,
):
    t_vals_dict, p_vals_dict, layer_positions = localizers(
        ckpt,
        dataset_names=["vpnl"],
        device=device,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
        ret_merged=True,
    )
    del t_vals_dict
    available_classes = [(key, label) for key, label in FLOC_CLASSES if key in p_vals_dict]
    plot_dog_breed_pvals(
        p_vals_dict,
        layer_positions,
        out_dir,
        classes=available_classes,
        file_prefix="floc",
        plot_label="fLoc",
        p_threshold=p_threshold,
        p_colorbar_min=p_colorbar_min,
        dpi=dpi,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a 5-way ImageNet category localizer on a transformed model."
    )
    parser.add_argument("--ckpt", default=MODEL_CKPT)
    parser.add_argument("--imagenet-root", type=Path, default=DEFAULT_IMAGENET_ROOT)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="dog_breed")
    parser.add_argument("--images-per-breed", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(PLOTS_DIR / "imagenet_dog_breed_localizers"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames-per-video", type=int, default=24)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--fwhm-mm", type=float, default=2.0)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    parser.add_argument("--p-threshold", type=float, default=LOCALIZER_P_THRESHOLD)
    parser.add_argument("--pmap-threshold", type=float, default=0.05)
    parser.add_argument("--floc-p-threshold", type=float, default=0.05)
    parser.add_argument("--t-threshold", type=float, default=LOCALIZER_T_THRESHOLD)
    parser.add_argument("--also-floc-pvals", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    classes = preset["classes"]
    colors = preset["colors"]
    file_prefix = preset["file_prefix"]
    datasets = collect_dog_breed_dataset(
        args.imagenet_root,
        classes=classes,
        images_per_breed=args.images_per_breed,
        seed=args.seed,
    )
    print(f"Selected ImageNet {preset['label']}:")
    for breed_name, items in datasets.items():
        print(f"  {breed_name}: {len(items)} images")

    model, _ = load_transformed_model(args.ckpt, device=args.device)
    model.eval()
    transform = _get_input_transform(model)

    original_name = model.name
    model.name = (
        f"{original_name}_{preset['name_suffix']}_n{args.images_per_breed}_seed{args.seed}"
    )
    try:
        with model.smoothing_enabled(
            fwhm_mm=args.fwhm_mm,
            resolution_mm=args.resolution_mm,
        ):
            layer_positions = _get_layer_positions(model)
            t_vals_dict, p_vals_dict = functional_localization_one_vs_rest(
                model,
                transform,
                datasets=datasets,
                batch_size=args.batch_size,
                device=args.device,
                video_fps=args.video_fps,
                frames_per_video=args.frames_per_video,
                include_positive_in_rest=True,
            )
    finally:
        model.name = original_name

    p_vals_dict = _correct_p_values(p_vals_dict)
    out_dir = ensure_dir(Path(args.out_dir) / Path(args.ckpt).stem.replace(".", "_"))
    plot_dog_breed_overlay(
        t_vals_dict,
        p_vals_dict,
        layer_positions,
        out_dir,
        classes=classes,
        colors=colors,
        file_prefix=file_prefix,
        p_threshold=args.p_threshold,
        t_threshold=args.t_threshold,
        dpi=args.dpi,
    )
    plot_dog_breed_tvals(
        t_vals_dict,
        layer_positions,
        out_dir,
        classes=classes,
        file_prefix=file_prefix,
        dpi=args.dpi,
    )
    plot_dog_breed_pvals(
        p_vals_dict,
        layer_positions,
        out_dir,
        classes=classes,
        file_prefix=file_prefix,
        p_threshold=args.pmap_threshold,
        dpi=args.dpi,
    )
    if args.also_floc_pvals:
        plot_floc_pvals(
            args.ckpt,
            out_dir,
            device=args.device,
            fwhm_mm=args.fwhm_mm,
            resolution_mm=args.resolution_mm,
            p_threshold=args.floc_p_threshold,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
