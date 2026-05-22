import argparse

import torch

import config
from models import clip_transform
from models import vit_transform
from validate import load_transformed_model
from validate import validate_floc


def crop_config_id(ckpt_name):
    cleaned = ckpt_name.replace("best_transformed_model_", "").replace("checkpoint_transformed_model_", "")
    if cleaned.endswith(".pt"):
        cleaned = cleaned[:-3]
    if "_epoch" in cleaned:
        cleaned = cleaned.split("_epoch")[0]
    return cleaned


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=["vpnl", "biomotion", "kanwisher", "pitzalis"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fwhm_mm", type=float, default=1.0)
    parser.add_argument("--resolution_mm", type=float, default=1.0)
    parser.add_argument("--viz_patches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_individual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_aggregate", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_transformed_model(checkpoint_name=args.checkpoint_name, device=device)
    transform = clip_transform if model.__class__.__name__ == "TopoTransformedCLIP" else vit_transform

    config_id = crop_config_id(args.checkpoint_name)
    figure_dir = config.CACHE_DIR / "figures" / config_id
    figure_dir.mkdir(parents=True, exist_ok=True)

    with model.smoothing_enabled(fwhm_mm=args.fwhm_mm, resolution_mm=args.resolution_mm):
        validate_floc(
            model,
            transform,
            dataset_names=args.dataset_names,
            viz_dir=figure_dir,
            viz_params={},
            batch_size=args.batch_size,
            device=device,
            viz_patches=args.viz_patches,
            plot_individual=args.plot_individual,
            plot_aggregate=args.plot_aggregate,
        )


if __name__ == "__main__":
    main()
