import argparse
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data import ImageNetVid, Kinetics400, SmthSmthV2
from models import vit_transform
from topo import SOMTopoVJEPA
from topo.som import initialize_weights_from_samples, som_batch_update, som_checkpoint_name


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _build_dataset(data_name):
    dataset_builders = {
        "smthsmthv2": SmthSmthV2,
        "kinetics400": Kinetics400,
        "imagenet": ImageNetVid,
    }
    try:
        dataset_cls = dataset_builders[data_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {data_name}") from exc
    return dataset_cls(train_transforms=vit_transform, test_transforms=vit_transform)


def _build_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(1, int(batch_size / 1.5)),
        pin_memory=True,
    )


def _checkpoint_dir():
    path = config.CACHE_DIR / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(args):
    return _checkpoint_dir() / som_checkpoint_name(
        args.data_name,
        batch_size=args.batch_size,
        seed=args.seed,
        layer_indices=args.layer_indices,
        unit_mm=args.unit_mm,
    )


@torch.no_grad()
def _sample_vectors(model, videos, max_samples, generator):
    features = model.extract_som_input_features(videos)
    vectors = model.flatten_som_input_features(features)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    if vectors.shape[0] <= max_samples:
        return vectors
    indices = torch.randperm(vectors.shape[0], device=vectors.device, generator=generator)[:max_samples]
    return vectors[indices]


def _schedule(start, end, epoch, num_epochs):
    if num_epochs <= 1:
        return end
    progress = epoch / (num_epochs - 1)
    return start * ((end / start) ** progress)


def _update_ema(old_value, new_value, momentum):
    if old_value is None:
        return float(new_value)
    return momentum * old_value + (1.0 - momentum) * float(new_value)


def train_som(model, train_loader, args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    model.to(device)
    model.eval()

    max_sigma = math.hypot(*model.grid_shape) / 2.0
    start_sigma = args.start_sigma or max_sigma
    end_sigma = args.end_sigma

    initialized = False
    global_step = 0
    qe_ma = None
    active_ma = None
    for epoch in range(args.num_epochs):
        lr = _schedule(args.start_lr, args.end_lr, epoch, args.num_epochs)
        sigma = _schedule(start_sigma, end_sigma, epoch, args.num_epochs)
        desc = f"SOM epoch {epoch + 1}/{args.num_epochs} lr={lr:.4f} sigma={sigma:.2f}"

        pbar = tqdm(train_loader, desc=desc)
        for batch_idx, batch in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            videos = batch[0].to(device, non_blocking=True)
            samples = _sample_vectors(model, videos, args.samples_per_batch, generator)

            if not initialized:
                initialize_weights_from_samples(model.som_weights, samples, generator=generator)
                initialized = True

            metrics = som_batch_update(
                model.som_weights,
                samples,
                model.som_grid_coordinates,
                lr=lr,
                sigma=sigma,
                chunk_size=args.som_batch_size,
            )
            if global_step >= args.stats_warmup_steps:
                qe_ma = _update_ema(qe_ma, metrics["quantization_error"], args.stats_momentum)
                active_ma = _update_ema(active_ma, metrics["active_units"], args.stats_momentum)
            pbar.set_postfix(
                {
                    "qe": f"{metrics['quantization_error']:.4f}",
                    "qe_ma": f"{qe_ma:.4f}" if qe_ma is not None else "warmup",
                    "active": metrics["active_units"],
                    "active_ma": f"{active_ma:.1f}" if active_ma is not None else "warmup",
                }
            )
            global_step += 1

    if not initialized:
        raise RuntimeError("SOM training did not see any batches.")

    return global_step


def _cpu_state_dict(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def get_args():
    parser = argparse.ArgumentParser(description="Train a VJEPA topographic SOM baseline.")
    parser.add_argument("--data_name", default="kinetics400", choices=["smthsmthv2", "kinetics400", "imagenet"])
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[22])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--samples_per_batch", type=int, default=4096, help="Maximum video-level vectors per batch.")
    parser.add_argument("--som_batch_size", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--start_lr", type=float, default=0.01)
    parser.add_argument("--end_lr", type=float, default=0.001)
    parser.add_argument("--start_sigma", type=float, default=None)
    parser.add_argument("--end_sigma", type=float, default=2.0)
    parser.add_argument("--stats_momentum", type=float, default=0.95)
    parser.add_argument("--stats_warmup_steps", type=int, default=1)
    parser.add_argument("--unit_mm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = get_args()
    _set_seed(args.seed)
    checkpoint_path = _checkpoint_path(args)

    data = _build_dataset(args.data_name)
    train_loader = _build_loader(data.trainset, args.batch_size, shuffle=True)
    model = SOMTopoVJEPA(layer_indices=args.layer_indices, unit_mm=args.unit_mm, seed=args.seed)

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint["som_model_state_dict"], strict=False)
        print(f"Resumed SOM checkpoint from {checkpoint_path}")

    steps = train_som(model, train_loader, args, args.device)
    torch.save(
        {
            "epoch": args.num_epochs - 1,
            "steps": steps,
            "args": vars(args),
            "som_model_state_dict": _cpu_state_dict(model),
            "layer_dims": getattr(model, "layer_dims", None),
            "layer_names": getattr(model, "layer_names", None),
        },
        checkpoint_path,
    )
    print(f"Saved SOM checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
