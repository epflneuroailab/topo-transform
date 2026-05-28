import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

import config
from data import ImageNetVid
from data import Kinetics400
from data import SmthSmthV2
from models import clip_transform
from models import vit_transform
from topo import GlobalSpatialCorrelationLoss
from topo import SpatialCorrelationLoss
from topo import TopoTransformedCLIP
from topo import TopoTransformedVideoMAE
from topo import TopoTransformedVJEPA

try:
    import wandb
except ImportError:
    wandb = None


if hasattr(config, "WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY


def get_config_id(model_name, data_name, lr, batch_size=32, seed=42, prefix="", split_suffix=""):
    lr_str = f"lr{lr}".replace(".", "p") if lr >= 0.001 else f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    split_part = f"_{split_suffix}" if split_suffix else ""
    return f"{prefix}{model_name}_{data_name}_{lr_str}_bs{batch_size}_sd{seed}{split_part}"


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _build_dataset(data_name, transform=vit_transform):
    dataset_builders = {
        "smthsmthv2": SmthSmthV2,
        "kinetics400": Kinetics400,
        "imagenet": ImageNetVid,
    }
    try:
        dataset_cls = dataset_builders[data_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {data_name}") from exc
    return dataset_cls(train_transforms=transform, test_transforms=transform)


def _build_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(batch_size / 1.5),
        pin_memory=True,
    )


def _split_dataset(dataset, split_name, split_seed=42):
    if split_name == "all":
        return dataset, ""
    if split_name not in {"half_a", "half_b"}:
        raise ValueError(f"Unknown train split: {split_name}")

    generator = torch.Generator()
    generator.manual_seed(split_seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    midpoint = len(indices) // 2
    selected = indices[:midpoint] if split_name == "half_a" else indices[midpoint:]
    return Subset(dataset, selected), split_name


def _checkpoint_dir():
    path = config.CACHE_DIR / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(config_id):
    return _checkpoint_dir() / f"best_transformed_model_{config_id}.pt"


def _results_path(storage, config_id):
    return storage / f"{config_id}_results.pkl"


def _load_training_state(model, optimizer, scheduler, checkpoint_path, storage, config_id, device):
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    if not checkpoint_path.exists():
        print(f"--resume_training is enabled, but no checkpoint found at {checkpoint_path}. Starting from scratch.")
        return start_epoch, best_val_loss, train_losses, val_losses, None

    print(f"\n{'=' * 70}\nResuming from: {checkpoint_path}\n{'=' * 70}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["transformed_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        for _ in range(checkpoint["epoch"] + 1):
            scheduler.step()

    results_path = _results_path(storage, config_id)
    if results_path.exists():
        results = torch.load(results_path)
        train_losses = results.get("train_losses", [])
        val_losses = results.get("val_losses", [])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
    print(f"Resuming from epoch {start_epoch + 1}, best val loss: {best_val_loss:.6f}\n{'=' * 70}\n")
    return start_epoch, best_val_loss, train_losses, val_losses, checkpoint.get("wandb_run_id")


def _save_checkpoint(model, optimizer, scheduler, epoch, config_id, train_loss, val_loss, path, wandb_run_id):
    torch.save(
        {
            "epoch": epoch,
            "config_id": config_id,
            "transformed_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "layer_dims": getattr(model, "layer_dims", None),
            "layer_names": getattr(model, "layer_names", None),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def _save_epoch_checkpoint(model, optimizer, scheduler, epoch, config_id, train_loss, val_loss, wandb_run_id):
    checkpoint_file = _checkpoint_dir() / f"checkpoint_transformed_model_{config_id}_epoch_{epoch + 1}.pt"
    torch.save(
        {
            "epoch": epoch,
            "config_id": config_id,
            "transformed_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "wandb_run_id": wandb_run_id,
        },
        checkpoint_file,
    )
    print(f"  -> Saved checkpoint at epoch {epoch + 1}")


def _save_results(storage, config_id, train_losses, val_losses, best_val_loss, model):
    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "config_id": config_id,
            "layer_configs": getattr(model, "layer_configs", None),
        },
        _results_path(storage, config_id),
    )


def _init_wandb(use_wandb, checkpoint_path, resume, device, train_loader, config_id, num_epochs, lr, wandb_project, wandb_run_name):
    if not use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is required when --use_wandb is set.")

    wandb_run_id = None
    if resume and checkpoint_path.exists():
        wandb_run_id = torch.load(checkpoint_path, map_location=device).get("wandb_run_id")

    wandb.init(
        project=wandb_project,
        name=wandb_run_name or config_id,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        config={
            "config_id": config_id,
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": train_loader.batch_size,
            "device": device,
        },
    )
    return wandb.run.id


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    config_id,
    storage,
    figure_dir,
    device="cuda",
    num_epochs=50,
    lr=1e-3,
    resume=True,
    use_wandb=False,
    wandb_project="topo-transform",
    wandb_run_name=None,
):
    checkpoint_path = _checkpoint_path(config_id)
    wandb_run_id = _init_wandb(
        use_wandb,
        checkpoint_path,
        resume,
        device,
        train_loader,
        config_id,
        num_epochs,
        lr,
        wandb_project,
        wandb_run_name,
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    if resume:
        start_epoch, best_val_loss, train_losses, val_losses, resume_run_id = _load_training_state(
            model,
            optimizer,
            scheduler,
            checkpoint_path,
            storage,
            config_id,
            device,
        )
        if wandb_run_id is None:
            wandb_run_id = resume_run_id

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            videos = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(*model(videos))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            if use_wandb:
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch * len(train_loader) + batch_idx,
                )

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                videos = batch[0].to(device, non_blocking=True)
                val_feature, layer_positions = model(videos)
                val_loss += criterion(val_feature, layer_positions).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=(epoch + 1) * len(train_loader),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config_id,
                train_loss,
                val_loss,
                checkpoint_path,
                wandb_run_id,
            )
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
            if use_wandb:
                wandb.log({"best_model_epoch": epoch + 1})

        _save_epoch_checkpoint(model, optimizer, scheduler, epoch, config_id, train_loss, val_loss, wandb_run_id)
        scheduler.step()
        _save_results(storage, config_id, train_losses, val_losses, best_val_loss, model)
        print()

    if use_wandb:
        wandb.finish()

    return model, train_losses, val_losses


def _plot_losses(train_losses, val_losses, config_id, figure_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"TopoTransform Training - {config_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figure_dir / f"topo_training_{config_id}.png", dpi=150, bbox_inches="tight")
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vjepa", choices=["vjepa", "clip", "videomae"])
    parser.add_argument("--data_name", type=str, default="kinetics400", choices=["smthsmthv2", "kinetics400", "imagenet"])
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[14, 18, 22], help="List of layer indices")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--samples_per_batch", type=int, default=8192 * 2)
    parser.add_argument("--neighborhoods_per_batch", type=int, default=16)
    parser.add_argument("--tissue_config", choices=["vtc", "small"], default="vtc")
    parser.add_argument("--rf_overlap", type=float, default=None)
    parser.add_argument("--inf_neighborhood", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_split", choices=["all", "half_a", "half_b"], default="all")
    parser.add_argument("--train_split_seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="tdann-transform")
    parser.add_argument("--resume_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    _set_seed(args.seed)

    transform = clip_transform if args.model_name == "clip" else vit_transform
    data = _build_dataset(args.data_name, transform=transform)
    trainset, split_suffix = _split_dataset(data.trainset, args.train_split, args.train_split_seed)
    print(f"Training split: {args.train_split} ({len(trainset)}/{len(data.trainset)} samples)")
    train_loader = _build_loader(trainset, args.batch_size, shuffle=True)
    val_loader = _build_loader(data.valset, args.batch_size, shuffle=False)

    model_classes = {"vjepa": TopoTransformedVJEPA, "clip": TopoTransformedCLIP, "videomae": TopoTransformedVideoMAE}
    model_cls = model_classes[args.model_name]
    rf_overlap = args.rf_overlap
    if args.tissue_config == "small" and rf_overlap is None:
        rf_overlap = 0.1
    inf_neighborhood = args.inf_neighborhood and args.tissue_config != "small"
    model = model_cls(
        layer_indices=args.layer_indices,
        single_sheet=True,
        inf_neighborhood=inf_neighborhood,
        tissue_config=args.tissue_config,
        rf_overlap_override=rf_overlap,
        seed=args.seed,
    )
    config_id = get_config_id(
        model.name,
        args.data_name,
        args.lr,
        args.batch_size,
        args.seed,
        prefix="global_",
        split_suffix=split_suffix,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tissue_config == "small":
        criterion = SpatialCorrelationLoss(
            num_layers=len(args.layer_indices),
            neighborhoods_per_batch=args.neighborhoods_per_batch,
            single_sheet=True,
        )
        print(
            "Using local neighborhood SpatialCorrelationLoss "
            f"({args.neighborhoods_per_batch} neighborhoods/batch) for small tissue."
        )
    else:
        criterion = GlobalSpatialCorrelationLoss(samples_per_batch=args.samples_per_batch)
        print(
            "Using global sampled SpatialCorrelationLoss "
            f"({args.samples_per_batch} sampled units/batch)."
        )

    storage_path = config.CACHE_DIR / "train_topo" / config_id
    storage_path.mkdir(parents=True, exist_ok=True)
    figure_dir = config.CACHE_DIR / "figures" / config_id
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}\nTraining {args.model_name} | Config: {config_id} | Device: {device}\n{'=' * 70}")
    _, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        config_id,
        storage_path,
        figure_dir,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        resume=args.resume_training,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=config_id,
    )

    _plot_losses(train_losses, val_losses, config_id, figure_dir)
    _save_results(storage_path, config_id, train_losses, val_losses, min(val_losses), model)
    print(f"\n{'=' * 70}\nTraining Complete! Best Val Loss: {min(val_losses):.6f}\n{'=' * 70}")


if __name__ == "__main__":
    main()
