import argparse
import copy
import glob
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator

from config import PLOTS_DIR
from utils import cached
from .get_neural_alignment import neural_alignment
from .get_task_performance import task_performance
from .get_validate_features import _validate_features
from topo.loss import GlobalSpatialCorrelationLoss
from .plot_utils import savefig


DEFAULT_PATTERN = (
    "/mnt/scratch/ytang/tdann/cache/checkpoints/"
    "checkpoint_transformed_model_global_vjepa_18_single_neighbInf_"
    "smthsmthv2_lr1e-4_bs32_sd40_epoch_*.pt"
)


def _extract_epoch(path):
    match = re.search(r"_epoch_(\d+)\.pt$", str(path))
    if not match:
        raise ValueError(f"Could not parse epoch from {path}")
    return int(match.group(1))


def _positions_to_device(layer_positions, device):
    moved = []
    for pos in layer_positions:
        pos_copy = copy.copy(pos)
        coords = pos.coordinates
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords.astype(np.float32))
        pos_copy.coordinates = coords.to(device)
        nbrs = pos.neighborhood_indices
        if isinstance(nbrs, np.ndarray):
            nbrs = torch.from_numpy(nbrs.astype(np.int64))
        pos_copy.neighborhood_indices = nbrs.to(device)
        moved.append(pos_copy)
    return moved


def estimate_topographic_loss(
    ckpt_name,
    batch_size=32,
    samples_per_batch=2048,
    max_batches=None,
    device="cpu",
):
    cache_name = (
        f"topographic_loss_{ckpt_name}_bs{batch_size}_spb{samples_per_batch}"
        f"_maxb{max_batches}_device{device}"
    )

    @cached(cache_name, rerun=False)
    def _cached_topographic_loss():
        all_features, layer_positions = _validate_features(ckpt_name)
        layer_positions = _positions_to_device(layer_positions, device)

        num_samples = all_features[0].shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)

        criterion = GlobalSpatialCorrelationLoss(samples_per_batch=samples_per_batch).to(device)
        losses = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, num_samples)
            batch_features = [feat[start:end].to(device) for feat in all_features]
            with torch.no_grad():
                loss = criterion(batch_features, layer_positions)
            losses.append(loss.item())

        return float(np.mean(losses))

    return _cached_topographic_loss()


def compute_timeseries(
    ckpt_paths,
    num_splits=1,
    batch_size=32,
    samples_per_batch=2048,
    max_batches=None,
    device="cpu",
):
    epochs = []
    neural_scores = []
    neural_scores_pre = []
    task_scores = []
    task_scores_pre = []
    topo_loss_scores = []

    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).name
        epoch = _extract_epoch(ckpt_path)

        scores_pre, scores_post, mask, ceiling = neural_alignment(ckpt_name, num_splits=num_splits)
        ceiling_np = np.asarray(ceiling)
        mask_np = np.asarray(mask)
        ceiling_val = float(np.mean(ceiling_np[:, mask_np]))
        scores_post_np = np.asarray(scores_post)
        scores_pre_np = np.asarray(scores_pre)
        neural_scores_pre.append(float(scores_pre_np.mean() / ceiling_val))
        neural_scores.append(float(scores_post_np.mean() / ceiling_val))

        imagenet_pre, imagenet_post = task_performance(ckpt_name, "imagenet")
        ssv2_pre, ssv2_post = task_performance(ckpt_name, "ssv2")
        task_scores_pre.append(float((imagenet_pre + ssv2_pre) / 2.0))
        task_scores.append(float((imagenet_post + ssv2_post) / 2.0))

        topo_loss_scores.append(
            estimate_topographic_loss(
                ckpt_name,
                batch_size=batch_size,
                samples_per_batch=samples_per_batch,
                max_batches=max_batches,
                device=device,
            )
        )

        epochs.append(epoch)

    order = np.argsort(epochs)
    epochs = np.array(epochs)[order]
    neural_scores = np.array(neural_scores)[order]
    neural_scores_pre = np.array(neural_scores_pre)[order]
    task_scores = np.array(task_scores)[order]
    task_scores_pre = np.array(task_scores_pre)[order]
    topo_loss_scores = np.array(topo_loss_scores)[order]

    pre_neural = float(np.mean(neural_scores_pre))
    pre_task = float(np.mean(task_scores_pre))
    return epochs, neural_scores, task_scores, topo_loss_scores, pre_neural, pre_task


def plot_timeseries(
    epochs,
    neural_scores,
    task_scores,
    topo_loss_scores,
    pre_neural,
    pre_task,
    save_path,
):
    fig, (ax_loss, ax_joint) = plt.subplots(1, 2, figsize=(6.2, 3.0), sharex=True)

    ax_loss.plot(epochs, topo_loss_scores, marker="o", color="#C62828", linewidth=1.6)
    ax_loss.set_ylabel("Topographic loss", fontsize=9)
    ax_loss.set_xlabel("Epoch", fontsize=9)

    ax_joint.plot(epochs, neural_scores, marker="o", color="#000000", linewidth=1.6, label="Neural alignment")
    ax_joint.axhline(
        pre_neural,
        color="#000000",
        linestyle="--",
        linewidth=1.2,
        label="Neural alignment (pre)",
    )
    ax_joint.set_ylabel("Neural alignment", fontsize=9)
    ax_joint.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_joint.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax_joint_right = ax_joint.twinx()
    ax_joint_right.plot(
        epochs,
        task_scores,
        marker="o",
        color="#6E6E6E",
        linewidth=1.6,
        label="Task perf. (avg)",
    )
    ax_joint_right.axhline(
        pre_task,
        color="#6E6E6E",
        linestyle="--",
        linewidth=1.2,
        label="Task perf. (avg, pre)",
    )
    ax_joint_right.set_ylabel("Task perf. (avg)", fontsize=9)
    ax_joint.set_xlabel("Epoch", fontsize=9)

    ax_loss.set_box_aspect(1)
    ax_joint.set_box_aspect(1)

    for ax in (ax_loss, ax_joint):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax.minorticks_off()

    ax_joint_right.spines["top"].set_visible(False)
    ax_joint_right.spines["right"].set_visible(True)
    ax_joint_right.tick_params(axis="both", labelsize=8)
    ax_joint_right.minorticks_off()

    handles_left, labels_left = ax_joint.get_legend_handles_labels()
    handles_right, labels_right = ax_joint_right.get_legend_handles_labels()
    ax_joint.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        fontsize=8,
        frameon=False,
        loc="best",
    )

    plt.tight_layout()
    savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot neural alignment, task performance, and topographic loss across checkpoints."
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Checkpoint glob pattern.")
    parser.add_argument("--num-splits", type=int, default=1, help="Splits for neural alignment.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for loss estimate.")
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=2048,
        help="Samples per batch for GlobalSpatialCorrelationLoss.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=8,
        help="Limit number of batches for loss estimate.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for loss estimation.",
    )
    parser.add_argument(
        "--out",
        default=str(PLOTS_DIR / "alignment_task_loss_timeseries.svg"),
        help="Output path for the plot.",
    )
    args = parser.parse_args()

    ckpt_paths = sorted(glob.glob(args.pattern))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {args.pattern}")

    epochs, neural_scores, task_scores, topo_loss_scores, pre_neural, pre_task = compute_timeseries(
        ckpt_paths,
        num_splits=args.num_splits,
        batch_size=args.batch_size,
        samples_per_batch=args.samples_per_batch,
        max_batches=args.max_batches,
        device=args.device,
    )

    plot_timeseries(
        epochs,
        neural_scores,
        task_scores,
        topo_loss_scores,
        pre_neural,
        pre_task,
        Path(args.out),
    )


if __name__ == "__main__":
    main()
