"""Figure 2b (right): exemplary unit-activation traces.

Single seeded source of truth for the 3 selected example units, shared by the figure
renderer and the Source Data export so the data always matches the chosen figure.

Reproduces the original selection (`test_autocorr.py`): the only randomness is the
probe unit (`probe_idx = np.random.choice(C*H*W, num_probes)[1]` under `seed`); units 2
and 3 are then deterministic. Rerun with different seeds to pick a good example:

    python -m scripts.plot_fig2b --seed 32
    python -m scripts.plot_fig2b --seed 33 --seed 34 ...
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import DEBUG_DIR, PLOTS_DIR
from .common import MODEL_CKPT
from .get_validate_features import validate_features
from .plot_utils import savefig


def fig2b_unit_data(all_features, positions, seed, layer_idx=0, max_num_stimuli=5, num_probes=5):
    """Select the 3 example units for a given seed and return their traces.

    Returns a dict: selected_indices, positions[3,2], activations[B,T,3], B, T,
    corr[3] (autocorrelation to the probe), distances[3] (cortical distance to probe).
    """
    lf = all_features[layer_idx]
    if lf.ndim == 5:
        B, T, C, H, W = lf.shape
    else:
        B, C, H, W = lf.shape
        T = 1
        lf = lf.unsqueeze(1)
    n_units = C * H * W

    # Probe unit: 2nd of `num_probes` random draws (mirrors visualize_random_autocorr).
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe_idx = int(np.random.choice(n_units, min(num_probes, n_units), replace=False)[1])

    # Unit selection (mirrors visualize_unit_activations_over_time); re-seed as it does.
    np.random.seed(seed)
    torch.manual_seed(seed)
    Bn = min(B, max_num_stimuli)
    lf = lf[:Bn]
    feats = lf.reshape(Bn, T, n_units)
    feats_flat = feats.reshape(-1, n_units)
    x = (feats_flat - feats_flat.mean(dim=0, keepdim=True)) / (feats_flat.std(dim=0, keepdim=True) + 1e-9)

    pos = positions[layer_idx].coordinates
    pos_np = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
    N = pos_np.shape[0]

    probe_x = x[:, probe_idx:probe_idx + 1]
    autocorr = torch.mm(probe_x.T, x).flatten() / x.shape[0]
    autocorr[probe_idx] = 0.0
    autocorr_np = autocorr.detach().cpu().numpy()
    distances = np.linalg.norm(pos_np - pos_np[probe_idx], axis=1)

    idx1 = probe_idx

    # Unit 2: strongly correlated, moderate distance.
    min_dist = np.percentile(distances[distances > 0], 7)
    correlated_mask = (autocorr_np > 0.55) & (distances < min_dist)
    if np.any(correlated_mask):
        ci = np.where(correlated_mask)[0]
        target = np.percentile(distances, 40)
        idx2 = int(ci[np.argmin(np.abs(distances[ci] - target))])
    else:
        vi = np.arange(N)[np.arange(N) != probe_idx]
        idx2 = int(vi[np.argmax(autocorr_np[vi])])

    # Unit 3: strongly decorrelated, larger distance.
    decorr_mask = (autocorr_np < -0.3) & (distances > np.percentile(distances, 50))
    if np.any(decorr_mask):
        di = np.where(decorr_mask)[0]
        idx3 = int(di[np.argmin(autocorr_np[di])])
    else:
        fi = np.where(distances > np.percentile(distances, 70))[0]
        idx3 = int(fi[np.argmin(autocorr_np[fi])]) if len(fi) else int(np.argmin(autocorr_np))

    selected = [idx1, idx2, idx3]
    feats_np = feats.detach().cpu().numpy()
    return {
        "seed": seed,
        "selected_indices": selected,
        "positions": pos_np[selected],
        "activations": feats_np[:, :, selected],  # [B, T, 3]
        "B": Bn,
        "T": T,
        "corr": autocorr_np[selected],
        "distances": distances[selected],
        "autocorr_map": autocorr_np,
        "all_positions": pos_np,
    }


def plot_fig2b(data, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    pos_np = data["all_positions"]
    autocorr_np = data["autocorr_map"]
    selected = data["selected_indices"]
    acts = data["activations"]
    B, T = data["B"], data["T"]

    fig = plt.figure(figsize=(9, 4.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

    ax_sp = fig.add_subplot(gs[:, 0])
    vmax = float(np.max(np.abs(autocorr_np)))
    ax_sp.scatter(pos_np[:, 0], pos_np[:, 1], c=autocorr_np, cmap="RdBu_r", s=1.5,
                  vmin=-vmax, vmax=vmax, edgecolors="none", rasterized=True)
    ax_sp.set_xlim(pos_np[:, 0].min(), pos_np[:, 0].max())
    ax_sp.set_ylim(pos_np[:, 1].min(), pos_np[:, 1].max())
    cmap = plt.get_cmap("RdBu_r")
    unit_colors = [cmap(0.88), cmap(0.88), cmap(0.12)]  # red, red, blue
    for idx, color in zip(selected, unit_colors):
        ax_sp.scatter(pos_np[idx, 0], pos_np[idx, 1], c=[color], s=200, marker="*",
                      linewidths=3, edgecolors="black", zorder=10)
    ax_sp.set_title("Selected Units\n(Autocorrelation Map)", fontsize=13, fontweight="bold")
    ax_sp.set_aspect("equal")
    ax_sp.set_facecolor("#f8f8f8")

    time_axis = np.arange(B * T)
    for k, color in enumerate(unit_colors):
        ax = fig.add_subplot(gs[k, 1])
        ax.plot(time_axis, acts[:, :, k].flatten(), color=color, linewidth=2, alpha=0.8)
        for b in range(1, B):
            ax.axvline(x=b * T - 0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, B * T - 1)
        ax.set_ylabel("Activation", fontsize=11)
        if k == 2:
            ax.set_xlabel("Time (across stimuli)", fontsize=11)
        else:
            ax.set_xticklabels([])
    fig.suptitle(f"Fig 2b unit activations (seed {seed})", fontsize=14, y=0.98)
    path = out_dir / f"fig2b_seed{seed}.svg"
    plt.savefig(path, bbox_inches="tight", dpi=300, format="svg")
    plt.close(fig)
    return path


def plot_source_traces():
    """Render the published traces from the cached Source Data payload."""
    candidates = (
        DEBUG_DIR / "fig2b_traces.json",
        PLOTS_DIR / "fig2b" / "fig2b_traces.json",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "fig2b_traces.json was not found in cache/debug or cache/plots/fig2b"
        )

    rows = json.loads(source.read_text())["rows"]
    time = np.asarray([int(row[2]) for row in rows])
    traces = np.asarray(
        [[float(row[3]), float(row[4]), float(row[5])] for row in rows]
    )
    colors = ("#B82531", "#B82531", "#2769A7")
    fig, axes = plt.subplots(3, 1, figsize=(4.2, 3.8), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(time, traces[:, index], color=colors[index], linewidth=1.7)
        for boundary in (11.5, 23.5, 35.5, 47.5):
            axis.axvline(
                boundary,
                color="#9E9E9E",
                linestyle="--",
                linewidth=1,
            )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.set_xlim(0, 59)

    axes[0].set_yticks([0, 1])
    axes[1].set_yticks([-0.5, 0])
    axes[2].set_yticks([0, 1])
    for clip, x_position in zip(
        ("S1", "S2", "S3", "S4", "S5"),
        (5.5, 17.5, 29.5, 41.5, 53.5),
    ):
        axes[0].text(
            x_position,
            axes[0].get_ylim()[1],
            clip,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[2].set_xlabel("Time (across stimuli)", fontsize=9)
    fig.tight_layout()
    savefig(
        PLOTS_DIR / "fig2b" / "fig2b_traces.svg",
        dpi=300,
        bbox_inches="tight",
    )


def main():
    parser = argparse.ArgumentParser(description="Render Fig 2b exemplary unit activations for chosen seed(s).")
    parser.add_argument("--checkpoint_name", default=MODEL_CKPT)
    parser.add_argument("--seed", type=int, nargs="+", default=[35], help="One or more seeds to render.")
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--skip_source_traces", action="store_true")
    args = parser.parse_args()

    all_features, positions = validate_features(args.checkpoint_name)
    out_dir = PLOTS_DIR / "fig2b"
    for seed in args.seed:
        data = fig2b_unit_data(all_features, positions, seed=seed, layer_idx=args.layer_idx)
        path = plot_fig2b(data, out_dir, seed)
        u = data["selected_indices"]
        print(f"seed={seed}: units={u} "
              f"corr={[round(float(c), 2) for c in data['corr']]} "
              f"dist={[round(float(d), 1) for d in data['distances']]} -> {path}")
    if not args.skip_source_traces:
        plot_source_traces()


if __name__ == "__main__":
    main()
