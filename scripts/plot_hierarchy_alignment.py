import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr
from torch.utils.data import Dataset

from config import PLOTS_DIR
from data.neural_data import get_data_loader
from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data.store import pickle_store
from models import vit_transform
from utils import cached
from validate import load_transformed_model
from validate.neural_decoding import decoding_test, make_decoder
from validate.rois import glasser, nsd

try:
    from .common import MODEL_CKPT
except ImportError:
    from scripts.common import MODEL_CKPT


class MaskedTargetLoader(Dataset):
    def __init__(self, data_loader, target_mask):
        self.data_loader = data_loader
        self.target_mask = target_mask

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, idx):
        data, target = self.data_loader[idx]
        masked_target = target[..., self.target_mask]
        return data, masked_target


def _val_selection(decoding_scores):
    val_scores, test_scores = decoding_scores[:, 0, 0], decoding_scores[:, 0, 1]
    best_val_idx = val_scores.argmax(0)
    test_score = test_scores[best_val_idx, np.arange(test_scores.shape[1])]
    return test_score, best_val_idx


def _build_mesh_graph(coords, faces):
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    edges = np.vstack(
        [
            np.stack([i, j], axis=1),
            np.stack([j, k], axis=1),
            np.stack([k, i], axis=1),
        ]
    )
    edges = np.vstack([edges, edges[:, ::-1]])
    dists = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    graph = coo_matrix((dists, (edges[:, 0], edges[:, 1])), shape=(coords.shape[0], coords.shape[0]))
    return graph.tocsr()


def _geodesic_distances_from_seed(coords, faces, seed_idx):
    graph = _build_mesh_graph(coords, faces)
    seed_idx = np.atleast_1d(seed_idx).astype(np.int64)
    distances = dijkstra(graph, indices=seed_idx.tolist(), directed=False)
    if distances.ndim == 1:
        return distances
    return distances.min(axis=0)


def _v1_posterior_indices(coords, v1_indices, posterior_fraction=0.1):
    if v1_indices.size == 0:
        return v1_indices

    posterior_fraction = float(np.clip(posterior_fraction, 1e-3, 1.0))
    y_coords = coords[v1_indices, 1]
    n_keep = max(1, int(np.ceil(v1_indices.size * posterior_fraction)))
    order = np.argsort(y_coords)  # lower y is more posterior
    keep = order[:n_keep]
    return v1_indices[keep]


@cached("fsaverage5_v1_posterior_geodesic_distances_with_seed_coords_mirror_rh_to_lh_v3", persistent=True)
def _cached_v1_geodesic_distances():
    from nilearn import datasets, surface

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    coords_lh, faces_lh = surface.load_surf_mesh(fsaverage.pial_left)
    coords_rh, faces_rh = surface.load_surf_mesh(fsaverage.pial_right)

    lh_count = coords_lh.shape[0]
    v1_mask = glasser.get_region_voxels(["V1"])
    v1_indices = np.where(v1_mask)[0]
    v1_lh = v1_indices[v1_indices < lh_count]
    v1_rh = v1_indices[v1_indices >= lh_count] - lh_count

    total = lh_count + coords_rh.shape[0]
    distances = np.full(total, np.inf, dtype=np.float64)

    seed_info = {}

    posterior_rh = np.array([], dtype=np.int64)
    anchor_rh_seed_pos = None
    if v1_rh.size > 0:
        posterior_rh = _v1_posterior_indices(coords_rh, v1_rh)
        distances[lh_count:] = _geodesic_distances_from_seed(coords_rh, faces_rh, posterior_rh)
        anchor_rh_seed_pos = int(np.argmin(coords_rh[posterior_rh, 1]))
        anchor_rh_local = posterior_rh[anchor_rh_seed_pos]
        seed_info["rh"] = {
            "anchor_index": int(anchor_rh_local + lh_count),
            "anchor_coord": coords_rh[anchor_rh_local].astype(float).tolist(),
            "num_posterior_nodes": int(posterior_rh.size),
        }

    if v1_lh.size > 0:
        anchor_lh_local = None
        if posterior_rh.size > 0:
            # Mirror RH seed coordinates into LH by x -> -x, then snap to nearest LH V1 vertices.
            mirrored_rh_coords = coords_rh[posterior_rh].copy()
            mirrored_rh_coords[:, 0] *= -1.0
            lh_v1_coords = coords_lh[v1_lh]
            nn = np.linalg.norm(mirrored_rh_coords[:, None, :] - lh_v1_coords[None, :, :], axis=2).argmin(axis=1)
            posterior_lh = np.unique(v1_lh[nn]).astype(np.int64)
            if anchor_rh_seed_pos is not None:
                anchor_lh_local = int(v1_lh[nn[anchor_rh_seed_pos]])
            if posterior_lh.size == 0:
                posterior_lh = _v1_posterior_indices(coords_lh, v1_lh)
        else:
            posterior_lh = _v1_posterior_indices(coords_lh, v1_lh)

        distances[:lh_count] = _geodesic_distances_from_seed(coords_lh, faces_lh, posterior_lh)
        if anchor_lh_local is None:
            anchor_lh_local = int(posterior_lh[np.argmin(coords_lh[posterior_lh, 1])])
        seed_info["lh"] = {
            "anchor_index": int(anchor_lh_local),
            "anchor_coord": coords_lh[anchor_lh_local].astype(float).tolist(),
            "num_posterior_nodes": int(posterior_lh.size),
        }

    return {"distances": distances, "seed_info": seed_info}


def _to_numpy(arr):
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _extract_unit_x_positions(model):
    if not hasattr(model, "layer_positions"):
        raise ValueError("Model does not expose layer_positions; cannot segment along x-axis.")

    # In single-sheet mode, model.layer_positions can contain repeated references
    # to the same concatenated map. Unit indexing follows flattened C*H*W ordering.
    if getattr(model, "single_sheet", False):
        coords = _to_numpy(model.layer_positions[0].coordinates)
        return coords[:, 0]

    x_positions = []
    for layer_pos in model.layer_positions:
        coords = _to_numpy(layer_pos.coordinates)
        x_positions.append(coords[:, 0])
    return np.concatenate(x_positions, axis=0)


def _build_segments(channel_x, n_segments):
    channel_x = np.asarray(channel_x).astype(np.float64)
    n_channels = channel_x.shape[0]
    if n_channels == 0:
        raise ValueError("No channels found to segment.")

    n_segments = int(min(max(1, n_segments), n_channels))
    sorted_idx = np.argsort(channel_x)
    splits = np.array_split(sorted_idx, n_segments)

    x_left = channel_x.min()
    segment_indices = []
    segment_centers = []
    segment_distances = []
    for idx in splits:
        if idx.size == 0:
            continue
        x_center = channel_x[idx].mean()
        segment_indices.append(idx.astype(np.int64))
        segment_centers.append(float(x_center))
        segment_distances.append(float(x_center - x_left))

    return segment_indices, np.asarray(segment_centers), np.asarray(segment_distances), x_left


class SegmentExtractor:
    def __init__(self, model, n_segments=12, mode="posttransform"):
        self.do_transform = mode == "posttransform"
        self.unit_x = _extract_unit_x_positions(model)
        self.segment_indices, self.segment_centers, self.segment_distances, self.left_origin = _build_segments(
            self.unit_x, n_segments
        )

    def __call__(self, model, inputs):
        with torch.no_grad():
            layer_features, _ = model(inputs, do_transform=self.do_transform)

        if len(layer_features) != 1:
            raise ValueError(
                f"Expected one concatenated feature source, got {len(layer_features)}. "
                "This script currently supports single-sheet outputs."
            )

        feat = layer_features[0].mean(dim=1)  # B, C, H, W
        feat_flat = feat.reshape(feat.shape[0], -1)  # B, C*H*W
        if feat_flat.shape[1] != self.unit_x.shape[0]:
            raise ValueError(
                f"Feature/position mismatch: flattened features={feat_flat.shape[1]}, "
                f"position units={self.unit_x.shape[0]}"
            )

        return [feat_flat[:, idx] for idx in self.segment_indices]


def _hemisphere_mask(n_vertices, hemisphere):
    hemisphere = str(hemisphere).lower()
    if hemisphere not in {"both", "lh", "rh"}:
        raise ValueError(f"Invalid hemisphere '{hemisphere}'. Expected one of: both, lh, rh.")
    mask = np.ones(n_vertices, dtype=bool)
    half = n_vertices // 2
    if hemisphere == "lh":
        mask[half:] = False
    elif hemisphere == "rh":
        mask[:half] = False
    return mask


def _slice_cached_result_by_hemisphere(result, hemisphere, region_mask, geodesic_all):
    n_vertices = region_mask.size
    hemi_mask = _hemisphere_mask(n_vertices, hemisphere)

    masked_indices = np.where(region_mask)[0]
    finite = np.isfinite(geodesic_all[region_mask])
    finite_masked_indices = masked_indices[finite]

    aligned_distance_full = result.get("aligned_distance_full")
    if aligned_distance_full is None:
        aligned_distance_full = np.full(n_vertices, np.nan, dtype=np.float64)
        aligned_distance_full[finite_masked_indices] = result["aligned_distance"]

    aligned_score_full = np.full(n_vertices, np.nan, dtype=np.float64)
    aligned_score_full[finite_masked_indices] = result["aligned_score"]

    valid = hemi_mask & np.isfinite(aligned_distance_full) & np.isfinite(aligned_score_full) & np.isfinite(geodesic_all)
    seed_info = result.get("seed_info", {})
    if hemisphere in {"lh", "rh"}:
        seed_info = {hemisphere: seed_info[hemisphere]} if hemisphere in seed_info else {}

    return {
        "geodesic": geodesic_all[valid],
        "aligned_distance": aligned_distance_full[valid],
        "aligned_score": aligned_score_full[valid],
        "aligned_distance_full": aligned_distance_full,
        "seed_info": seed_info,
        "hemisphere": hemisphere,
        "num_segments": result["num_segments"],
    }


def hierarchy_alignment(ckpt_name, n_segments=12, num_splits=1, mode="posttransform", hemisphere="both"):
    def _safe_key(text):
        return (
            str(text)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
            .replace(".", "_")
        )

    cache_store = pickle_store.add_node("scripts").add_node("plot_hierarchy_alignment")
    base_key = (
        f"hierarchy_alignment_ckpt{_safe_key(ckpt_name)}"
        f"_N{int(n_segments)}_splits{int(num_splits)}_mode{_safe_key(mode)}"
    )
    cache_key_both = f"{base_key}_v4"
    cache_key = cache_key_both if hemisphere == "both" else f"{base_key}_hemi{_safe_key(hemisphere)}_v4"
    if cache_store.exists(cache_key):
        print(f"Loading cached hierarchy alignment: {cache_key}")
        return cache_store.load(cache_key)

    region_mask = nsd.get_region_voxels(
        [
            "high-ventral",
            "high-lateral",
            "high-dorsal",
        ]
    )
    geodesic_payload = _cached_v1_geodesic_distances()
    geodesic_all = geodesic_payload["distances"] if isinstance(geodesic_payload, dict) else geodesic_payload

    # Avoid rerunning decoding for LH/RH if a "both" cache already exists.
    if hemisphere in {"lh", "rh"} and cache_store.exists(cache_key_both):
        print(f"Loading cached hierarchy alignment: {cache_key_both}")
        shared = cache_store.load(cache_key_both)
        sliced = _slice_cached_result_by_hemisphere(shared, hemisphere, region_mask, geodesic_all)
        cache_store.store(sliced, cache_key)
        print(f"Stored hierarchy alignment cache: {cache_key}")
        return sliced

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    hemi_mask = _hemisphere_mask(region_mask.size, hemisphere)
    mask = region_mask & hemi_mask
    seed_info = {}
    if isinstance(geodesic_payload, dict):
        seed_info = geodesic_payload.get("seed_info", {})
        for hemi in ("lh", "rh"):
            if hemi in seed_info:
                info = seed_info[hemi]
                coord = info["anchor_coord"]
                print(
                    f"Posterior V1 seed ({hemi.upper()}): "
                    f"vertex={info['anchor_index']}, "
                    f"coord=({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}), "
                    f"posterior_nodes={info['num_posterior_nodes']}"
                )
    else:
        geodesic_all = geodesic_payload
    geodesic = geodesic_all[mask]
    finite = np.isfinite(geodesic)

    extractor = SegmentExtractor(model=model, n_segments=n_segments, mode=mode)
    print(f"Using {len(extractor.segment_indices)} x-axis segments.")
    print(f"Hemisphere={hemisphere.upper()} selected voxel count: {mask.sum()}")

    decoder = make_decoder(test_type="regress", device=device)
    ratios = (0.8, 0.1, 0.1)
    batch_size = 16

    voxel_segment_distances = []
    voxel_test_scores = []

    for split in range(num_splits):
        print(f"=== Split {split + 1}/{num_splits} ===")
        seed = 42 + split
        trainset, valset, testset, _ = get_compilation(vit_transform, ratios=ratios, return_ceiling=True, seed=seed)

        tr = MaskedTargetLoader(trainset, mask)
        va = MaskedTargetLoader(valset, mask)
        te = MaskedTargetLoader(testset, mask)

        train_loader = get_data_loader(tr, batch_size=batch_size, shuffle=True, num_workers=batch_size)
        val_loader = get_data_loader(va, batch_size=batch_size, shuffle=False, num_workers=batch_size)
        test_loader = get_data_loader(te, batch_size=batch_size, shuffle=False, num_workers=batch_size)

        scores = decoding_test(
            model=model,
            get_features=extractor,
            train_loaders=[train_loader],
            test_loaders=[val_loader, test_loader],
            downsampler=None,
            decoder=decoder,
            device=device,
        )

        test_score, best_segment_idx = _val_selection(scores)
        best_segment_distance = extractor.segment_distances[best_segment_idx]
        voxel_segment_distances.append(best_segment_distance)
        voxel_test_scores.append(test_score)

    voxel_segment_distances = np.asarray(voxel_segment_distances)  # [splits, voxels]
    voxel_test_scores = np.asarray(voxel_test_scores)  # [splits, voxels]

    aligned_distance = voxel_segment_distances.mean(axis=0)
    aligned_score = voxel_test_scores.mean(axis=0)

    masked_indices = np.where(mask)[0]
    finite_masked_indices = masked_indices[finite]
    aligned_distance_full = np.full(mask.shape[0], np.nan, dtype=np.float64)
    aligned_distance_full[finite_masked_indices] = aligned_distance[finite]

    result = {
        "geodesic": geodesic[finite],
        "aligned_distance": aligned_distance[finite],
        "aligned_score": aligned_score[finite],
        "aligned_distance_full": aligned_distance_full,
        "seed_info": {
            k: v
            for k, v in seed_info.items()
            if hemisphere == "both" or k == hemisphere
        },
        "hemisphere": hemisphere,
        "num_segments": len(extractor.segment_indices),
    }
    cache_store.store(result, cache_key)
    print(f"Stored hierarchy alignment cache: {cache_key}")
    return result


def _plot_segment_distance_flatmap(aligned_distance_full, save_path, dpi=300, seed_info=None):
    from nilearn import datasets, plotting, surface

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    n_lh = len(aligned_distance_full) // 2
    lh_map = aligned_distance_full[:n_lh]
    rh_map = aligned_distance_full[n_lh:]

    vmax = float(np.nanmax(aligned_distance_full))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), subplot_kw={"projection": "3d"})
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.flat_left,
        stat_map=lh_map,
        hemi="left",
        view="dorsal",
        title="LH segment distance to left origin",
        colorbar=False,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        axes=axes[0],
        figure=fig,
    )
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.flat_right,
        stat_map=rh_map,
        hemi="right",
        view="dorsal",
        title="RH segment distance to left origin",
        colorbar=False,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        axes=axes[1],
        figure=fig,
    )
    if seed_info:
        coords_lh, _ = surface.load_surf_mesh(fsaverage.flat_left)
        coords_rh, _ = surface.load_surf_mesh(fsaverage.flat_right)
        if "lh" in seed_info:
            idx = int(seed_info["lh"]["anchor_index"])
            if 0 <= idx < coords_lh.shape[0]:
                pt = coords_lh[idx]
                axes[0].scatter(pt[0], pt[1], pt[2], c="red", s=40, marker="o", depthshade=False)
        if "rh" in seed_info:
            idx_global = int(seed_info["rh"]["anchor_index"])
            idx = idx_global - n_lh
            if 0 <= idx < coords_rh.shape[0]:
                pt = coords_rh[idx]
                axes[1].scatter(pt[0], pt[1], pt[2], c="red", s=40, marker="o", depthshade=False)

    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0.0, vmax=vmax))
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Segment distance to left origin")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved flat surface map to {save_path}")


def plot_hierarchy_alignment(
    ckpt_name,
    n_segments=12,
    num_splits=1,
    mode="posttransform",
    hemisphere="both",
    max_segment_distance=1000,
    min_prediction_score=None,
    output_path=None,
    dpi=300,
):
    result = hierarchy_alignment(
        ckpt_name=ckpt_name,
        n_segments=n_segments,
        num_splits=num_splits,
        mode=mode,
        hemisphere=hemisphere,
    )

    mask = result["aligned_distance"] < float(max_segment_distance)
    total = int(mask.size)
    print(
        f"Filtering voxels with aligned_distance < {max_segment_distance}: "
        f"kept {int(mask.sum())}/{total}"
    )
    if min_prediction_score is not None:
        score_mask = result["aligned_score"] > float(min_prediction_score)
        mask &= score_mask
        print(
            f"Filtering voxels with prediction score > {float(min_prediction_score):.3f}: "
            f"kept {int(mask.sum())}/{total}"
        )

    kept = int(mask.sum())
    if kept < 2:
        raise ValueError(
            "Need at least 2 points after filtering, "
            f"but got {kept}/{total}."
        )

    x = result["aligned_distance"][mask]
    y = result["geodesic"][mask]
    color = result["aligned_score"][mask]

    # Holistic linear fit across all retained voxels.
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.array([x.min(), x.max()], dtype=np.float64)
    fit_y = slope * fit_x + intercept

    # Pearson correlation between aligned distance and geodesic.
    corr, pval = pearsonr(x, y)
    corr = float(corr)
    pval = float(pval)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    scatter = ax.scatter(
        x,
        y,
        c=color,
        cmap="cividis",
        s=20,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.2,
    )
    ax.plot(
        fit_x,
        fit_y,
        color="#d62728",
        linewidth=2.0,
        alpha=0.95,
        label=f"Linear fit: y={slope:.2f}x+{intercept:.2f}",
    )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Decoding score (r)")

    ax.set_xlabel("Posterior–anterior position (model sheet segment, mm)")
    ax.set_ylabel("Posterior–anterior position (human cortex voxel, mm)")
    ax.text(
        0.02,
        0.98,
        f"Pearson r = {corr:.3f}\np = {pval:.3e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    print(f"Pearson correlation (aligned distance vs geodesic): r={corr:.6f}, p={pval:.6e}")

    if output_path is None:
        ckpt_tag = ckpt_name.replace("/", "_").replace(".", "_")
        output_path = (
            PLOTS_DIR
            / f"plot_hierarchy_alignment_{mode}_{hemisphere}_N{result['num_segments']}_{ckpt_tag}.svg"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    flatmap_path = output_path.with_name(f"{output_path.stem}_fsaverage5_flat{output_path.suffix}")
    _plot_segment_distance_flatmap(
        aligned_distance_full=result["aligned_distance_full"],
        save_path=flatmap_path,
        dpi=dpi,
        seed_info=result.get("seed_info"),
    )

    return fig, ax, result


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot NSD high-level voxel hierarchy alignment.")
    parser.add_argument("--ckpt-name", type=str, default=MODEL_CKPT, help="Checkpoint name.")
    parser.add_argument("--n-segments", type=int, default=6, help="Number of x-axis segments.")
    parser.add_argument("--num-splits", type=int, default=1, help="Number of random splits.")
    parser.add_argument(
        "--mode",
        type=str,
        default="posttransform",
        choices=["pretransform", "posttransform"],
        help="Use pre- or post-transform features.",
    )
    parser.add_argument(
        "--hemisphere",
        type=str,
        default="both",
        choices=["both", "lh", "rh"],
        help="Hemisphere(s) to include for regression and plotting.",
    )
    parser.add_argument("--output-path", type=str, default=None, help="Output figure path.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure dpi.")
    parser.add_argument(
        "--max-segment-distance",
        type=float,
        default=1000.0,
        help="Only keep voxels with aligned segment distance to left origin below this threshold.",
    )
    parser.add_argument(
        "--min-prediction-score",
        type=float,
        default=0.5,
        help="Only keep voxels with prediction/decoding score above this threshold.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    plot_hierarchy_alignment(
        ckpt_name=args.ckpt_name,
        n_segments=args.n_segments,
        num_splits=args.num_splits,
        mode=args.mode,
        hemisphere=args.hemisphere,
        max_segment_distance=args.max_segment_distance,
        min_prediction_score=args.min_prediction_score,
        output_path=args.output_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
