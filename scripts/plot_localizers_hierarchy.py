import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from config import PLOTS_DIR
from utils import cached

from .common import (
    MODEL_CKPT,
    LOCALIZER_P_THRESHOLD,
    LOCALIZER_T_THRESHOLD,
    all_roi_colors,
    roi_groups,
)
from .get_localizers import localizers, get_localizer_human
from .localizer_registry import get_roi_t_threshold
from .plot_utils import ensure_dir, savefig, to_numpy
from .analysis_utils import CKPT_GROUPS


def _counts_on_grid(values, grid):
    if values.size == 0:
        return np.zeros(grid.size, dtype=int)
    idx = np.searchsorted(grid, values)
    valid = (idx >= 0) & (idx < grid.size) & (grid[idx] == values)
    if not np.all(valid):
        index_map = {v: i for i, v in enumerate(grid)}
        idx = np.array([index_map[v] for v in values], dtype=int)
    return np.bincount(idx, minlength=grid.size)


def _collect_model_roi_xs(roi_name, t_vals_dict, p_vals_dict, layer_positions, p_threshold, t_threshold):
    xs = []
    p_vals_list = p_vals_dict[roi_name]
    t_vals_list = t_vals_dict[roi_name]
    for layer_idx, (p_vals, t_vals) in enumerate(zip(p_vals_list, t_vals_list)):
        pos = to_numpy(layer_positions[layer_idx])
        t_threshold_used = get_roi_t_threshold(roi_name, t_threshold)
        mask = (p_vals.flatten() < p_threshold) & (t_vals.flatten() > t_threshold_used)
        if np.any(mask):
            xs.append(pos[mask, 0])
    if not xs:
        return np.array([], dtype=float)
    return np.concatenate(xs, axis=0)


def _model_hierarchy_counts(t_vals_dict, p_vals_dict, layer_positions, rois):
    all_x = np.concatenate([to_numpy(lp)[:, 0] for lp in layer_positions], axis=0)
    all_x = np.round(all_x, 6)
    x_grid = np.unique(all_x)

    counts = {}
    for roi_name in rois:
        xs = _collect_model_roi_xs(
            roi_name,
            t_vals_dict,
            p_vals_dict,
            layer_positions,
            LOCALIZER_P_THRESHOLD,
            LOCALIZER_T_THRESHOLD,
        )
        xs = np.round(xs, 6)
        counts[roi_name] = _counts_on_grid(xs, x_grid)

    return x_grid, counts


def _plot_counts(x_vals, counts_by_roi, rois, title, xlabel, ylabel, save_path, dpi=300):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for roi_name in rois:
        display_name, color = all_roi_colors[roi_name]
        ax.plot(x_vals, counts_by_roi[roi_name], color=color, linewidth=2.0, label=display_name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    out = savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out}")
    return out


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
    n = coords.shape[0]
    graph = coo_matrix((dists, (edges[:, 0], edges[:, 1])), shape=(n, n))
    return graph.tocsr()


def _geodesic_distances_from_seed(coords, faces, seed_idx):
    graph = _build_mesh_graph(coords, faces)
    distances = dijkstra(graph, indices=[seed_idx], directed=False)
    return distances[0]


def _v1_center_index(coords, v1_indices):
    centroid = coords[v1_indices].mean(axis=0)
    deltas = coords[v1_indices] - centroid
    return v1_indices[np.argmin(np.sum(deltas ** 2, axis=1))]


def _cached_v1_geodesic_distances():
    @cached("fsaverage5_v1_geodesic_distances", persistent=True)
    def _compute():
        from nilearn import datasets, surface
        from validate.rois import glasser

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

        if v1_lh.size > 0:
            center_lh = _v1_center_index(coords_lh, v1_lh)
            distances[:lh_count] = _geodesic_distances_from_seed(coords_lh, faces_lh, center_lh)

        if v1_rh.size > 0:
            center_rh = _v1_center_index(coords_rh, v1_rh)
            distances[lh_count:] = _geodesic_distances_from_seed(coords_rh, faces_rh, center_rh)

        return distances

    return _compute()


def _human_hierarchy_counts(rois, n_bins=40):
    distances = _cached_v1_geodesic_distances()
    finite = np.isfinite(distances)
    dist_finite = distances[finite]
    bins = np.linspace(dist_finite.min(), dist_finite.max(), n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    counts = {}
    masks = get_localizer_human(rois)
    for roi_name, mask in zip(rois, masks):
        roi_dist = distances[mask & finite]
        counts[roi_name] = np.histogram(roi_dist, bins=bins)[0]
    return centers, counts


def plot_model_hierarchy(t_vals_dict, p_vals_dict, layer_positions, store_dir):
    ensure_dir(store_dir)

    floc_rois = roi_groups["fLoc2"]
    motion_rois = roi_groups["motion3"]

    x_grid, floc_counts = _model_hierarchy_counts(
        t_vals_dict, p_vals_dict, layer_positions, floc_rois
    )
    _, motion_counts = _model_hierarchy_counts(
        t_vals_dict, p_vals_dict, layer_positions, motion_rois
    )

    _plot_counts(
        x_grid,
        floc_counts,
        floc_rois,
        title="fLoc2 hierarchy (model)",
        xlabel="Hierarchy (left to right)",
        ylabel="Voxel count",
        save_path=f"{store_dir}/floc2_hierarchy.svg",
    )

    _plot_counts(
        x_grid,
        motion_counts,
        motion_rois,
        title="motion3 hierarchy (model)",
        xlabel="Hierarchy (left to right)",
        ylabel="Voxel count",
        save_path=f"{store_dir}/motion3_hierarchy.svg",
    )


def plot_human_hierarchy(store_dir, n_bins=40):
    ensure_dir(store_dir)

    floc_rois = roi_groups["fLoc2"]
    motion_rois = roi_groups["motion3"]

    x_bins, floc_counts = _human_hierarchy_counts(floc_rois, n_bins=n_bins)
    _, motion_counts = _human_hierarchy_counts(motion_rois, n_bins=n_bins)

    _plot_counts(
        x_bins,
        floc_counts,
        floc_rois,
        title="fLoc2 hierarchy (human)",
        xlabel="Geodesic distance to V1 center (mm)",
        ylabel="Voxel count",
        save_path=f"{store_dir}/floc2_hierarchy.svg",
    )

    _plot_counts(
        x_bins,
        motion_counts,
        motion_rois,
        title="motion3 hierarchy (human)",
        xlabel="Geodesic distance to V1 center (mm)",
        ylabel="Voxel count",
        save_path=f"{store_dir}/motion3_hierarchy.svg",
    )


if __name__ == "__main__":
    base_store_dir = PLOTS_DIR / "localizers_hierarchy"
    ensure_dir(base_store_dir)

    for group_name, ckpt_list in CKPT_GROUPS.items():
        if not ckpt_list:
            continue
        exemplar_ckpt = ckpt_list[0]
        if group_name == "MODEL":
            exemplar_ckpt = MODEL_CKPT
        group_store_dir = base_store_dir / group_name.lower()
        ensure_dir(group_store_dir)
        t_vals_dict, p_vals_dict, layer_positions = localizers(exemplar_ckpt, ret_merged=True)
        plot_model_hierarchy(t_vals_dict, p_vals_dict, layer_positions, group_store_dir)

    human_store_dir = base_store_dir / "human"
    plot_human_hierarchy(human_store_dir)
