import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from config import PLOTS_DIR
from utils import cached
from models import vit_transform
from validate import load_transformed_model
from validate.smoothness import NSD_HIGH
from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data import get_data_loader

from .common import MODEL_CKPT


def _flatten_features(features):
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    if features.ndim == 5:
        b, t, c, h, w = features.shape
        features = features.reshape(b * t, c * h * w)
    elif features.ndim == 4:
        b, c, h, w = features.shape
        features = features.reshape(b, c * h * w)
    elif features.ndim == 3:
        b, t, v = features.shape
        features = features.reshape(b * t, v)
    elif features.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    return features


def _normalize_features(features):
    features = features - features.mean(axis=1, keepdims=True)
    denom = features.std(axis=1, keepdims=True) + 1e-10
    return features / denom


def _compute_binned_similarity(features, distances, n_bins=11, max_distance=None):
    triu = np.triu_indices_from(distances, k=1)
    dist_flat = distances[triu]
    sim_flat = (features @ features.T)[triu] / features.shape[1]

    finite = np.isfinite(dist_flat)
    if max_distance is not None:
        keep = finite & (dist_flat <= max_distance)
    else:
        keep = finite
    if keep is not None:
        dist_flat = dist_flat[keep]
        sim_flat = sim_flat[keep]

    if dist_flat.size == 0:
        return None, None, None, None

    bins = np.linspace(dist_flat.min(), dist_flat.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(dist_flat, bins)
    means = []
    stds = []
    counts = []
    for i in range(1, len(bins)):
        vals = sim_flat[bin_indices == i]
        if vals.size == 0:
            means.append(np.nan)
            stds.append(np.nan)
            counts.append(0)
        else:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            counts.append(int(vals.size))
    return bin_centers, np.array(means), np.array(stds), np.array(counts)


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


def _geodesic_distances(coords, faces, sample_indices):
    graph = _build_mesh_graph(coords, faces)
    distances = dijkstra(graph, indices=sample_indices, directed=False)
    return distances[:, sample_indices]


def _cached_geodesic_distances_high(sample_indices, seed):
    import hashlib

    sample_bytes = np.asarray(sample_indices, dtype=np.int64).tobytes()
    key = hashlib.md5(sample_bytes + str(seed).encode()).hexdigest()[:8]
    cache_name = f"fsaverage5_geodesic_distances_high_{key}"

    @cached(cache_name, persistent=True)
    def _compute(sample_indices, seed):
        from nilearn import datasets, surface

        fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
        coords_lh, faces_lh = surface.load_surf_mesh(fsaverage.pial_left)
        coords_rh, faces_rh = surface.load_surf_mesh(fsaverage.pial_right)

        lh_count = coords_lh.shape[0]
        sample_indices = np.asarray(sample_indices)
        lh_sample = sample_indices[sample_indices < lh_count]
        rh_sample = sample_indices[sample_indices >= lh_count] - lh_count

        dist_lh = None
        dist_rh = None
        if lh_sample.size > 0:
            dist_lh = _geodesic_distances(coords_lh, faces_lh, lh_sample)
        if rh_sample.size > 0:
            dist_rh = _geodesic_distances(coords_rh, faces_rh, rh_sample)

        total = sample_indices.size
        dist = np.full((total, total), np.inf, dtype=np.float64)
        lh_idx = np.where(sample_indices < lh_count)[0]
        rh_idx = np.where(sample_indices >= lh_count)[0]
        if dist_lh is not None:
            dist[np.ix_(lh_idx, lh_idx)] = dist_lh
        if dist_rh is not None:
            dist[np.ix_(rh_idx, rh_idx)] = dist_rh
        return dist

    return _compute(sample_indices, seed)


def _collect_fmri_batches(batch_size=16, max_batches=None, seed=42):
    trainset, valset, testset = get_compilation(
        vit_transform, type="clip", ratios=(0.8, 0.1, 0.1), seed=seed
    )
    loader = get_data_loader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    fmri_batches = []
    stim_batches = []
    for i, batch in enumerate(tqdm(loader, desc="Loading fMRI batches")):
        stim, response = batch[0], batch[1]
        stim_batches.append(stim)
        fmri_batches.append(response)
        if max_batches is not None and i + 1 >= max_batches:
            break
    return stim_batches, fmri_batches


def _prepare_human_fmri(fmri_batches):
    responses = []
    for response in fmri_batches:
        if torch.is_tensor(response):
            response = response.cpu().numpy()
        if response.ndim == 3 and response.shape[1] == 1:
            response = response[:, 0, :]
        responses.append(response)
    responses = np.concatenate(responses, axis=0)
    responses = responses[:, NSD_HIGH]
    return responses


def _prepare_model_fmri_features(
    stim_batches,
    ckpt_name,
    fwhm_mm=2.0,
    resolution_mm=1.0,
    device=None,
    do_transform=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    feats = []
    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        for stim in tqdm(stim_batches, desc="Extracting model features"):
            stim = stim.to(device)
            with torch.no_grad():
                outputs, _ = model(stim, do_transform=do_transform)
            feats.append(outputs[0].cpu())
        positions = model.smoothed_layer_positions[0].coordinates.cpu().numpy()

    feats = torch.cat(feats, dim=0)
    return feats, positions


def plot_wiring_cost_fmri(
    ckpt_name,
    save_path=PLOTS_DIR / "plot_wiring_cost_fmri.svg",
    n_bins=11,
    subsample=2000,
    max_distance=65,
    seed=42,
    max_batches=None,
):
    np.random.seed(seed)

    stim_batches, fmri_batches = _collect_fmri_batches(max_batches=max_batches, seed=seed)
    human_responses = _prepare_human_fmri(fmri_batches)

    model_features, model_positions = _prepare_model_fmri_features(
        stim_batches, ckpt_name, fwhm_mm=2.0, resolution_mm=1.0, do_transform=True
    )
    pretransform_features, pretransform_positions = _prepare_model_fmri_features(
        stim_batches, ckpt_name, fwhm_mm=2.0, resolution_mm=1.0, do_transform=False
    )

    model_flat = _flatten_features(model_features)
    pretransform_flat = _flatten_features(pretransform_features)
    human_flat = _flatten_features(human_responses)

    model_units = model_flat.shape[1]
    pretransform_units = pretransform_flat.shape[1]
    human_units = human_flat.shape[1]
    if model_units == 0 or pretransform_units == 0 or human_units == 0:
        raise ValueError("Empty model or human features after preprocessing.")

    model_indices = np.arange(model_units)
    if subsample is not None and model_units > subsample:
        model_indices = np.random.choice(model_units, subsample, replace=False)
    model_flat = model_flat[:, model_indices]
    model_positions = model_positions[model_indices]

    pretransform_indices = np.arange(pretransform_units)
    if subsample is not None and pretransform_units > subsample:
        pretransform_indices = np.random.choice(pretransform_units, subsample, replace=False)
    pretransform_flat = pretransform_flat[:, pretransform_indices]
    pretransform_positions = pretransform_positions[pretransform_indices]

    human_indices = np.arange(human_units)
    if subsample is not None and human_units > subsample:
        human_indices = np.random.choice(human_units, subsample, replace=False)
    human_flat = human_flat[:, human_indices]

    model_features_norm = _normalize_features(model_flat.T)
    model_dist = np.linalg.norm(
        model_positions[:, None, :] - model_positions[None, :, :], axis=-1
    )
    model_bins, model_mean, model_std, model_counts = _compute_binned_similarity(
        model_features_norm, model_dist, n_bins=n_bins, max_distance=max_distance
    )

    pretransform_features_norm = _normalize_features(pretransform_flat.T)
    pretransform_dist = np.linalg.norm(
        pretransform_positions[:, None, :] - pretransform_positions[None, :, :], axis=-1
    )
    pretransform_bins, pretransform_mean, pretransform_std, pretransform_counts = _compute_binned_similarity(
        pretransform_features_norm, pretransform_dist, n_bins=n_bins, max_distance=max_distance
    )

    human_features_norm = _normalize_features(human_flat.T)
    human_indices_global = np.where(NSD_HIGH)[0][human_indices]
    human_dist = _cached_geodesic_distances_high(human_indices_global, seed=seed)
    human_bins, human_mean, human_std, human_counts = _compute_binned_similarity(
        human_features_norm, human_dist, n_bins=n_bins, max_distance=max_distance
    )

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if pretransform_bins is not None:
        pretransform_ci = pretransform_std
        pretransform_mask = (
            np.isfinite(pretransform_bins)
            & np.isfinite(pretransform_mean)
            & np.isfinite(pretransform_ci)
        )
        ax.plot(
            pretransform_bins[pretransform_mask],
            pretransform_mean[pretransform_mask],
            color="#9AA0A6",
            linestyle="-.",
            label="Initial model fMRI",
            linewidth=2.0,
        )
        ax.fill_between(
            pretransform_bins[pretransform_mask],
            (pretransform_mean - pretransform_ci)[pretransform_mask],
            (pretransform_mean + pretransform_ci)[pretransform_mask],
            color="#9AA0A6",
            alpha=0.2,
            linewidth=0,
        )
    if human_bins is not None:
        human_ci = human_std
        human_mask = np.isfinite(human_bins) & np.isfinite(human_mean) & np.isfinite(human_ci)
        ax.plot(
            human_bins[human_mask],
            human_mean[human_mask],
            color="black",
            linestyle="--",
            label="Human fMRI",
            linewidth=2.2,
        )
        ax.fill_between(
            human_bins[human_mask],
            (human_mean - human_ci)[human_mask],
            (human_mean + human_ci)[human_mask],
            color="black",
            alpha=0.2,
            linewidth=0,
        )
    if model_bins is not None:
        model_ci = model_std
        model_mask = np.isfinite(model_bins) & np.isfinite(model_mean) & np.isfinite(model_ci)
        ax.plot(model_bins[model_mask], model_mean[model_mask], color="#FA0022", label="Model fMRI", linewidth=2.2)
        ax.fill_between(
            model_bins[model_mask],
            (model_mean - model_ci)[model_mask],
            (model_mean + model_ci)[model_mask],
            color="#FA0022",
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xlabel("Cortical distance (mm)")
    ax.set_ylabel("Response correlation")
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight", facecolor="none")
    print(f"Saved to {save_path}")
    plt.close()


def main():
    plot_wiring_cost_fmri(MODEL_CKPT)


if __name__ == "__main__":
    main()
