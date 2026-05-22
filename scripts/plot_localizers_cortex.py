import os
import numpy as np
from nilearn import datasets, surface
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import PLOTS_DIR
from .common import all_roi_colors
from .get_localizers import get_localizer_human

ROIS = ["face", "body", "place", "mt", "v6", "psts"]


def _build_colors():
    n_vertices = 20484
    colors = np.ones((n_vertices, 4)) * [0.5, 0.5, 0.5, 1.0]
    masks = get_localizer_human(ROIS)
    for mask, roi in zip(masks, ROIS):
        _, color = all_roi_colors[roi]
        colors[mask] = color + (1.0,)
    return colors


def main():
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")
    colors = _build_colors()

    mesh_left = surface.load_surf_mesh(fsavg["flat_left"])
    mesh_right = surface.load_surf_mesh(fsavg["flat_right"])

    coords_left, faces_left = mesh_left[0], mesh_left[1]
    coords_right, faces_right = mesh_right[0], mesh_right[1]
    coords_right += np.array([0, 300, 0])

    left_colors = colors[:10242]
    right_colors = colors[10242:]

    triangles_left = coords_left[faces_left]
    triangles_right = coords_right[faces_right]

    face_colors_left = left_colors[faces_left[:, 0]]
    face_colors_right = right_colors[faces_right[:, 0]]

    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh_plot_left = Poly3DCollection(
        triangles_left,
        facecolors=face_colors_left,
        edgecolors="none",
        linewidth=0,
        antialiased=False,
    )
    mesh_plot_left.set_facecolor(face_colors_left)
    ax.add_collection3d(mesh_plot_left)

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=90, azim=0)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])

    save_path = os.path.join(PLOTS_DIR, "localizers_cortex.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Saved localizers cortex visualization to", save_path)


if __name__ == "__main__":
    main()
