import os
import pickle
import sys

import numpy as np
import cortex

from config import PLOTS_DIR
from .common import all_roi_colors
from .get_localizers import get_localizer_human
from neuroparc.atlas import Atlas

ROIS = ["face", "body", "place", "mt", "v6", "psts"]
N_VERTICES = 163842 * 2


def _fsaverage7_masks(rois):
    masks = get_localizer_human(rois)
    for i, mask in enumerate(masks):
        surface = Atlas("fsaverage", mask.astype(float))
        masks[i] = surface.label_surface("fsaverage7")
    return masks


def _roi_color_data(rois):
    masks = _fsaverage7_masks(rois)
    vertex_colors = np.ones((N_VERTICES, 3)) * np.nan
    for mask, roi in zip(masks, rois):
        _, color = all_roi_colors[roi]
        vertex_colors[mask > 0.5] = color

    return {
        "roi_colors": {
            "left": vertex_colors[:163842, :],
            "right": vertex_colors[163842:, :],
        }
    }


def _save_debug_pickle(data, path="temp_roi_colors_fsaverage7.pkl"):
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def _load_debug_pickle(path="temp_roi_colors_fsaverage7.pkl"):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _build_cortex_data(data):
    subject = "fsaverage"
    if subject not in cortex.db.subjects:
        cortex.db.add_subject_from_fsaverage(subject)

    cortex_data = {}
    for name, hemi_dict in data.items():
        cortex_data[name] = {}
        for hemi in ["left", "right"]:
            rgb = hemi_dict[hemi]
            cortex_data[name][hemi] = cortex.VertexRGB(
                red=rgb[:, 0],
                green=rgb[:, 1],
                blue=rgb[:, 2],
                subject=subject,
                xfmname="identity",
                hemi=hemi,
            )
    return cortex_data


def _save_cortex_screenshots(cortex_data):
    for name, hemi_dict in cortex_data.items():
        view = cortex.web.show_hemi(hemi_dict["left"])
        os.makedirs(PLOTS_DIR, exist_ok=True)
        cortex.webgl.screenshot(view, os.path.join(PLOTS_DIR, f"{name}_cortex.png"))


def main():
    data = _roi_color_data(ROIS)
    _save_debug_pickle(data)
    sys.exit()
    data = _load_debug_pickle()
    cortex_data = _build_cortex_data(data)
    _save_cortex_screenshots(cortex_data)


if __name__ == "__main__":
    main()
