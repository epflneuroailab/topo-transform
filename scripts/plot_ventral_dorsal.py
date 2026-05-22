from config import PLOTS_DIR
import numpy as np
import matplotlib.pyplot as plt

from .common import MODEL_CKPT
from .get_localizers import get_localizer_model


def _combine_masks(masks, rois):
    combined_mask = np.zeros_like(masks[rois[0]], dtype=bool)
    for roi in rois:
        combined_mask = combined_mask | masks[roi]
    return combined_mask


def _save_mask(mask, path):
    plt.imshow(mask)
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    ventral_rois = ["face", "body", "place"]
    dorsal_rois = ["v6", "psts"]
    dorsal_rois = ["mt"]

    mask_list = get_localizer_model(
        rois=ventral_rois + dorsal_rois,
        ckpt_name=MODEL_CKPT,
    )
    masks = {
        roi: np.array(mask, dtype=bool)
        for roi, mask in zip(ventral_rois + dorsal_rois, mask_list)
    }

    mask_ventral = _combine_masks(masks, ventral_rois)[0, 0]
    mask_dorsal = _combine_masks(masks, dorsal_rois)[0, 0]

    _save_mask(mask_ventral, PLOTS_DIR / "ventral_mask.png")
    _save_mask(mask_dorsal, PLOTS_DIR / "dorsal_mask.png")
    breakpoint()

if __name__ == "__main__":
    main()
