from config import PLOTS_DIR
import numpy as np
import matplotlib.pyplot as plt

from .common import MODEL_CKPT
from .get_localizers import get_localizer_model
from .get_localizers import localizers
from .plot_utils import ensure_dir, savefig


def main():
    store_dir = ensure_dir(PLOTS_DIR)

    rois = [
        'face',
        'body',
        'place',
        # 'mt',
        'v6',
        'psts',
    ]

    colors = [
        (0.75, 0.00, 0.00),
        (0.00, 0.45, 0.00),
        (0.80, 0.35, 0.00),
        # (0.95, 0.20, 0.70),
        (0.0, 0.85, 0.95),
        (0.85, 0.85, 0.0),
    ]

    t_vals_dicts, p_vals_dicts, layer_positions = localizers(MODEL_CKPT, ret_merged=True)
    t_val_motion = t_vals_dicts['robert_motion'][0].flatten()
    t_val_object = t_vals_dicts['robert_static'][0].flatten()
    masks = get_localizer_model(rois, MODEL_CKPT)
    masks = [m[0].flatten() for m in masks]

    # plot a coordinate motion vs object, where the dots are colored by their roi

    plt.scatter(
        t_val_object,
        t_val_motion,
        color='gray',
        alpha=0.3,
        s=2,
    )

    for mask, roi, color in zip(masks, rois, colors):
        plt.scatter(
            t_val_object[mask],
            t_val_motion[mask],
            label=roi,
            color=color,
            alpha=1,
            s=2,
        )

    plt.xlabel('Object t-value')
    plt.ylabel('Motion t-value')
    plt.title('Model Localizer: Motion vs Object')
    plt.legend(markerscale=5)
    path = store_dir / "model_localizer_motion_vs_object.png"
    savefig(path)
    print("Saved plot to", path)


if __name__ == "__main__":
    main()
