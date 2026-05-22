from pathlib import Path

from config import PLOTS_DIR

from .analysis_utils import CKPT_GROUPS
from .get_localizers import localizers
from .plot_localizers import plot_all_rois
from .plot_utils import ensure_dir


def _ckpt_dirname(ckpt_name):
    return Path(ckpt_name).stem


if __name__ == "__main__":
    base_store_dir = ensure_dir(PLOTS_DIR / "localizers" / "topotransform" / "seeds")

    ckpt_list = CKPT_GROUPS.get("TopoTransform", [])
    if not ckpt_list:
        raise ValueError("No checkpoints found for TopoTransform in CKPT_GROUPS.")

    for ckpt_name in ckpt_list:
        ckpt_store_dir = ensure_dir(base_store_dir / _ckpt_dirname(ckpt_name))
        t_vals_dict, p_vals_dict, layer_positions = localizers(ckpt_name, ret_merged=True)
        plot_all_rois(
            t_vals_dict,
            p_vals_dict,
            layer_positions,
            ckpt_store_dir,
        )
