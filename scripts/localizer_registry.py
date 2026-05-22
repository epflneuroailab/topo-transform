from dataclasses import dataclass
from typing import Optional

from .common import LOCALIZER_BIOMOTION_T_THRESHOLD
from .common import LOCALIZER_FLOW_T_THRESHOLD


@dataclass(frozen=True)
class HumanRoiSpec:
    source: str
    region: object


@dataclass(frozen=True)
class RoiSpec:
    result_key: Optional[str] = None
    human: Optional[HumanRoiSpec] = None
    threshold_group: str = "default"


ROI_SPECS = {
    "face": RoiSpec(
        result_key="face",
        human=HumanRoiSpec("visf", "faces"),
    ),
    "face-dynamic": RoiSpec(result_key="Faces_localizer"),
    "place": RoiSpec(
        result_key="place",
        human=HumanRoiSpec("visf", "places"),
    ),
    "place-dynamic": RoiSpec(result_key="Scenes_localizer"),
    "body": RoiSpec(
        result_key="body",
        human=HumanRoiSpec("visf", "bodies"),
    ),
    "body-dynamic": RoiSpec(result_key="Bodies_localizer"),
    "object": RoiSpec(result_key="object"),
    "character": RoiSpec(human=HumanRoiSpec("visf", "characters")),
    "car": RoiSpec(result_key="car"),
    "instrument": RoiSpec(result_key="instrument"),
    "Faces_static": RoiSpec(result_key="Faces_static"),
    "Bodies_static": RoiSpec(result_key="Bodies_static"),
    "Scenes_static": RoiSpec(result_key="Scenes_static"),
    "Faces_moving": RoiSpec(result_key="Faces_moving"),
    "Bodies_moving": RoiSpec(result_key="Bodies_moving"),
    "Scenes_moving": RoiSpec(result_key="Scenes_moving"),
    "Faces_static_localizer": RoiSpec(result_key="Faces_static_localizer"),
    "Bodies_static_localizer": RoiSpec(result_key="Bodies_static_localizer"),
    "Scenes_static_localizer": RoiSpec(result_key="Scenes_static_localizer"),
    "Faces_moving_localizer": RoiSpec(result_key="Faces_moving_localizer"),
    "Bodies_moving_localizer": RoiSpec(result_key="Bodies_moving_localizer"),
    "Scenes_moving_localizer": RoiSpec(result_key="Scenes_moving_localizer"),
    "face-detailed": RoiSpec(result_key="face-detailed"),
    "body-detailed": RoiSpec(result_key="body-detailed"),
    "place-detailed": RoiSpec(result_key="place-detailed"),
    "car-detailed": RoiSpec(result_key="car-detailed"),
    "instrument-detailed": RoiSpec(result_key="instrument-detailed"),
    "character-detailed": RoiSpec(result_key="character-detailed"),
    "v6": RoiSpec(
        result_key="V6-enhanced",
        human=HumanRoiSpec("glasser", ["V6"]),
        threshold_group="flow",
    ),
    "v6-enhanced": RoiSpec(
        result_key="V6-enhanced",
        human=HumanRoiSpec("glasser", ["V6"]),
        threshold_group="flow",
    ),
    "psts": RoiSpec(
        result_key="pSTS-enhanced",
        human=HumanRoiSpec("glasser", ["TPOJ1"]),
        threshold_group="biomotion",
    ),
    "psts-enhanced": RoiSpec(
        result_key="pSTS-enhanced",
        human=HumanRoiSpec("glasser", ["TPOJ1"]),
        threshold_group="biomotion",
    ),
    "mt": RoiSpec(
        result_key="MT-Huk",
        human=HumanRoiSpec("visf", "hMT"),
        threshold_group="flow",
    ),
    "v6-direct": RoiSpec(result_key="V6", threshold_group="flow"),
    "psts-direct": RoiSpec(result_key="pSTS", threshold_group="biomotion"),
    "mt-direct": RoiSpec(result_key="MT-Huk", threshold_group="flow"),
    "V6": RoiSpec(result_key="V6", threshold_group="flow"),
    "V6-enhanced": RoiSpec(result_key="V6-enhanced", threshold_group="flow"),
    "pSTS": RoiSpec(result_key="pSTS", threshold_group="biomotion"),
    "pSTS-enhanced": RoiSpec(result_key="pSTS-enhanced", threshold_group="biomotion"),
    "MT-Huk": RoiSpec(result_key="MT-Huk", threshold_group="flow"),
}


def _get_roi_spec(roi_name):
    try:
        return ROI_SPECS[roi_name]
    except KeyError as exc:
        raise ValueError(f"Unknown roi: {roi_name}") from exc


def get_localizer_result_key(roi_name):
    result_key = _get_roi_spec(roi_name).result_key
    if result_key is None:
        raise ValueError(f"Unknown roi: {roi_name}")
    return result_key


def get_roi_t_threshold(roi_name, default_threshold):
    threshold_group = _get_roi_spec(roi_name).threshold_group
    if threshold_group == "flow":
        return LOCALIZER_FLOW_T_THRESHOLD
    if threshold_group == "biomotion":
        return LOCALIZER_BIOMOTION_T_THRESHOLD
    return default_threshold


def get_human_localizer_mask(roi_name):
    from validate.rois import glasser
    from validate.rois import visf

    human = _get_roi_spec(roi_name.lower()).human
    if human is None:
        raise ValueError(f"Unknown roi: {roi_name}")
    if human.source == "visf":
        return visf.get_region_voxels(human.region)
    return glasser.get_region_voxels(human.region)


__all__ = [
    "ROI_SPECS",
    "RoiSpec",
    "get_human_localizer_mask",
    "get_localizer_result_key",
    "get_roi_t_threshold",
]
