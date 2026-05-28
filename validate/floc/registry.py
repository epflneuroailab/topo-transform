from dataclasses import dataclass
from typing import Callable
from typing import Optional

from .afraz import localize_afraz
from .categories import localize_categories
from .konkle import localize_konkle
from .motion import localize_motion
from .pitcher import localize_pitcher
from .pitcher import localize_pitcher_human
from .psts import localize_psts
from .robert import localize_robert
from .robert import localize_robert_human
from .temporal import localize_temporal
from .v6 import localize_v6


@dataclass(frozen=True)
class FlocTestSpec:
    name: str
    model_runner: Callable
    human_runner: Optional[Callable] = None
    include_in_validate: bool = False
    include_in_localizers: bool = False


def _category_runner(dataset_name):
    def runner(model, transform, **kwargs):
        return localize_categories(model, transform, dataset_name=dataset_name, **kwargs)

    return runner


def _simple_runner(localizer_fn):
    def runner(model, transform, **kwargs):
        return localizer_fn(model, transform, **kwargs)

    return runner


def _duration_3s_runner(label, localizer_fn):
    def runner(model, transform, video_fps, **kwargs):
        print(f"For {label} dataset, using duration = 3 seconds")
        kwargs.pop("frames_per_video", None)
        return localizer_fn(
            model,
            transform,
            video_fps=video_fps,
            frames_per_video=video_fps * 3,
            **kwargs,
        )

    return runner


FLOC_TESTS = {
    "vpnl": FlocTestSpec(
        name="vpnl",
        model_runner=_category_runner("vpnl"),
        include_in_validate=True,
        include_in_localizers=True,
    ),
    "kanwisher": FlocTestSpec(
        name="kanwisher",
        model_runner=_category_runner("kanwisher"),
        include_in_validate=True,
        include_in_localizers=True,
    ),
    "vpnl_detailed": FlocTestSpec(
        name="vpnl_detailed",
        model_runner=_category_runner("vpnl_detailed"),
        include_in_localizers=False,
    ),
    "vpnl_detail_classes": FlocTestSpec(
        name="vpnl_detail_classes",
        model_runner=_category_runner("vpnl_detail_classes"),
        include_in_localizers=False,
    ),
    "motion": FlocTestSpec(
        name="motion",
        model_runner=_simple_runner(localize_motion),
    ),
    "pitzalis": FlocTestSpec(
        name="pitzalis",
        model_runner=_simple_runner(localize_v6),
        include_in_validate=True,
        include_in_localizers=True,
    ),
    "biomotion": FlocTestSpec(
        name="biomotion",
        model_runner=_simple_runner(localize_psts),
        include_in_validate=True,
        include_in_localizers=True,
    ),
    "temporal": FlocTestSpec(
        name="temporal",
        model_runner=_simple_runner(localize_temporal),
    ),
    "pitcher": FlocTestSpec(
        name="pitcher",
        model_runner=_duration_3s_runner("Pitcher", localize_pitcher),
        human_runner=localize_pitcher_human,
        include_in_validate=True,
        include_in_localizers=True,
    ),
    "robert": FlocTestSpec(
        name="robert",
        model_runner=_duration_3s_runner("Robert", localize_robert),
        human_runner=localize_robert_human,
        include_in_validate=True,
        include_in_localizers=True,
    ),
    "afraz": FlocTestSpec(
        name="afraz",
        model_runner=_simple_runner(localize_afraz),
    ),
    "konkle": FlocTestSpec(
        name="konkle",
        model_runner=_simple_runner(localize_konkle),
        include_in_localizers=True,
    ),
}

FLOC_DATASETS = [name for name, spec in FLOC_TESTS.items() if spec.include_in_validate]
LOCALIZER_DATASETS = [name for name, spec in FLOC_TESTS.items() if spec.include_in_localizers]


def get_floc_test_spec(dataset_name):
    try:
        return FLOC_TESTS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_name: {dataset_name}") from exc


def get_model_localizer_runner(dataset_name):
    return get_floc_test_spec(dataset_name).model_runner


def get_human_localizer_runner(dataset_name):
    human_runner = get_floc_test_spec(dataset_name).human_runner
    if human_runner is None:
        raise ValueError(f"Unknown dataset_name for human: {dataset_name}")
    return human_runner


__all__ = [
    "FLOC_DATASETS",
    "FLOC_TESTS",
    "FlocTestSpec",
    "LOCALIZER_DATASETS",
    "get_floc_test_spec",
    "get_human_localizer_runner",
    "get_model_localizer_runner",
]
