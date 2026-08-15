from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import torch


_SVG_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SVG_RECT_PATH = re.compile(
    rf'<path d="(?P<d>\s*M\s+{_SVG_NUMBER}\s+{_SVG_NUMBER}'
    rf'(?:\s+L\s+{_SVG_NUMBER}\s+{_SVG_NUMBER}){{3}}\s+z\s*)"'
    rf'(?P<attrs>[^>]*)/>',
    re.DOTALL,
)
_SVG_POINT = re.compile(rf"[ML]\s+({_SVG_NUMBER})\s+({_SVG_NUMBER})")
_SVG_CLIPPED_RECT = re.compile(
    r'(<rect\b[^>]*?)\s+clip-path="url\(#[^)]+\)"([^>]*/>)',
    re.DOTALL,
)
_SVG_CLIP_REFERENCE = re.compile(r'\s+clip-path="url\(#[^)]+\)"')
_SVG_CLIP_DEFINITION = re.compile(r'\s*<clipPath\b[^>]*>.*?</clipPath>', re.DOTALL)
_UNCLIPPED_COMPONENT_SVGS = {
    "localizer_tvals_comparison.svg",
    "localizer_motion_mae_comparison.svg",
}


def convert_svg_rect_paths(path):
    """Convert Matplotlib's axis-aligned rectangle paths to native SVG rects."""
    svg_path = Path(path)
    if svg_path.suffix.lower() != ".svg" or not svg_path.exists():
        return 0

    source = svg_path.read_text(encoding="utf-8")
    converted = 0

    def _replace(match):
        nonlocal converted
        points = [(float(x), float(y)) for x, y in _SVG_POINT.findall(match.group("d"))]
        if len(points) != 4:
            return match.group(0)

        xs = sorted(set(x for x, _ in points))
        ys = sorted(set(y for _, y in points))
        if len(xs) != 2 or len(ys) != 2:
            return match.group(0)
        if set(points) != {(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[0]), (xs[1], ys[1])}:
            return match.group(0)
        for start, end in zip(points, points[1:] + points[:1]):
            if start[0] != end[0] and start[1] != end[1]:
                return match.group(0)

        converted += 1
        number = lambda value: format(value, ".15g")
        attrs = re.sub(r'\s+clip-path="url\(#[^)]+\)"', "", match.group("attrs"))
        return (
            f'<rect x="{number(xs[0])}" y="{number(ys[0])}" '
            f'width="{number(xs[1] - xs[0])}" height="{number(ys[1] - ys[0])}"'
            f'{attrs}/>'
        )

    updated = _SVG_RECT_PATH.sub(_replace, source)
    updated, unclipped = _SVG_CLIPPED_RECT.subn(r"\1\2", updated)
    if converted or unclipped:
        svg_path.write_text(updated, encoding="utf-8")
    return converted + unclipped


def remove_svg_clip_paths(path):
    """Remove axes-sized clipping masks from fully in-bounds bar-chart components."""
    svg_path = Path(path)
    if svg_path.suffix.lower() != ".svg" or not svg_path.exists():
        return 0

    source = svg_path.read_text(encoding="utf-8")
    updated, removed = _SVG_CLIP_REFERENCE.subn("", source)
    if removed and "clip-path=" not in updated:
        updated = _SVG_CLIP_DEFINITION.sub("", updated)
    if removed:
        svg_path.write_text(updated, encoding="utf-8")
    return removed


def normalize_publication_svg(path):
    """Normalize editable SVG primitives and known in-bounds bar-chart groups."""
    svg_path = Path(path)
    changed = convert_svg_rect_paths(svg_path)
    if svg_path.name in _UNCLIPPED_COMPONENT_SVGS:
        changed += remove_svg_clip_paths(svg_path)
    return changed


def ensure_dir(path):
    if path is None:
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def savefig(path, **kwargs):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = plt.savefig(output_path, **kwargs)
    targets = generated if isinstance(generated, (list, tuple)) else [output_path]
    for target in targets:
        normalize_publication_svg(target)
    plt.close()
    return output_path


def to_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)
