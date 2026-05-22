import numpy as np


# Here we just use a heuristic configuration
# 
# in TDANN:
# RETINA 0.0475 / 2.4   = 0.0198  
# V1     1.626  / 35.75 = 0.0455  
# V2     3.977  / 35    = 0.1136  
# V4     2.545  / 22.4  = 0.1136  
# VTC    31.818 / 70.0  = 0.4545
#
# with Yash's electrode data alignment:
# blocks.[0,1] = retina
# blocks.[2,3,4,5] = V1
# blocks.[6,7] = V2
# blocks.[8,9,10,11,12,13] = V4v/d
# blocks.[14,15,16,17,18,19,20,21,22,23] = higher visual cortex

NEIGHBORHOOD_WIDTHS = {
    "retina": 0.0475,
    "V1": 1.626,
    "V2": 3.977,
    "V4": 2.545,
    "VTC": 31.818,
}
TISSUE_SIZES = {
    "retina": 2.4,
    "V1": 35.75,
    "V2": 35.0,
    "V4": 22.4,
    "VTC": 70.0,
}
VJEPA_LAYER_ASSIGNMENTS = {
    "retina": [0, 1],
    "V1": [2, 3, 4, 5],
    "V2": [6, 7],
    "V4": [8, 9, 10, 11, 12, 13],
    "VTC": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
}
INITIAL_RF_OVERLAP = 0.1

def _get_tissue_configs(layer_indices, layer_assignments, exponentially_interpolate=False, constant_rf_overlap=False, large_neighborhood=False):
    layer_tissue_sizes = []
    layer_neighborhood_widths = []
    layer_rf_overlaps = []

    num_stages = len(layer_assignments)
    areas = list(layer_assignments.keys())
    
    for s in range(num_stages):

        if s == 0:
            tissue_size_start = TISSUE_SIZES['retina']
            neighborhood_start = NEIGHBORHOOD_WIDTHS['retina']
            rf_overlap_start = INITIAL_RF_OVERLAP / 2
        else:
            area_start = areas[s - 1]
            tissue_size_start = TISSUE_SIZES[area_start]
            neighborhood_start = NEIGHBORHOOD_WIDTHS[area_start]
            rf_overlap_start = rf_overlap

        area_end = areas[s]
        layers = layer_assignments[area_end]
        tissue_size_end = TISSUE_SIZES[area_end]
        neighborhood_end = NEIGHBORHOOD_WIDTHS[area_end]
        num_layers = len(layers)
        rf_overlap_end = min(rf_overlap_start * 2, 1) if not constant_rf_overlap else rf_overlap_start

        rf_overlap = rf_overlap_end

        if exponentially_interpolate:
            tissue_sizes = [tissue_size_end] * num_layers
            neighborhood_widths = _exponentially_interpolate(neighborhood_start, neighborhood_end, num_layers)
            rf_overlaps = _exponentially_interpolate(rf_overlap_start, rf_overlap_end, num_layers)
        else:
            tissue_sizes = [tissue_size_end] * num_layers
            neighborhood_widths = [neighborhood_end] * num_layers
            rf_overlaps = [rf_overlap_end] * num_layers

        layer_tissue_sizes.extend(tissue_sizes)
        layer_neighborhood_widths.extend(neighborhood_widths)
        layer_rf_overlaps.extend(rf_overlaps)
    
    layer_tissue_sizes = [layer_tissue_sizes[i] for i in layer_indices]
    layer_neighborhood_widths = [layer_neighborhood_widths[i] for i in layer_indices]
    layer_rf_overlaps = [layer_rf_overlaps[i] for i in layer_indices]
    
    if large_neighborhood:
        layer_neighborhood_widths = [size for size in layer_tissue_sizes]

    print("Layer tissue sizes:", layer_tissue_sizes)
    print("Layer neighborhood widths:", layer_neighborhood_widths)
    print("Layer RF overlaps:", layer_rf_overlaps)

    return layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps


def _get_tissue_configs_v2(layer_indices, layer_assignments, exponentially_interpolate=True, constant_rf_overlap=False, large_neighborhood=False):
    layer_tissue_sizes = []
    layer_neighborhood_widths = []
    layer_rf_overlaps = []

    assert exponentially_interpolate
    assert not constant_rf_overlap

    # NW / TS Exponential 0.9827    y = 0.0249 * exp(0.156 * x) + 0.0069
    # RF Overlap Exponential 0.9984 y = 0.382 * exp(0.0562 * x) - 0.282

    num_stages = len(layer_assignments)
    areas = list(layer_assignments.keys())

    for l in range(24):
        tissue_size = 30
        neighborhood_width = (0.0249 * np.exp(0.156 * l) + 0.0069) * tissue_size
        rf_overlap = 0.382 * np.exp(0.0562 * l) - 0.282
        rf_overlap = min(1.0, max(0, rf_overlap))
        layer_tissue_sizes.append(tissue_size)
        layer_neighborhood_widths.append(neighborhood_width)
        layer_rf_overlaps.append(rf_overlap)

    layer_tissue_sizes = [layer_tissue_sizes[i] for i in layer_indices]
    layer_neighborhood_widths = [layer_neighborhood_widths[i] for i in layer_indices]
    layer_rf_overlaps = [layer_rf_overlaps[i] for i in layer_indices]

    if large_neighborhood:
        layer_neighborhood_widths = [size for size in layer_tissue_sizes]
    
    print("Layer tissue sizes:", layer_tissue_sizes)
    print("Layer neighborhood widths:", layer_neighborhood_widths)
    print("Layer RF overlaps:", layer_rf_overlaps)

    return layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps


def _exponentially_interpolate(start, end, num_points, lower_bound=0.01):
    if num_points == 1:
        return [start]
    if start == 0:
        start = lower_bound
    if end == 0:
        end = lower_bound

    return [start * (end / start) ** ((i+1) / (num_points)) for i in range(num_points)]


# single cortical sheet arrangement based on NSD stream rois
# assume high ventral, lateral, dorsal regions have similar tissue sizes and neighborhood widths 
def _get_tissue_configs_v3(layer_indices, layer_assignments=None, exponentially_interpolate=True, constant_rf_overlap=False, large_neighborhood=False, inf_neighborhood=False):
    for layer_index in layer_indices:
        assert layer_index in [14, 18, 22]

    layer_tissue_sizes = [TISSUE_SIZES['VTC']] * len(layer_indices)
    layer_neighborhood_widths = [NEIGHBORHOOD_WIDTHS['VTC']] * len(layer_indices)
    layer_rf_overlaps = [1] * len(layer_indices)

    if large_neighborhood:
        layer_neighborhood_widths = [size for size in layer_tissue_sizes]

    if inf_neighborhood:
        layer_neighborhood_widths = [np.inf for _ in layer_tissue_sizes]

    print("Layer tissue sizes:", layer_tissue_sizes)
    print("Layer neighborhood widths:", layer_neighborhood_widths)
    print("Layer RF overlaps:", layer_rf_overlaps)

    return layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps

