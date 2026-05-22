from neuroparc import atlas as natlas
from neuroparc.extra import glasser
import numpy as np

class MAP:
    atlas = natlas.Atlas("NSD-Streams")
    label_surface = atlas.label_surface("fsaverage5")

    rev_label_name_map = {
        "early": 1,
        "mid-ventral": 2,
        "mid-lateral": 3,
        "mid-dorsal": 4,
        "high-ventral": 5,
        "high-lateral": 6,
        "high-dorsal": 7,
    }

    regions = list(rev_label_name_map.keys())

    def get_region_labels(region):
        if isinstance(region, str):
            region = [region]
        labels = []
        for r in region:
            labels.append(MAP.rev_label_name_map[r.lower()])
        return labels

    def get_nsel(region):
        region_labels = MAP.get_region_labels(region)
        return np.isin(MAP.label_surface, region_labels)

def get_region_voxels(region):
    region = [region] if not isinstance(region, list) else region
    nsel = MAP.get_nsel(region)
    return nsel

if __name__ == "__main__":

    for region in MAP.regions:
        sel = get_region_voxels(region)
        print(f"Region: {region}, Num Voxels: {sel.sum()}")

        # from visual import plot_single_factor
        # import matplotlib.pyplot as plt

        # plot_single_factor(sel.astype(float), vmin=0, vmax=1, cmap='viridis', with_colorbar=True)
        # plt.savefig(f"{region}_nsd_region.png")
        # plt.close()