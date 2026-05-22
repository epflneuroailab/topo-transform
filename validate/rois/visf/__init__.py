import os
import numpy as np


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

labels = None
all_names = None

# Define group ROI mapping
group_roi_map = {
    "faces": ["L-mFus-faces", "L-pFus-faces", "L-IOG-faces",
            "R-mFus-faces", "R-pFus-faces", "R-IOG-faces"],
    
    "bodies": ["L-OTS-bodies", "L-ITG-bodies", "L-MTG-bodies", "L-LOS-bodies",
            "R-OTS-bodies", "R-ITG-bodies", "R-MTG-bodies", "R-LOS-bodies"],
    "characters": ["L-pOTS-characters", "L-IOS-characters"],
    
    "places": ["L-CoS-places", "R-CoS-places", "R-TOS-places"],
    
    "hMT": ["L-hMT", "R-hMT"],
    
    "retinotopic": ["L-v1d", "L-v2d", "L-v3d",
                    "L-v1v", "L-v2v", "L-v3v",
                    "R-v1d", "R-v2d", "R-v3d",
                    "R-v1v", "R-v2v", "R-v3v"]
}

def get_region_voxels(region):

    global labels
    global all_names

    if labels is None or all_names is None:
        # Load surface labels
        labels = np.load(f'{FILE_DIR}/labels_fsaverage5.npy', allow_pickle=True)
        all_names = np.load(f'{FILE_DIR}/names_fsaverage5.npy', allow_pickle=True)

    roi_label_map = {k:v for v, k in enumerate(all_names)}

    if region in group_roi_map:
        region = group_roi_map[region]
    elif not isinstance(region, list):
        region = [region]
    region_labels = [roi_label_map[r] for r in region]
    nsel = np.isin(labels, region_labels)
    return nsel
