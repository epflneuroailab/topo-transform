import numpy as np
from esda.moran import Moran
from libpysal.weights import W

from utils import cached

# code from https://github.com/epflneuroailab/topolm/blob/main/fMRI/Hauptman2024_fMRI_plotting/src/processing/morans.py

def compute_morans_I_neural(data, adjacency_list):
    # Filter valid data and prepare adjacency list
    valid_mask = ~np.isnan(data) & (data != 0)
    valid_indices = np.where(valid_mask)[0]
    valid_data = data[valid_mask]
    
    # Map valid indices to positions in valid_data for adjacency list
    valid_index_map = {idx: pos for pos, idx in enumerate(valid_indices)}
    valid_adjacency_list = {
        valid_index_map[idx]: [valid_index_map[nbr] for nbr in adjacency_list[idx] if nbr in valid_index_map]
        for idx in valid_indices
    }
    valid_adjacency_list = {idx: nbrs for idx, nbrs in valid_adjacency_list.items() if nbrs}

    # Prepare data for Moran's I computation
    w = W(valid_adjacency_list)
    ordered_data = valid_data[[w.id_order]]
    moran = Moran(ordered_data, w)
    return moran.I


def compute_adjacency_list(faces, num_vertices):
    adjacency_list = {i: set() for i in range(num_vertices)}
    for face in faces:
        i, j, k = face
        adjacency_list[i].update([j, k])
        adjacency_list[j].update([i, k])
        adjacency_list[k].update([i, j])
    # Convert sets to lists
    adjacency_list = {key: list(val) for key, val in adjacency_list.items()}
    return adjacency_list


@cached('nsd_high_adjacency_list', persistent=True)
def compute_nsd_high_adjacency_list():
    from validate.rois.nsd import get_region_voxels
    sel = get_region_voxels(["high-ventral", "high-lateral", "high-dorsal"])
    sel_lh = sel[:len(sel)//2]
    sel_rh = sel[len(sel)//2:]

    # get faces from nilearn
    from nilearn import datasets, surface
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    coords_lh, faces_lh = surface.load_surf_mesh(fsaverage.pial_left)
    coords_rh, faces_rh = surface.load_surf_mesh(fsaverage.pial_right)

    def _filter_faces(faces, sel):
        sel = np.where(sel)[0]
        sel_set = set(sel)
        filtered_faces = []
        for face in faces:
            if all(vertex in sel_set for vertex in face):
                new_face = [np.where(sel == vertex)[0][0] for vertex in face]
                filtered_faces.append(new_face)
        return np.array(filtered_faces)

    faces_lh_filtered = _filter_faces(faces_lh, sel_lh)
    faces_rh_filtered = _filter_faces(faces_rh, sel_rh)

    adj_lh = compute_adjacency_list(faces_lh_filtered, sum(sel_lh))
    adj_rh = compute_adjacency_list(faces_rh_filtered, sum(sel_rh))

    return adj_lh, adj_rh