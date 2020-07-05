import numpy as np


def sample_pos(idx, min_distance=0, stdev=15, max_distance=45):
    """For given"""
    r = np.random.normal(0, stdev)
    c = np.ceil(abs(r))
    pos_idx = int((min_distance + c) * np.sign(r))
    return pos_idx + idx


def get_scene_scan_indices(scene, scan, scene_id_arr, scan_id_arr, image_idx_arr):
    """Returns ordered list of indices of images belonging to scene scan."""
    mask = np.logical_and(
        scene_id_arr == scene,
        scan_id_arr == scan,
    )
    N_IMAGES = mask.sum()
    indices = np.where(mask)[0]
    indices = indices[np.argsort(image_idx_arr[mask].astype(int))]
    return indices


def get_scene_indices(scene, scene_id_arr, scan_id_arr, image_idx_arr):
    """Returns indices of images belonging to scene (all scans that belong to this scene)"""
    mask = (scene_id_arr == scene)
    N_IMAGES = mask.sum()
    indices = np.where(mask)[0]
    return indices
