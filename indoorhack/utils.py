import numpy as np
import matplotlib.pyplot as plt


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


def generate_pos_neg_plot(pd_lst, nd_lst, epoch, dirname):
    m = max(max(pd_lst), max(nd_lst))
    plt.figure(figsize=(5, 5))
    plt.plot([0, m], [0, m], c="red", linewidth=0.1)
    plt.xlabel("pos dist")
    plt.ylabel("neg dist")

    plt.scatter(pd_lst, nd_lst)
    for i in range(len(pd_lst)):
        plt.annotate(str(i), (pd_lst[i], nd_lst[i]), xytext=(2.5, 2.5), textcoords="offset pixels")
    plt.grid()
    plt.savefig(dirname / (str(epoch)+"_scatter.png"))
    plt.show()
    
    plt.hist(pd_lst, alpha=0.7, bins=30, label="pos")
    plt.hist(nd_lst, alpha=0.7, bins=30, label="neg")
    plt.legend()
    plt.savefig(dirname / (str(epoch)+"_hst.png"))
    plt.show()


def scale_fix(X, x_min=0, x_max=1):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    return x_min + nom/denom
