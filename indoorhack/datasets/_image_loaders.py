

def h5py_loader(h5py_file_path):
    f = h5py.File(h5py_file_path, "r")
    return lambda x: f["/".join((tuple(x.parts[-2].split("_")) + (x.parts[-1].split(".")[0],)))][:]
