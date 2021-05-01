import numpy as np
from scipy.io import loadmat
import os

def mat2npy(mat_chanmap_dir):
    mat_chanmap = loadmat(mat_chanmap_dir)
    x = mat_chanmap['xcoords']
    y = mat_chanmap['ycoords']
    
    npy_chanmap = np.hstack([x,y])
    np.save('chanmap.npy', npy_chanmap)
    
    return npy_chanmap

def merge_filtered_files(filtered_location, output_directory, delete=True):
    filenames = os.listdir(filtered_location)
    filenames_sorted = sorted(filenames)
    f_out = os.path.join(output_directory, "standardized.bin")
    f = open(f_out, 'wb')
    for fname in filenames_sorted:
        if '.ipynb' in fname:
            continue
        res = np.load(os.path.join(filtered_location, fname)).astype('float32')
        res.tofile(f)
        if delete==True:
            os.remove(os.path.join(filtered_location, fname))