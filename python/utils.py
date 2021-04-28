import numpy as np
from scipy.io import loadmat

def mat2npy(mat_chanmap_dir):
    mat_chanmap = loadmat(mat_chanmap_dir)
    x = mat_chanmap['xcoords']
    y = mat_chanmap['ycoords']
    
    npy_chanmap = np.hstack([x,y])
    np.save('chanmap.npy', npy_chanmap)
    
    return npy_chanmap