import h5py
import numpy as np

# test the h5 files for corruption

with h5py.File('data/cxr.h5', 'r') as f:
    # Check random samples throughout the dataset
    indices = [0, 100, 1000, 10000, 50000, 100000, 200000, 377109]
    
    for i in indices:
        if i < f['cxr'].shape[0]:
            img = f['cxr'][i]
            print(f"Image {i}: min={img.min()}, max={img.max()}, mean={img.mean():.2f}, non-zero={np.count_nonzero(img)}")