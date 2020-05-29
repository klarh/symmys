import numpy as np

def hash_sample(dataset, N=64, initial_length=.1):
    """Create a "summary" representation of a point cloud using a hash map.

    Only one point for each cell with a grid size of a particular
    length scale will be returned in the resulting array. This length
    scale starts at `initial_length` and is decreased until at least
    `N` points are returned.

    :param dataset: (num_points, 3) set of (x, y, z) coordinates
    :param N: Minimum size of the returned summary array
    :param initial_length: Initial lengthscale to bin coordinates by
    """
    if N >= len(dataset):
        return dataset
    found_data_size = 0
    bin_size = initial_length
    while found_data_size < N:
        bins = (np.asarray(dataset)/bin_size).astype(np.int32)
        found_data_size = len(np.unique(bins, axis=0))
        bin_size *= found_data_size/N

    result = dict(zip(map(tuple, bins), dataset))
    return np.array(list(result.values()))
