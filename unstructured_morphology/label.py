"""
Labelling operation on unstructured grids.
"""

import numpy as np
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors


def unstructured_label(
    mask: np.ndarray[bool], *coords, dxy: float = None, connectivity: int = 1
) -> np.ndarray[np.int32]:
    """Calculate connected labels from unstructured data. 

    Parameters
    ----------
    mask : np.ndarray[bool]
        1D array of points, with True values to be connected into labels
    *coords : np.ndarray[float]
        1D arrays of coordinates of the points in mask
    dxy : float, optional
        search radius within which neighbours are found, by default None
    connectivity : int, optional
        emulates the connectivity of scipy.ndimage.label. Increses the search 
        radius by sqrt(connectivity), by default 1

    Returns
    -------
    np.ndarray[np.int32]
        1D array of labels
    """
    nn = NearestNeighbors(algorithm="ball_tree", metric="euclidean")
    nn = nn.fit(np.stack([c[mask] for c in coords], -1))
    nn_radius = dxy * connectivity**0.5 if dxy is not None else None
    labels = (
        csgraph.connected_components(
            nn.radius_neighbors_graph(radius=nn_radius)
        )[1]
        + 1
    )

    output = np.zeros(mask.shape, dtype=np.int32)
    output[mask] = labels

    return output
