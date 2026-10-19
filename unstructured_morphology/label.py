"""
Labelling operation on unstructured data
"""

import numpy as np
import uxarray
from scipy.sparse import csgraph, coo_array
from sklearn.neighbors import NearestNeighbors


def unstructured_label(
    mask: np.ndarray[bool], *coords, dxy: float = 1, connectivity: int = 1
) -> np.ndarray[np.int32]:
    """Calculate connected labels from unstructured data. Aims to emulate the
    behaviour of scipy.ndimage.label when used with a regular grid.

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
        csgraph.connected_components(nn.radius_neighbors_graph(radius=nn_radius))[1] + 1
    )

    output = np.zeros(mask.shape, dtype=np.int32)
    output[mask] = labels

    return output


def uxarray_label(mask: uxarray.UxDataArray[bool]) -> uxarray.UxDataArray[int]:
    """Calculate connected labels from uxarray unstructured grid data. Aims 
    to emulte the behaviour of scipy.ndimage.label

    Parameters
    ----------
    mask : uxarray.UxDataArray[bool]
        UXArray dataarray with grid information, where True values represent 
        the regions to be connected into labels.

    Returns
    -------
    uxarray.UxDataArray[int]
        UXArray dataarray of labelled regions
    
    """
    neighbours = mask.uxgrid.face_face_connectivity[mask.values].values
    origins = np.repeat(mask.n_face.values[mask.values], neighbours.shape[1])
    wh_valid = np.isin(neighbours.ravel(), origins)
    neighbours = neighbours.ravel()[wh_valid]
    origins = origins[wh_valid]
    data = np.ones(origins.size)

    neighbours_graph = coo_array(
        (data, [origins, neighbours.ravel()]), (mask.n_face.size, mask.n_face.size)
    )
    labels = csgraph.connected_components(neighbours_graph)[1]

    output = mask.copy(data=np.zeros(mask.shape, dtype=np.int32))
    output[mask.values] = np.unique(labels[mask.values], return_inverse=True)[1] + 1

    return output
