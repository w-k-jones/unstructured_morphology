"""
Watershed segmentation on unstructured data
"""

from operator import itemgetter
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors
from numba import njit


def unstructured_watershed(
    image: np.ndarray[float],
    *coords,
    markers: np.ndarray[int] = None,
    mask: np.ndarray[bool] = None,
    dxy: float = 1,
    connectivity: int = 1
) -> np.ndarray[np.int32]:
    """Perform watershed segmentation on unstructured data. Aims to emulate the
    behaviour of sklearn.segmentation.watershed when used with a regular grid.

    Parameters
    ----------
    image : np.ndarray[float]
        1D array of points over which to perform watershed semgention
    *coords : np.ndarray[float]
        1D arrays of coordinates of the points in image
    markers : np.ndarray[int], optional
        1D array of marker values to segment image into. If not provided, image
        will be segmented to local minima, by default None
    mask : np.ndarray[bool], optional
        1D array of points to segment. True values will be segmented, False
        values will not. If None, all points in the image will be assigned
        segments. By default None
    dxy : float, optional
        search radius within which neighbours are found, by default None
    connectivity : int, optional
        emulates the connectivity of scipy.ndimage.label. Increses the search
        radius by sqrt(connectivity), by default 1

    Returns
    -------
    np.ndarray[np.int32]
        1D array of segmented regions, either to markers or local minima
        depending on whether the markers keyword was provided.
    """
    coord_stack = np.stack([c.ravel()[mask.ravel()] for c in coords], -1)
    # Create markers and mask if either are not provided
    if markers is None:
        markers = np.zeros(image.shape, dtype=np.int32)
        label_offset = 1
    else:
        label_offset = np.max(markers) + 1
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    # Mask image and markers ahead of time to save on indexing
    masked_image = image[mask]
    masked_markers = markers[mask]

    n_locs = masked_image.size

    # Setup BallTree to find neighbours
    nn_radius = dxy * connectivity**0.5 if dxy is not None else None
    nn = NearestNeighbors(radius=nn_radius, algorithm="ball_tree", metric="euclidean")
    nn = nn.fit(coord_stack)
    distances, neighbours = nn.radius_neighbors(
        radius=nn_radius
    )  # About 25% of time taken

    start_nodes, end_nodes, plateau_start_nodes, plateau_end_nodes = process_neighbours(
        masked_image, masked_markers, neighbours, distances
    )  # About 25% of time taken

    # Handle plateau nodes: we need to find the shortest path to a lower value, non-plateau point
    if len(plateau_start_nodes):
        start_nodes, end_nodes = process_plateaus(
            plateau_start_nodes, plateau_end_nodes, start_nodes, end_nodes
        )

    # Build graph and find connected components
    graph = sparse.coo_matrix(
        (np.ones(len(start_nodes), dtype=np.float32), (start_nodes, end_nodes)),
        shape=(n_locs, n_locs),
    ).tocsr()

    masked_output = (
        csgraph.connected_components(graph, directed=False)[1] + label_offset
    )

    # Now find which labels contain markers and relabel them to the correct values
    if label_offset > 1 and np.any(masked_output != 0):
        marker_labels = list(set(np.unique(markers).tolist()) - set([0]))
        marker_map = dict(
            zip(
                marker_labels,
                ndi.labeled_comprehension(
                    masked_output, masked_markers, marker_labels, list, np.object_, None
                ),
            )
        )
        label_map = np.arange(masked_output.max() + 1, dtype=np.int32)
        label_map[sum(list(marker_map.values()), [])] = sum(
            [[k] * len(v) for k, v in marker_map.items()], []
        )
        masked_output = label_map[masked_output]

        # Merge regions that are segmented to a local minima to the appropriate marker
        if masked_output.max() >= label_offset:
            masked_output = merge_regions(
                masked_output, masked_image, neighbours, label_offset
            )  # About 40% of time taken

    output = np.zeros(image.shape, dtype=np.int32)
    output[mask] = masked_output

    return output


@njit()
def find_neighbour_nodes(i, n, d, masked_image, masked_markers):
    if not masked_markers[i]:
        if len(n) > 1:
            diffs = (masked_image[n] - masked_image[i]) / d
            wh_min = np.argmin(diffs)
            min_diffs = diffs[wh_min]
            if min_diffs < 0:
                return np.array([n[wh_min]])
            if min_diffs == 0:
                # Check for plateaus
                wh_zero_diff = diffs == 0
                n_zeros = np.count_nonzero(wh_zero_diff)
                if n_zeros > 1:
                    return n[wh_zero_diff]
                return np.array([n[wh_min]])
        elif len(n) == 1 and masked_image[n[0]] <= masked_image[i]:
            return n[0:]

    return np.array([-1], dtype=np.int64)


def process_neighbours(masked_image, masked_markers, neighbours, distances):
    n_points = len(neighbours)

    # preallocate
    start_nodes = np.empty(n_points, dtype=np.int32)
    end_nodes = np.empty(n_points, dtype=np.int32)

    plateau_start_nodes = []
    plateau_end_nodes = []

    insert_loc = 0
    for i in range(n_points):
        neighbour_info = find_neighbour_nodes(
            i, neighbours[i], distances[i], masked_image, masked_markers
        )
        if len(neighbour_info) == 1 and neighbour_info[0] >= 0:
            start_nodes[insert_loc], end_nodes[insert_loc] = (
                i,
                neighbour_info[0],
            )
            insert_loc += 1
        elif len(neighbour_info) > 1:
            plateau_start_nodes.extend([i] * neighbour_info.size)
            plateau_end_nodes.extend(neighbour_info)

    start_nodes = start_nodes[:insert_loc]
    end_nodes = end_nodes[:insert_loc]

    return start_nodes, end_nodes, plateau_start_nodes, plateau_end_nodes


def process_plateaus(plateau_start_nodes, plateau_end_nodes, start_nodes, end_nodes):
    unique_nodes, node_index, plateau_nodes = np.unique(
        plateau_start_nodes + plateau_end_nodes,
        return_index=True,
        return_inverse=True,
    )
    new_nodes = plateau_nodes[node_index]
    nps = plateau_nodes[: len(plateau_start_nodes)]
    npe = plateau_nodes[len(plateau_start_nodes) :]
    plateau_graph = sparse.coo_matrix(
        (np.ones(len(plateau_start_nodes)), (nps, npe)),
        shape=(len(unique_nodes), len(unique_nodes)),
    ).tocsr()
    wh_plateau = np.isin(new_nodes, nps)
    is_plateau = new_nodes[wh_plateau]
    isnt_plateau = new_nodes[~wh_plateau]

    shortest_paths = csgraph.shortest_path(plateau_graph, indices=is_plateau)[
        :, isnt_plateau
    ]
    # Check for if there is no valid path which indicates a minima plateau
    wh_path = np.isfinite(np.nanmin(shortest_paths, axis=1))
    if np.any(wh_path):
        new_plateau_start_node = unique_nodes[is_plateau[wh_path]]
        new_plateau_end_node = unique_nodes[
            isnt_plateau[np.nanargmin(shortest_paths[wh_path], axis=1)]
        ]
        # new_plateau_weights = (
        #     masked_image[new_plateau_start_node] - masked_image[new_plateau_end_node]
        # )

    else:
        new_plateau_start_node = []
        new_plateau_end_node = []
        # new_plateau_weights = []

    if np.any(~wh_path):
        wh_minima_plateau = np.isin(
            plateau_start_nodes, unique_nodes[is_plateau[~wh_path]]
        )
        minima_plateau_start_node = np.array(plateau_start_nodes)[wh_minima_plateau]
        minima_plateau_end_node = np.array(plateau_end_nodes)[wh_minima_plateau]
        # minima_plateau_weights = np.full(np.sum(wh_minima_plateau), 1e8)
    else:
        minima_plateau_start_node = []
        minima_plateau_end_node = []
        # minima_plateau_weights = []

    start_nodes = np.concatenate(
        [start_nodes, new_plateau_start_node, minima_plateau_start_node]
    )
    end_nodes = np.concatenate(
        [end_nodes, new_plateau_end_node, minima_plateau_end_node]
    )
    # weights = np.concatenate([weights, new_plateau_weights, minima_plateau_weights])

    return start_nodes, end_nodes


def merge_regions(masked_output, masked_image, neighbours, label_offset):
    region_edge_info = {}
    unique_labels = list(set(np.unique(masked_output).tolist()) - set([0]))

    for label, value, n in zip(masked_output, masked_image, neighbours):
        if label and len(n):
            neighbour_labels = masked_output[n]
            neighbour_values = masked_image[n]
            # This loop makes up the main time cost, should look into optimising
            for j in range(len(n)):
                # Only add connections where the neighbouring label is > and not a marker to avoid duplicates
                if label < neighbour_labels[j] >= label_offset:
                    key = (label, neighbour_labels[j])
                    neighbour_value = neighbour_values[j]
                    new_min, new_max = (
                        (value, neighbour_value)
                        if value <= neighbour_value
                        else (neighbour_value, value)
                    )
                    # If new_max, new_min is less than existing value, replace
                    try:
                        if new_max < region_edge_info[key][0]:
                            region_edge_info[key] = (new_max, new_min)
                        elif new_max == region_edge_info[key][1]:
                            if new_min < region_edge_info[key][1]:
                                region_edge_info[key] = (new_max, new_min)
                    except KeyError:
                        region_edge_info[key] = (new_max, new_min)

    # Create remapping array for labels
    label_map = np.zeros(np.max(unique_labels) + 1, dtype=np.int32)
    label_map[unique_labels] = unique_labels

    # Loop over connections, starting from the lowest, and merge labels
    for (start, end), _ in sorted(region_edge_info.items(), key=itemgetter(1)):
        mapped_start, mapped_end = label_map[start], label_map[end]
        if mapped_start < mapped_end >= label_offset:
            label_map[label_map == mapped_end] = mapped_start
        # After merging some end labels will be < start, so need to check for this
        elif mapped_end < mapped_start >= label_offset:
            label_map[label_map == mapped_start] = mapped_end

    masked_output = label_map[masked_output]
    # Clean up any remaining minima labels
    if masked_output.max() >= label_offset:
        masked_output[masked_output >= label_offset] = 0

    return masked_output
