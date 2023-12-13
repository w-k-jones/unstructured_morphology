"""
Watershed segmentation on unstructured data
"""

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors
from numba import njit


@njit()
def find_neighbour_nodes(i, n, d, image, markers, mask):
    if markers[mask][i] == 0:
        if len(n) > 1:
            diffs = (image[mask][n] - image[mask][i]) / d
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
        elif len(n) == 1 and image[mask][n[0]] <= image[mask][i]:
            return n[0:]

    return np.array([-1], dtype=np.int64)


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
    coord_stack
    # Create mask if not provided
    if markers is None:
        markers = np.zeros(image.shape, dtype=np.int32)
        label_offset = 1
    else:
        label_offset = np.max(markers) + 1
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    n_locs = np.count_nonzero(mask)

    # Setup BallTree to find neighbours
    nn_radius = dxy * connectivity**0.5 if dxy is not None else None
    nn = NearestNeighbors(radius=nn_radius, algorithm="ball_tree", metric="euclidean")
    nn = nn.fit(coord_stack)
    distances, neighbours = nn.radius_neighbors(
        radius=nn_radius
    )  # This could be moved to the loop below to reduce memory usage

    # preallocate
    start_nodes = np.empty(distances.size, dtype=np.int32)
    end_nodes = np.empty(distances.size, dtype=np.int32)
    # weights = np.empty(distances.size, dtype=np.float32)

    plateau_start_nodes = []
    plateau_end_nodes = []

    # Loop over nodes and assign neighbours to relevant lists - this is by far the slowest part of the process, need to speed up somehow
    insert_loc = 0
    for i, (d, n) in enumerate(zip(distances, neighbours)):
        neighbour_info = find_neighbour_nodes(
            i, n, d, image.ravel(), markers.ravel(), mask.ravel()
        )
        if len(neighbour_info) == 1 and neighbour_info[0] > 0:
            start_nodes[insert_loc], end_nodes[insert_loc] = (
                i,
                neighbour_info[0],
            )
            insert_loc += 1
        elif len(neighbour_info) == 2:
            plateau_start_nodes.extend([i] * neighbour_info.size)
            plateau_end_nodes.extend(neighbour_info)

    start_nodes = start_nodes[:insert_loc]
    end_nodes = end_nodes[:insert_loc]
    # weights = weights[:insert_loc]

    # Handle plateau nodes: we need to find the shortest path to a lower value, non-plateau point
    if len(plateau_start_nodes):
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
            #     image[mask][new_plateau_start_node] - image[mask][new_plateau_end_node]
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

    # Build graph and find connected components
    graph = sparse.coo_matrix(
        (np.ones(len(start_nodes), dtype=np.float32), (start_nodes, end_nodes)),
        shape=(n_locs, n_locs),
    ).tocsr()

    output = np.zeros(image.shape, dtype=np.int32)

    output[mask] = csgraph.connected_components(graph, directed=False)[1] + label_offset

    # Now find which labels contain markers and relabel them to the correct values
    if label_offset > 1 and np.any(output != 0):
        marker_labels = list(set(np.unique(markers).tolist()) - set([0]))
        marker_map = dict(
            zip(
                marker_labels,
                ndi.labeled_comprehension(
                    output, markers, marker_labels, list, np.object_, None
                ),
            )
        )
        label_map = np.arange(output.max() + 1, dtype=np.int32)
        label_map[sum(list(marker_map.values()), [])] = sum(
            [[k] * len(v) for k, v in marker_map.items()], []
        )
        output = label_map[output]

        # Merge regions that are segmented to a local minima to the appropriate marker
        if output.max() >= label_offset:
            region_edge_info = []
            for label in list(set(np.unique(output).tolist()) - set([0])):
                wh_label = np.where(output[mask] == label)[0]
                for i, n in enumerate(
                    nn.radius_neighbors(coord_stack[wh_label], radius=nn_radius)[1]
                ):
                    if len(n):
                        neighbour_labels = output[mask][n]
                        # Only add connections where the neighbouring label is > and not a marker to avoid duplicates
                        wh_diff_label = np.where(
                            (neighbour_labels > label)
                            & (neighbour_labels >= label_offset)
                        )[0]
                        region_edge_info.extend(
                            [
                                [
                                    label,
                                    neighbour_labels[wh],
                                    min(image[mask][wh_label[i]], image[mask][n[wh]]),
                                    max(image[mask][wh_label[i]], image[mask][n[wh]]),
                                ]
                                for wh in wh_diff_label
                            ]
                        )
            # Create dataframe, sort by max then min, then reduce to just the first connection for each label
            reduced_df = (
                pd.DataFrame(
                    region_edge_info, columns=["start", "end", "min_val", "max_val"]
                )
                .sort_values(["max_val", "min_val"])
                .groupby("start")
                .apply(lambda df: df.groupby("end").first())[["min_val", "max_val"]]
                .reset_index()
                .sort_values(["max_val", "min_val", "start", "end"])
            )

            # Save original labels as keys for remapping
            keys = np.concatenate([reduced_df.start, reduced_df.end])

            # Loop over connections, starting from the lowest, and merge labels
            for i in reduced_df.index:
                start = reduced_df.loc[i, "start"]
                end = reduced_df.loc[i, "end"]
                if start < end >= label_offset:
                    reduced_df.loc[reduced_df["start"] == end, "start"] = start
                    reduced_df.loc[reduced_df["end"] == end, "end"] = start
                # After merging some end labels will be < start, so need to check for this
                elif end < start >= label_offset:
                    reduced_df.loc[reduced_df["start"] == start, "start"] = end
                    reduced_df.loc[reduced_df["end"] == start, "end"] = end

            # Create mapping from orginal regions to merged regions
            values = np.concatenate([reduced_df.start, reduced_df.end])
            wh_map = values < keys
            region_mapper = dict(zip(keys[wh_map], values[wh_map]))
            label_map = np.zeros(output.max() + 1, dtype=np.int32)
            label_map[:label_offset] = np.arange(label_offset, dtype=np.int32)
            label_map[list(region_mapper.keys())] = list(region_mapper.values())

            output = label_map[output]
            # Clean up any remaining minima labels
            if output.max() >= label_offset:
                output[output >= label_offset] = 0
    return output
