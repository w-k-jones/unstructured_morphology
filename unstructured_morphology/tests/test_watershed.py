import numpy as np
from skimage.segmentation import watershed

from unstructured_morphology.watershed import unstructured_watershed


def test_single_no_markers():
    test_arr = -np.array(
        [
            [0, 0, 1, 2, 1, 0, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 2, 3, 4, 3, 2, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 0, 1, 2, 1, 0, 0],
        ],
        dtype=float,
    )

    mask = test_arr != 0

    assert np.all(
        watershed(test_arr, mask=mask)
        == unstructured_watershed(
            test_arr,
            *np.meshgrid(*[range(s) for s in test_arr.shape], indexing="ij"),
            mask=mask,
            dxy=1
        )
    )


def test_two_no_markers():
    test_arr = -np.array(
        [
            [0, 1, 2, 1, 0, 0, 1, 0],
            [1, 2, 3, 2, 1, 1, 2, 1],
            [2, 3, 4, 3, 2, 2, 3, 2],
            [1, 2, 3, 2, 1, 1, 2, 1],
            [0, 1, 2, 1, 0, 0, 1, 0],
        ],
        dtype=float,
    )

    mask = test_arr != 0

    assert np.all(
        watershed(test_arr, mask=mask)
        == unstructured_watershed(
            test_arr,
            *np.meshgrid(*[range(s) for s in test_arr.shape], indexing="ij"),
            mask=mask
        )
    )


def test_single_with_marker():
    test_arr = -np.array(
        [
            [0, 0, 1, 2, 1, 0, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 2, 3, 4, 3, 2, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 0, 1, 2, 1, 0, 0],
        ],
        dtype=float,
    )

    mask = test_arr != 0

    markers = np.zeros(test_arr.shape, dtype=np.int32)
    markers[2, 3] = 1

    assert np.all(
        watershed(test_arr, markers=markers, mask=mask)
        == unstructured_watershed(
            test_arr,
            *np.meshgrid(*[range(s) for s in test_arr.shape], indexing="ij"),
            markers=markers,
            mask=mask
        )
    )

    # Offset marker from local minima
    markers = np.zeros(test_arr.shape, dtype=np.int32)
    markers[2, 2] = 1

    assert np.all(
        watershed(test_arr, markers=markers, mask=mask)
        == unstructured_watershed(
            test_arr,
            *np.meshgrid(*[range(s) for s in test_arr.shape], indexing="ij"),
            markers=markers,
            mask=mask
        )
    )


def test_complex_two_markers():
    # Test a more complex case
    test_arr = -np.array(
        [
            [0, 1, 2, 1, 0, 0, 1, 0, 0, 1, 2, 1, 0],
            [1, 2, 3, 2, 1, 1, 2, 1, 1, 2, 3, 2, 1],
            [2, 3, 4, 3, 2, 2, 3, 2, 2, 3, 4, 3, 2],
            [1, 2, 3, 2, 1, 1, 2, 1, 1, 2, 3, 2, 1],
            [0, 1, 2, 1, 0, 0, 1, 0, 0, 1, 2, 1, 0],
            [0, 0, 0, 0, 0, 1, 2, 1, 0, 2, 3, 2, 0],
            [0, 1, 2, 1, 0, 0, 1, 0, 0, 1, 2, 1, 0],
        ],
        dtype=float,
    )

    mask = test_arr != 0
    test_arr[:, :-6] -= 1
    test_arr[~mask] = 0

    markers = np.zeros(test_arr.shape, dtype=np.int32)
    markers[2, 2] = 1
    markers[2, -3] = 2

    assert np.all(
        watershed(test_arr, markers=markers, mask=mask)
        == unstructured_watershed(
            test_arr,
            *np.meshgrid(*[range(s) for s in test_arr.shape], indexing="ij"),
            markers=markers,
            mask=mask
        )
    )


def test_plateau():
    # Test segmentation of plateaus
    test_arr = -np.array(
        [
            [0, 0.5, 1, 1, 1, 3, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 2, 1, 1, 1, 0.5, 0],
        ],
        dtype=float,
    )

    mask = test_arr != 0

    assert np.all(
        watershed(test_arr, mask=mask)
        == unstructured_watershed(
            test_arr,
            *np.meshgrid(*[range(s) for s in test_arr.shape], indexing="ij"),
            mask=mask
        )
    )
