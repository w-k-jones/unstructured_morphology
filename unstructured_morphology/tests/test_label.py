import numpy as np
from scipy import ndimage as ndi
from unstructured_morphology.label import unstructured_label


def test_label():
    test_labels = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 2, 2],
            [1, 0, 3, 0, 2, 2],
            [0, 0, 3, 0, 0, 0],
            [0, 3, 3, 0, 0, 4],
        ]
    )

    assert np.all(
        test_labels
        == unstructured_label(
            test_labels != 0,
            *np.meshgrid(*[range(s) for s in test_labels.shape], indexing="ij"),
            dxy=1
        )
    )

    assert np.all(
        ndi.label(test_labels != 0)[0]
        == unstructured_label(
            test_labels != 0,
            *np.meshgrid(*[range(s) for s in test_labels.shape], indexing="ij"),
            dxy=1
        )
    )
