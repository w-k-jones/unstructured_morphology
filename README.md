# Unstructured Morphology
Performing morphological operators on arbitrary unstructured grids in Python

Currently `label` and `watershed` have been implemented using a combination of BallTree neighbour searches and connected components.

 - `unstructured_label` produces results identical to `scipy.ndimage.label` and runs it approximately the samee time
 - `unstructured watershed` produces nearly identical results to `skimage.segmentation.watershed`, but is approx. 18x slower. Some optimisation may be possible. Different results can be produced around watershed lines, where the scikit-image watershed may segment these pixels to the nearest marker, rather than the downhill slope, in some cases.
