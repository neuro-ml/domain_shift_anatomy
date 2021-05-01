import random
import numpy as np

from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from dpipe.im.slices import iterate_slices
from dpipe.itertools import lmap
from dpipe.batch_iter import unpack_args


SPATIAL_DIMS = (-3, -2, -1)


def get_random_slice(*arrays, interval: int = 1):
    slc = np.random.randint(arrays[0].shape[-1] // interval) * interval
    return tuple(array[..., slc] for array in arrays)


def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def center_choice_random(inputs, y_patch_size):
    x, y = inputs
    center = sample_center_uniformly(y.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS)
    return x, y, center


def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper


def center_choice(inputs, y_patch_size, y_index=1, drop_y=False, nonzero_fraction=0.5, shift_value=(17, 17, 17)):
    """`centers` comes last."""
    *spatial_inputs, centers = inputs

    y_patch_size = np.array(y_patch_size)
    y_shape = spatial_inputs[y_index].shape

    if drop_y:
        spatial_inputs = (inp for i, inp in enumerate(spatial_inputs) if i != y_index)

    if len(centers) > 0 and np.random.uniform() < nonzero_fraction:
        center = random.choice(centers)
    else:
        center = sample_center_uniformly(y_shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS)

    if isinstance(shift_value, int):
        shift_value = [shift_value] * 3
    center += np.array([np.random.randint(-v, v) if v > 0 else 0 for v in shift_value])
    center = np.array([np.clip(c, 0, s - 1) for c, s in zip(center, np.array(y_shape)[np.array(SPATIAL_DIMS)])])

    return (*spatial_inputs, center)


def extract_patch(inputs, x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS):
    x, y, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min, axis=spatial_dims)
    y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0, axis=spatial_dims)
    return x_patch, y_patch


