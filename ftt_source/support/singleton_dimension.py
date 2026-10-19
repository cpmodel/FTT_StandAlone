"""Helpers for temporarily removing singleton third dimensions."""

from collections.abc import Iterable, MutableMapping
from typing import Any

import numpy as np


def squeeze_singleton_third_axis(
    data: MutableMapping[str, Any], variable_names: Iterable[str]
) -> set[str]:
    """Squeeze singleton third axes in ``data`` and return the changed names."""
    squeezed_variables = set()

    for variable_name in variable_names:
        value = data.get(variable_name)
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 3
            and value.shape[2] == 1
        ):
            data[variable_name] = value[:, :, 0]
            squeezed_variables.add(variable_name)

    return squeezed_variables


def unsqueeze_singleton_third_axis(
    data: MutableMapping[str, Any], variable_names: Iterable[str]
) -> None:
    """Restore singleton third axes for variables changed by the squeeze helper."""
    for variable_name in variable_names:
        value = data.get(variable_name)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            data[variable_name] = value[:, :, np.newaxis]


def squeeze_all_singleton_axes(
    data: MutableMapping[str, Any], variable_names: Iterable[str]
) -> dict[str, tuple[int, ...]]:
    """Squeeze all singleton axes and return the original shape for each change."""
    original_shapes = {}

    for variable_name in variable_names:
        value = data.get(variable_name)
        if isinstance(value, np.ndarray) and 1 in value.shape:
            original_shapes[variable_name] = value.shape
            data[variable_name] = np.squeeze(value)

    return original_shapes


def restore_all_singleton_axes(
    data: MutableMapping[str, Any], original_shapes: dict[str, tuple[int, ...]]
) -> None:
    """Restore arrays changed by :func:`squeeze_all_singleton_axes`."""
    for variable_name, original_shape in original_shapes.items():
        data[variable_name] = data[variable_name].reshape(original_shape)
