import numpy as np

from ftt_source.support.singleton_dimension import (
    restore_all_singleton_axes,
    squeeze_all_singleton_axes,
    squeeze_singleton_third_axis,
    unsqueeze_singleton_third_axis,
)


def test_squeeze_and_unsqueeze_singleton_third_axis():
    original = np.arange(6).reshape(2, 3, 1)
    data = {"squeezed": original.copy(), "unchanged": np.ones((2, 3, 2))}

    squeezed_variables = squeeze_singleton_third_axis(
        data, {"squeezed", "unchanged", "missing"}
    )

    assert squeezed_variables == {"squeezed"}
    np.testing.assert_array_equal(data["squeezed"], original[:, :, 0])
    assert data["unchanged"].shape == (2, 3, 2)

    unsqueeze_singleton_third_axis(data, squeezed_variables)

    np.testing.assert_array_equal(data["squeezed"], original)


def test_unsqueeze_does_not_change_variables_that_are_no_longer_two_dimensional():
    data = {"value": np.ones((2, 3, 2))}

    unsqueeze_singleton_third_axis(data, {"value"})

    assert data["value"].shape == (2, 3, 2)


def test_squeeze_and_restore_all_singleton_axes():
    original = np.arange(3).reshape(1, 3, 1)
    data = {"global": original.copy(), "unchanged": np.ones((2, 3))}

    original_shapes = squeeze_all_singleton_axes(data, {"global"})

    assert data["global"].shape == (3,)
    assert original_shapes == {"global": (1, 3, 1)}

    restore_all_singleton_axes(data, original_shapes)

    np.testing.assert_array_equal(data["global"], original)
