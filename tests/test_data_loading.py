import numpy as np
import polars as pl
import pytest

from ftt_source.support.data_loading import _fill_from_input_df


def test_fill_warns_and_skips_unmatched_compact_coordinate_rows():
    frame = pl.DataFrame(
        {
            'RTI': ['A', 'missing'],
            '2020': [1.0, 9.0],
            '2021': [2.0, 9.0],
        }
    )
    target = np.zeros((2, 2))

    with pytest.warns(UserWarning, match='1 input row skipped'):
        _fill_from_input_df(
            frame,
            'VAR',
            {'VAR': ['RTI', 'TIME']},
            {'RTI': ['A', 'B'], 'TIME': [2020, 2021]},
            {'RTI': {'a': 0, 'b': 1}, 'TIME': {'2020': 0, '2021': 1}},
            [2020, 2021],
            {2020: 0, 2021: 1},
            {'VAR': 2020},
            target,
        )

    np.testing.assert_array_equal(target, [[1.0, 2.0], [0.0, 0.0]])
