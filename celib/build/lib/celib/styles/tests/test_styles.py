# -*- coding: utf-8 -*-
"""
test_styles
===========
Tests for `celib.styles` module.

"""

import unittest

from matplotlib import cycler
import matplotlib.pyplot as plt
from matplotlib import colors

import celib


class TestStyles(unittest.TestCase):

    def setUp(self):
        plt.rcdefaults()

    def test_apply_defaults(self):
        # NB Implicitly checks celib.styles.use() at the same time

        # Check that the matplotlib defaults include the top and right spines
        self.assertTrue(plt.rcParams['axes.spines.top'] is True)
        self.assertTrue(plt.rcParams['axes.spines.right'] is True)

        # Apply the CE defaults and check that the spines are now off
        celib.styles.apply_defaults()
        self.assertTrue(plt.rcParams['axes.spines.right'] is False)
        self.assertTrue(plt.rcParams['axes.spines.top'] is False)

        # Check that the cycler is as expected (the CE three-colour palette)
        expected_cycler = (cycler('linestyle', ['-', '--', ':', '-.']) *
                           cycler('color', ['c5446e', 'aab71d',
                                            '49c9c5', '009fe3',
                                            '909090', '0b1f2c']))
        self.assertTrue(plt.rcParams['axes.prop_cycle'] == expected_cycler)

    def test_create_cycler(self):
        expected = (cycler('linestyle', ['-', '--', ':', '-.']) *
                    cycler('color', ['009fe3', '909090', '0b1f2c']))
        result = celib.styles.create_cycler(color=['009fe3', '909090', '0b1f2c'],
                                            linestyle=['-', '--', ':', '-.'])
        self.assertTrue(result == expected)

    def test_interpolate_colours(self):
        expected = [tuple([i / 4] * 3) for i in range(5)]
        result = celib.styles.interpolate_colours(palette=((0, 0, 0), (1, 1, 1)),
                                                  n=5)
        self.assertTrue(result == expected)

    def test_add_ce_colours(self):
        # Test whether CE colours are in list of named colours
        ce_colours = [
                'ce-pink-main',
                'ce-teal-main',
                'ce-green-main',
                'ce-cyan-1',
                'ce-accent-2',
                'ce-accent-3',
                'ce-pink-subtle',
                'ce-teal-subtle',
                'ce-green-subtle',
                'ce-light-cyan',
                'ce-light-cyan-2']

        expected_colors = list(colors.get_named_colors_mapping().keys())
        self.assertTrue(all(i in expected_colors for i in ce_colours))


if __name__ == '__main__':
    unittest.main()
