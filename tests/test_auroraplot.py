#!/usr/bin/env python

import unittest
from typing import List, Union

import numpy as np

from auroraplot import wrap_degrees, wrap_degrees_pm180


class TestDt64tools(unittest.TestCase):
    @staticmethod
    def get_angles(data_type) -> List:
        a = [-720, -600, -480, -360, -240, -120, -180, 0, 120, 180, 240, 360, 480, 600, 720]
        return [data_type(a) for a in a]

    def test_wrap_degrees(self):
        expected_list = [0, 120, 240, 0, 120, 240, 180, 0, 120, 180, 240, 0, 120, 240, 0]
        for t, expected in zip(self.get_angles(int), expected_list):
            self.assertEqual(wrap_degrees(t), int(expected))

        for t, expected in zip(self.get_angles(float), expected_list):
            self.assertEqual(wrap_degrees(t), float(expected))

        results_array = wrap_degrees(np.array(self.get_angles(int)))
        for result, expected in zip(results_array, expected_list):
            self.assertEqual(result, expected)

        results_array = wrap_degrees(np.array(self.get_angles(float)))
        for result, expected in zip(results_array, expected_list):
            self.assertEqual(result, expected)

    def test_wrap_degrees_pm180(self):
        expected_list = [0, 120, -120, 0, 120, -120, -180, 0, 120, -180, -120, 0, 120, -120, 0]
        for t, expected in zip(self.get_angles(int), expected_list):
            self.assertEqual(wrap_degrees_pm180(t), int(expected))

        for t, expected in zip(self.get_angles(float), expected_list):
            self.assertEqual(wrap_degrees_pm180(t), float(expected))

        results_array = wrap_degrees_pm180(np.array(self.get_angles(int)))
        for result, expected in zip(results_array, expected_list):
            self.assertEqual(result, expected)

        results_array = wrap_degrees_pm180(np.array(self.get_angles(float)))
        for result, expected in zip(results_array, expected_list):
            self.assertEqual(result, expected)
