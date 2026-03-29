#!/usr/bin/env python

import numpy as np
from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import auroraplot.datasets.lancsrio  # noqa
import auroraplot.riodata  # noqa


class TestRioData(unittest.TestCase):
    def test_calculate_qdc_start_time(self):
        expected_results = [  # test date, expected result
            # The IRIS riometer started operating on 1994-09-02, and that date should be used for the first QDC.
            [np.datetime64("1994-09-02"), np.datetime64("1994-09-02")],
            [np.datetime64("1994-09-03"), np.datetime64("1994-09-02")],  # The next day should give the same result
            [np.datetime64("1994-09-16"), np.datetime64("1994-09-16")],  # The second QDC for IRIS
            [np.datetime64("1994-09-17"), np.datetime64("1994-09-16")],  # The next day should give the same result
        ]
        for t, expected_qdc_st in expected_results:
            qdc_st = auroraplot.riodata.RioData.calculate_qdc_start_time(t, "LANCSRIO", "KIL1")
            self.assertEqual(qdc_st, expected_qdc_st)

    def test_calculate_qdc_create_times(self):
        expected_results = [
            # test date, expected start time, expected load start time, expected load end time.
            # If it seems odd that the load start time was before the instrument started operating that is because
            # initially QDcs were created using 14 days of data (it was solar minimum). Later using 18 days
            # become normal practice.
            [
                np.datetime64("1994-09-02"),
                np.datetime64("1994-09-02"),
                np.datetime64("1994-08-31"),
                np.datetime64("1994-09-18"),
            ],
            [
                np.datetime64("1994-09-03"),
                np.datetime64("1994-09-02"),
                np.datetime64("1994-08-31"),
                np.datetime64("1994-09-18"),
            ],
            [
                np.datetime64("1994-09-16"),
                np.datetime64("1994-09-16"),
                np.datetime64("1994-09-14"),
                np.datetime64("1994-10-02"),
            ],
            [
                np.datetime64("1994-09-17"),
                np.datetime64("1994-09-16"),
                np.datetime64("1994-09-14"),
                np.datetime64("1994-10-02"),
            ],
        ]
        for t, expected_qdc_st, expected_load_st, expected_load_et in expected_results:
            qdc_st, load_st, load_et = auroraplot.riodata.RioQDC.calculate_qdc_create_times(t, "LANCSRIO", "KIL1")
            self.assertEqual(qdc_st, expected_qdc_st)
            self.assertEqual(load_st, expected_load_st)
            self.assertEqual(load_et, expected_load_et)


if __name__ == "__main__":
    unittest.main()
