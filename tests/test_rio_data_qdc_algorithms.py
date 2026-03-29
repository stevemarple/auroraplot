#!/usr/bin/env python

from math import nan
from numpy import arange, nan, ndarray
from numpy.testing import assert_array_almost_equal
from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from auroraplot.riodata.qdc_algorithms import (
    get_algorithms,
    get_default_algorithm,
    QdcAlgorithmBase,
    UpperEnvelope,
)  # noqa


class TestRioDataQdcAlgorithms(unittest.TestCase):
    def test_qdc_algorithm(self):
        """The QdcAlgorithm class should not be instantiated directly.

        Ensure it raise errors
        """
        alg = QdcAlgorithmBase()
        data = [arange(10, dtype=int)]
        self.assertRaises(NotImplementedError, alg.get_processing)
        self.assertRaises(NotImplementedError, alg.process, data)

    def test_get_algorithms(self):
        # We should not expect the algorithms list to be in any particular order. Compare equality using sets.
        algorithms = set(get_algorithms())
        expected = {UpperEnvelope}
        self.assertSetEqual(algorithms, expected)
        for alg in algorithms:
            self.assertTrue(issubclass(alg, QdcAlgorithmBase))

    def test_get_default_algorithm(self):
        default_algorithm = get_default_algorithm()
        self.assertTrue(issubclass(default_algorithm, QdcAlgorithmBase))
        self.assertIn(default_algorithm, get_algorithms())

    def test_upper_envelope(self):
        data = [
            arange(10, dtype=int) * 1,  # => row 4
            arange(10, dtype=int) * 2,  # => row 3
            arange(10, dtype=int) * 4,  # => row 1
            arange(10, dtype=int) * nan,
            arange(10, dtype=int) * 3,  # => row 2
            arange(10, dtype=int) * 5,  # => row 0
        ]

        ue_row_0 = UpperEnvelope(rows=[0])
        expected_qdc = arange(10, dtype=int) * 5
        result: ndarray = ue_row_0.process(data)
        assert_array_almost_equal(result, expected_qdc)

        ue_row_1_2 = UpperEnvelope(rows=[1, 2])
        expected_qdc = arange(10, dtype=int) * 3.5
        result: ndarray = ue_row_1_2.process(data)
        assert_array_almost_equal(result, expected_qdc)


if __name__ == "__main__":
    unittest.main()
