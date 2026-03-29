#!/usr/bin/env python

import numpy as np
from pathlib import Path
import sys
from typing import Union
import unittest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import auroraplot.dt64tools as dt64  # noqa
from auroraplot.dt64tools import (
    dt64_range,
    from_hms,
    is_leap_year,
    julian_date,
    get_date,
    get_days_in_year,
    get_sidereal_day,
    get_sidereal_time_offset,
    get_time_of_day,
    gmst,
    lmst_to_utc,
    strftime,
    to_dhms,
    to_ydhms,
    utc_to_lmst,
    wrap_day,
)  # noqa


class TestDt64tools(unittest.TestCase):
    test_dates_list = [
        np.datetime64("1970-01-01"),
        np.datetime64("1970-01-02"),
        np.datetime64("1996-01-01"),
        np.datetime64("2000-07-14"),
        np.datetime64("2000-07-14T14:00:00"),
        np.datetime64("2030-01-01T12:34:56"),
    ]
    test_dates_array = np.zeros_like(test_dates_list)
    test_dates_array[:] = test_dates_list
    test_timedeltas_list = [
        np.timedelta64(1, "s"),
        np.timedelta64(1, "m") + np.timedelta64(10, "ms"),
        np.timedelta64(1, "h") + np.timedelta64(5, "s"),
        np.timedelta64(1, "D") + np.timedelta64(12, "h"),
    ]
    test_timedeltas_with_years_list = [
        np.timedelta64(1, "s"),
        np.timedelta64(1, "m") + np.timedelta64(10, "ms"),
        np.timedelta64(1, "h") + np.timedelta64(5, "s"),
        np.timedelta64(1, "D") + np.timedelta64(12, "h"),
        np.timedelta64(1, "Y"),
        np.timedelta64(10, "Y"),
    ]

    @staticmethod
    def assertSimilarTimes(
        a: Union[np.datetime64, np.timedelta64],
        b: Union[np.datetime64, np.timedelta64],
        tol=np.timedelta64(1, "s"),
        msg: str = None,
    ) -> None:
        # assert isinstance(a, type(b)) and isinstance(b, type(a)), f'types are different ({type(a).__name} and {type(b).__name})'
        diff = a - b if a > b else b - a
        diff_str = strftime(diff, "%dD %Hh %Mm %S.%#s")
        s = f"values are too different ({a} and {b} = {diff_str}."
        if msg is not None:
            s += f" {msg}"
        if isinstance(diff, np.ndarray):
            assert (diff >= -tol).all() and (diff <= tol).all(), s
        else:
            assert -tol <= diff <= tol, s

    def test_wrap_day(self):
        hours_list = list(range(-32, 65, 8))
        test_times_list = [np.timedelta64(h, "h") for h in hours_list]
        test_times_array = np.zeros_like(test_times_list)
        test_times_array[:] = test_times_list
        expected_list = [np.timedelta64(h % 24, "h") for h in hours_list]

        for t, expected in zip(test_times_list, expected_list):
            self.assertEqual(wrap_day(t), expected)

        results_array = wrap_day(test_times_array)
        for result, expected in zip(results_array, expected_list):
            self.assertEqual(result, expected)

    def test_get_date(self):
        expected_list = [
            np.datetime64("1970-01-01"),
            np.datetime64("1970-01-02"),
            np.datetime64("1996-01-01"),
            np.datetime64("2000-07-14"),
            np.datetime64("2000-07-14"),
            np.datetime64("2030-01-01"),
        ]
        for t, expected in zip(self.test_dates_list, expected_list):
            self.assertEqual(get_date(t), expected)

        results_array = get_date(self.test_dates_array)
        for result, expected in zip(results_array, expected_list):
            self.assertEqual(result, expected)

    def test_get_time_of_day(self):
        expected_list = [
            from_hms(0, 0, 0),
            from_hms(0, 0, 0),
            from_hms(0, 0, 0),
            from_hms(0, 0, 0),
            from_hms(14, 0, 0),
            from_hms(12, 34, 56),
        ]
        for t, expected in zip(self.test_dates_list, expected_list):
            self.assertEqual(get_time_of_day(t), expected)

        results_array = get_time_of_day(self.test_dates_array)
        for result, expected in zip(results_array, expected_list):
            self.assertEqual(result, expected)

    def test_is_leap_year(self):
        for y in range(1890, 2108):
            if y % 4 or y in (1900, 2100):
                self.assertFalse(is_leap_year(y), msg=f"Incorrect result for {y}")
            else:
                self.assertTrue(is_leap_year(y), msg=f"Incorrect result for {y}")

    def test_get_days_in_year(self):
        for y in range(1890, 2108):
            if y % 4 != 0 or y in (1900, 2100):
                self.assertEqual(get_days_in_year(y), 365, msg=f"Incorrect result for {y}")
            else:
                self.assertEqual(get_days_in_year(y), 366, msg=f"Incorrect result for {y}")

    def test_to_dhms(self):
        expected_list = [
            (0, 0, 0, 1),
            (0, 0, 1, 0.01),
            (0, 1, 0, 5),
            (1, 12, 0, 0),
        ]
        for ts, expected in zip(self.test_timedeltas_list, expected_list):
            for a, b in zip(to_dhms(ts), expected):
                self.assertAlmostEqual(a, b, delta=0.000001, msg=f"Incorrect result for {ts}")

    def test_to_ydhms(self):
        expected_list = [
            (0, 0, 0, 0, 1),
            (0, 0, 0, 1, 0.01),
            (0, 0, 1, 0, 5),
            (0, 1, 12, 0, 0),
            (1, 0, 0, 0, 0),
            (10, 0, 0, 0, 0),
        ]
        for ts, expected in zip(self.test_timedeltas_list, expected_list):
            for a, b in zip(to_ydhms(ts), expected):
                self.assertAlmostEqual(a, b, delta=0.000001, msg=f"Incorrect result for {ts}")

    def test_to_str(self):
        expected_datetimes_list = [
            "1970-01-01",
            "1970-01-02",
            "1996-01-01",
            "2000-07-14",
            "2000-07-14T14:00:00",
            "2030-01-01T12:34:56",
        ]
        # Test datetime64
        for t, expected in zip(self.test_dates_list, expected_datetimes_list):
            self.assertEqual(expected, dt64.to_str(t))

        expected_timedeltas_list = [
            "1s",
            "1m 0.010s",
            "1h 5s",
            "1d 12h",
            "1y",
            "10y",
        ]
        # Test timedelta64
        for ts, expected in zip(self.test_timedeltas_with_years_list, expected_timedeltas_list):
            self.assertEqual(expected, dt64.to_str(ts))

    def test_julian_date(self):
        # Results from https://www.fourmilab.ch/cgi-bin/Earth
        expected_list = [
            2440587.5,  # 1970-01-01
            2440588.5,  # 1970-01-02
            2450083.5,  # 1996-01-01
            2451739.5,  # 2000-07-14
            2451740.083333333,  # 2000-07-14T14:00:00
            2462503.024259259,  # 2030-01-01T12:34:56
        ]
        for t, expected in zip(self.test_dates_list, expected_list):
            self.assertAlmostEqual(julian_date(t), expected)

        results_array = julian_date(self.test_dates_array)
        for result, expected in zip(results_array, expected_list):
            self.assertAlmostEqual(result, expected)

    def test_get_unix_day_number(self):
        expected_list = [0, 1, 9496, 11152, 11152, 21915]
        for t, expected in zip(self.test_dates_list, expected_list):
            self.assertAlmostEqual(dt64.get_unix_day_number(t), expected)

        results_array = dt64.get_unix_day_number(self.test_dates_array)
        for result, expected in zip(results_array, expected_list):
            self.assertAlmostEqual(result, expected)

    def test_get_sidereal_day(self):
        self.assertSimilarTimes(get_sidereal_day(), from_hms(23, 56, 4, ms=100))

    def test_gmst(self):
        expected_list = [
            from_hms(6, 40, 55, ms=113),
            from_hms(6, 44, 51, ms=668),
            from_hms(6, 39, 44, ms=878),
            from_hms(19, 28, 40, ms=568),
            from_hms(19, 28, 40, 568),
            from_hms(6, 42, 46, ms=1),
        ]
        for t, expected in zip(self.test_dates_list, expected_list):
            self.assertSimilarTimes(gmst(t), expected)

        results_array = gmst(self.test_dates_array)
        for result, expected in zip(results_array, expected_list):
            self.assertSimilarTimes(result, expected)

    def test_utc_to_lmst(self):
        longitude_list = [0, 0, 20.79, 20.79, -94.08, -94.08]
        expected_list = [
            from_hms(6, 40, 55, ms=113),
            from_hms(6, 44, 51, ms=668),
            from_hms(8, 2, 54, ms=478),
            from_hms(20, 51, 50, ms=168),
            from_hms(3, 14, 39, ms=358),
            from_hms(13, 3, 26, ms=827),
        ]
        for t, lon, expected in zip(self.test_dates_list, longitude_list, expected_list):
            self.assertSimilarTimes(utc_to_lmst(t, lon), expected, msg=f"Incorrect result for {t}")

        results_array = utc_to_lmst(self.test_dates_array, np.array(longitude_list))
        for result, expected in zip(results_array, expected_list):
            self.assertSimilarTimes(result, expected)

    def test_lmst_to_utc(self):
        lmst_list = [
            from_hms(0, 0, 0),
            from_hms(1, 2, 3),
            from_hms(4, 5, 6),
            from_hms(10, 0, 0),
            from_hms(20, 0, 0),
            from_hms(23, 59, 0),
        ]
        longitude_list = [0, 0, 20.79, 20.79, -94.08, -94.08]
        date_list = [get_date(d) for d in self.test_dates_list]
        # date_list = self.test_dates_list
        expected_list = [
            np.datetime64("1970-01-01T17:16:14.568"),
            # np.datetime64('1969-12-31T17:20:10.568'), # np.datetime64('1970-01-01T17:16:14.568'),
            np.datetime64("1970-01-01T18:18:07.493") + get_sidereal_day(),
            np.datetime64("1995-12-31T20:02:50.481") + get_sidereal_day(),
            np.datetime64("2000-07-13T13:09:56.620") + get_sidereal_day(),
            np.datetime64("2000-07-14T06:46:31.850"),
            np.datetime64("2030-01-01T23:28:41.786"),
        ]
        for lmst, lon, d, expected in zip(lmst_list, longitude_list, date_list, expected_list):
            self.assertSimilarTimes(lmst_to_utc(lmst, lon, d), expected, msg=f"Incorrect result for {lmst}")

        results_array = lmst_to_utc(np.array(lmst_list), np.array(longitude_list), np.array(self.test_dates_array))
        for result, expected in zip(results_array, expected_list):
            self.assertSimilarTimes(result, expected)

    def test_get_sidereal_time_offset(self):
        longitude_list = [0, 0, 20.79, 20.79, -94.08, -94.08]
        t_offsets_start = np.timedelta64(0, "s")
        t_offsets_end = np.timedelta64(1, "D")
        t_offsets_step = np.timedelta64(10, "m")
        t_offsets = np.array(list(dt64_range(t_offsets_start, t_offsets_end, t_offsets_step)))
        expected_first_offset = [
            from_hms(6, 39, 49, ms=432),
            from_hms(6, 43, 45, ms=342),
            from_hms(8, 1, 35, ms=366),
            from_hms(20, 48, 25, ms=84),
            from_hms(3, 14, 7, ms=468),
            from_hms(13, 1, 18, ms=468),
        ]

        for t, lon, efo in zip(self.test_dates_list, longitude_list, expected_first_offset):
            t_array = t + t_offsets
            sidt_offsets, sid_day_num = get_sidereal_time_offset(t_array, lon, st=t)
            # print(f'---\n{t_array[0]} -> {sidt_offsets[0]}\n{t_array[-1]} -> {sidt_offsets[-1]}')

            self.assertSimilarTimes(sidt_offsets[0], efo)
            # All offsets should be 0 <= offset < sidereal day
            self.assertTrue(np.all(sidt_offsets >= np.timedelta64(0, "s")))
            self.assertTrue(np.all(sidt_offsets < get_sidereal_day("ms")))

            # The offsets should increase by t_offsets_step except when the day number increased
            expected_diff = np.diff(sidt_offsets)
            expected_diff[np.diff(sid_day_num) < 0] = -get_sidereal_day("ms") + t_offsets_step
            self.assertTrue(np.all(np.diff(sidt_offsets) == expected_diff))


if __name__ == "__main__":
    unittest.main()
