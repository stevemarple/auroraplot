#!/usr/bin/env python
import auroraplot as ap
import auroraplot.data
import auroraplot.dt64tools as dt64
import auroraplot.datasets.lancsrio
import auroraplot.filter
import auroraplot.riodata
from auroraplot.riodata.qdc_algorithms import UpperEnvelope
import logging
import matplotlib as mpl
import numpy as np
import os
from pathlib import Path
import pickle
from typing import Any, Union

if os.name == "posix" and ("DISPLAY" not in os.environ or not os.environ["DISPLAY"]):
    mpl.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PathType = Union[str, Path]

save_dir = Path.home() / "Auroraplot"


def pickle_save(obj: Any, path: PathType):
    p = Path(path)
    if p.suffix != ".pickle":
        p = Path(str(p) + ".pickle")
    print(f"Saving to {p}")
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path: PathType):
    p = Path(path)
    if p.suffix != ".pickle":
        p = Path(str(p) + ".pickle")
    print(f"Loading from {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    # Set default values for figure
    mpl.rcParams["figure.facecolor"] = "w"
    mpl.rcParams["figure.figsize"] = [8, 6]
    mpl.rcParams["figure.subplot.bottom"] = 0.1
    mpl.rcParams["figure.subplot.left"] = 0.12
    mpl.rcParams["figure.subplot.right"] = 0.925
    mpl.rcParams["figure.subplot.top"] = 0.85
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["legend.fontsize"] = "medium"
    mpl.rcParams["legend.numpoints"] = 1

    t = np.datetime64("2019-08-20")
    project = "LANCSRIO"
    site = "KAR1"
    # site = "KIL1"

    qdc_cadence = np.timedelta64(10, "s")
    q = dt64.floor(t, qdc_cadence)
    print(q)

    qdc_t, qdc_load_st, qdc_load_et = auroraplot.riodata.RioData.calculate_qdc_create_times(t, project, site)
    qdc_load_et = qdc_load_st + np.timedelta64(8, "D")

    print(f"Making QDC for {auroraplot.format_project_site(project, site)}")
    print(f"QDC: {qdc_t} ({dt64.fmt_dt64_range(qdc_load_st, qdc_load_et)})")
    power_data = ap.load_data(
        project,
        site,
        data_type="RioPower",
        start_time=qdc_load_st,
        end_time=qdc_load_et,
        channels=["50"],
    )
    pickle_save(power_data, save_dir / "power_data")
    print(power_data)
    fh = plt.figure()
    ah = fh.gca()
    power_data.plot(channels=["50"], axes=ah, label="Original")
    plt.show(block=False)

    sliding_median_filter = auroraplot.filter.SlidingMedianFilter(window=np.timedelta64(590, "s"), cadence=qdc_cadence)
    smoothed_power_data = sliding_median_filter(power_data, inplace=False)
    print(smoothed_power_data)
    smoothed_power_data.plot(channels=["50"], axes=ah, label="Smoothed", color="red")
    pickle_save(smoothed_power_data, save_dir / "smoothed_power_data")

    qdc_algorithm = UpperEnvelope(rows=[1, 2])
    qdc = power_data.make_qdc(qdc_algorithm, cadence=qdc_cadence)
    print(qdc)
    pickle_save(qdc, save_dir / "qdc")

    sgf_2h = auroraplot.filter.SavitzkyGolayFilter(
        polyorder=3, window=np.timedelta64(7210, "s"), cadence=np.timedelta64(10, "s"), mode="wrap"
    )
    smoothed_qdc = sgf_2h(qdc)
    print(smoothed_qdc)
    pickle_save(smoothed_qdc, save_dir / "smoothed_qdc")

    fh_qdc = plt.figure()
    ah_qdc = fh_qdc.gca()
    qdc.plot(channels=["50"], title="QDC comparison", axes=ah_qdc, label="Original")
    smoothed_qdc.plot(channels=["50"], axes=ah_qdc, label="Savitzky-Golay filter 2h window", color="red")
    plt.show()


if __name__ == "__main__":
    main()
