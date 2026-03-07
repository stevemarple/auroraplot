#!/usr/bin/env python

import auroraplot as ap
import auroraplot.data
import auroraplot.dt64tools as dt64
import auroraplot.datasets.lancsrio
import auroraplot.riodata
from auroraplot.riodata.qdc_algorithms import UpperEnvelope
import logging
import matplotlib as mpl
import numpy as np
import os

if os.name == "posix" and ("DISPLAY" not in os.environ or not os.environ["DISPLAY"]):
    mpl.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    q = dt64.floor(t, np.timedelta64(10, "s"))
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
    print(power_data)
    algorithm = UpperEnvelope(rows=[1, 2])
    power_data.plot(channels=["50"])

    qdc = power_data.make_qdc(algorithm)
    print(qdc)
    qdc.plot(channels=["50"], title="Fix me")
    plt.show()


if __name__ == "__main__":
    main()
