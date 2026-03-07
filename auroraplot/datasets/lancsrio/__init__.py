import copy
from decimal import Decimal
import logging
import numpy as np
from pathlib import Path
from typing import List

import auroraplot as ap
from auroraplot.data import generic_load_converter
from auroraplot.datasets.aurorawatchnet import check_temperature, check_voltage, load_awn_data
from auroraplot.riodata import RioRawPower, RioPower, RioAbs, RioQDC
from auroraplot.temperaturedata import TemperatureData
from auroraplot.voltagedata import VoltageData

import scipy

logger = logging.getLogger(__name__)

project_name = "LANCSRIO"
project_name_lc = project_name.lower()
base_url = "/data/riometer/"


def make_channels_from_beam_count(num_beams: int):
    """
    Make chanel names

    Beams are numbered from 1.
    """
    # return list(map(str, range(1, num_beams + 1)))
    return np.array(list(map(str, range(1, num_beams + 1))))


sites = {
    "AND2": {
        "location": "Andøya, Norway",
        "latitude": Decimal("69.1458"),
        "longitude": Decimal("16.0292"),
        "elevation": 10,  # Estimated
        "start_time": np.datetime64("2006-01-17T00:00:00+0000", "s"),
        "end_time": np.datetime64("2025-01-01T00:00:00+0000", "s"),
        "frequency": 38.2e6,
        "beams": make_channels_from_beam_count(50),  # 1-49: imaging beams; 50: widebeam
        "copyright": "Lancaster University",
        "hardware_type": "AIRIS",
        "imaging_riometer": True,
        "data_types": {},
    },
    "AND3": {
        "location": "Andøya, Norway",
        "latitude": Decimal("69.1458"),
        "longitude": Decimal("16.0292"),
        "elevation": 10,  # Estimated
        "start_time": np.datetime64("2025-10-01T00:00:00+0000", "s"),
        "end_time": None,
        "frequency": 30.0e6,
        "beams": make_channels_from_beam_count(1),  # Widebeam only
        "copyright": "Lancaster University",
        "hardware_type": "AuroraWatchNet",
        "imaging_riometer": False,
        "data_types": {},
    },
    "KAR1": {
        # WARNING: Temporary linearization data
        "location": "Kárhóll, Iceland",
        "latitude": 65.708105,
        "longitude": -17.369623,
        "elevation": 115,  # From site GNSS
        "start_time": np.datetime64("2019-08-05T00:00:00+0000", "s"),
        "end_time": None,  # np.datetime64('2026-01-01T00:00:00+0000', 's'),
        "frequency": 38.235e6,
        "beams": make_channels_from_beam_count(50),  # 1-49: imaging beams; 50: widebeam
        "copyright": "Lancaster University",
        "hardware_type": "AuroraWatchNet",
        "imaging_riometer": True,
        "data_types": {},
    },
    "KIL1": {
        "location": "Kilpisjärvi, Finland",
        "latitude": Decimal("69.05"),
        "longitude": Decimal("20.79"),
        "elevation": 470,  # Estimated
        "start_time": np.datetime64("1994-09-30T00:00:00+0000", "s"),
        "end_time": np.datetime64("2026-01-01T00:00:00+0000", "s"),
        "frequency": 38.2e6,
        "beams": make_channels_from_beam_count(50),  # 1-49: imaging beams; 50: widebeam
        "copyright": "Lancaster University",
        "hardware_type": "IRIS",
        "imaging_riometer": True,
        "data_types": {},
    },
}

# Data types that are supported by riometers using the AuroraWatchNet data logger system
awn_data_types = {
    "RioRawPower": {
        "default": "1s",
        "1s": {
            "path": base_url + "{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt",
            "duration": np.timedelta64(24, "h"),
            "format": "aurorawatchnet",
            "constructor": RioRawPower,
            "timestamp_method": "unixtime",
            "load_converter": generic_load_converter,
            "nominal_cadence": np.timedelta64(1000000, "us"),
            "units": "V",
        },
    },
    "RioPower": {
        "default": "1s",
        "1s": {
            "path": base_url + "{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt",
            "duration": np.timedelta64(24, "h"),
            "format": "aurorawatchnet",
            "constructor": RioRawPower,
            "timestamp_method": "unixtime",
            "data_check": check_temperature,  # TODO: Implement this in generic_load_converter()
            "load_converter": RioPower.load_from_raw_power,
            "nominal_cadence": np.timedelta64(1000000, "us"),
            "units": "dBm",
        },
    },
    "RioQDC": {
        "default": "10s",
        "10s": {
            "path": base_url + "qdc/{site_lc}/%Y/%m/{site_lc}_%Y%m%d_{archive_lc}.txt",
            "duration": np.timedelta64(24, "h"),
            "load_converter": load_awn_data,
            "nominal_cadence": np.timedelta64(1000000, "us"),
            "units": "dBm",
        },
    },
    "RioAbs": {
        "default": "from_power",
        "from_power": {
            "path": base_url + "{site_lc}/%Y/%m/{site_lc}_%Y%m%d.txt",
            "duration": np.timedelta64(24, "h"),
            "load_converter": load_awn_data,
            "nominal_cadence": np.timedelta64(1000000, "us"),
            "units": "dBm",
        },
    },
    # House-keeping data channels
    "TemperatureData": {
        "realtime": {
            "channels": np.array(
                [  # 'Sensor temperature',
                    "System temperature",
                ]
            ),
            "path": base_url + "{site_lc}/%Y/%m/{site_lc}_%Y%m%d_temp.txt",
            "duration": np.timedelta64(24, "h"),
            "format": "aurorawatchnet",
            "constructor": TemperatureData,
            "timestamp_method": "unixtime",
            "data_check": check_temperature,  # TODO: Implement this in generic_load_converter()
            "load_converter": generic_load_converter,
            "nominal_cadence": np.timedelta64(30000000, "us"),
            "units": "\N{DEGREE SIGN}C",
            "sort": True,
        },
    },
    "VoltageData": {
        "realtime": {
            "channels": np.array(["Riometer supply voltage"]),
            "path": base_url + "{site_lc}/%Y/%m/{site_lc}_%Y%m%d_rio_v.txt",
            "duration": np.timedelta64(24, "h"),
            "format": "aurorawatchnet",
            "constructor": VoltageData,
            "timestamp_method": "unixtime",
            "data_check": check_voltage,  # TODO: Implement this in generic_load_converter()
            "load_converter": generic_load_converter,
            "nominal_cadence": np.timedelta64(30000000, "us"),
            "units": "V",
            "sort": True,
        },
    },
}

for site_name in sites:
    site_lc = site_name.lower()
    filename_code_corrections = {
        "and3": "rio-mo1",  # Initially set up with wrong name
        # 'kar1': 'kar1',
    }
    filename_code = filename_code_corrections.get(site_lc, site_lc)
    # filename_code = site_lc
    # if site_lc == 'and3':
    #     filename_code = 'rio-mo1'  # Initially set up with wrong name
    site = sites[site_name]
    if "line_color" not in site:
        site["line_color"] = [1, 0, 0]

    # Populate the data types
    if "data_types" not in site:
        site["data_types"] = {}
    sdt = site["data_types"]

    if site["hardware_type"] == "AuroraWatchNet":
        for dt in awn_data_types.keys():
            if dt not in sdt:
                sdt[dt] = {}
            for an, av in awn_data_types[dt].items():  # Archive name and value
                if an not in sdt[dt]:
                    sdt[dt][an] = copy.deepcopy(av)
                    if an == "default":
                        continue

                    if "channels" not in sdt[dt][an]:
                        sdt[dt][an]["channels"] = copy.copy(site["beams"])
                    channels = sdt[dt][an]["channels"]
                    # Format the path
                    if not hasattr(sdt[dt][an]["path"], "__call__"):
                        sdt[dt][an]["path"] = sdt[dt][an]["path"].format(
                            project=project_name,
                            project_lc=project_name_lc,
                            site=site,
                            site_lc=site_lc,
                            data_type=dt,
                            data_type_lc=dt.lower(),
                            archive=an,
                            archive_lc=an.lower(),
                            channels=channels,
                            channels_lc=[c.lower() for c in channels],
                        )

                elif sdt[dt] is None:
                    # None used as a placeholder to prevent automatic population, now clean up
                    del sdt[dt]
                    continue

                # RioRawPower needs a linearization function for conversion to RioPower
                if dt == "RioRawPower" and "RioRawPower_linearization_table" not in site:
                    file = Path(__file__).parent / f"riorawpower_linearize_{project_name_lc}_{site_lc}.txt"
                    if file.exists():
                        logger.debug(f"Reading linearization file {file}")
                        site["RioRawPower_linearization_table"] = np.loadtxt(file, unpack=True)

    # RioQDC needs a duration, cadence, and offset in order to calculate which QDC to use.
    required_qdc_keys = {"qdc_cadence", "qdc_offset"}
    site_keys = set(site.keys())
    if len(set(required_qdc_keys).intersection(site_keys)) == 0:
        # None of the required keys are present, add them all.
        site["qdc_cadence"] = np.timedelta64(14, "D")  # How often QDCs occur
        site["qdc_offset"] = np.timedelta64(8, "D")  # How to adjust the start time
        # QDCs can be created from a longer period of duration than their cadence.
        site["qdc_create_duration"] = np.timedelta64(18, "D")
        # When using a longer period the start time may need adjustment
        site["qdc_create_offset"] = np.timedelta64(6, "D")  # How to adjust the start time


project = {
    "name": "Lancaster University Riometer Network",
    "abbreviation": "LANCSRIO",
    "url": "https://www.lancaster.ac.uk/physics/",  # No project website, use Physics Department
    "sites": sites,
}

ap.add_project(project_name, project)
