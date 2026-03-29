import copy
import logging
import re
from typing import List, Tuple

from urllib.request import urlopen
from urllib.parse import urlparse

import numpy as np
import numpy.fft
import matplotlib.pyplot as plt

import auroraplot as ap
import auroraplot.riodata.qdc_algorithms
from auroraplot import get_site_info
from auroraplot.data import Data, DataProcessingError, generic_load_converter
import auroraplot.dt64tools as dt64
import auroraplot.tools
import math
import scipy.interpolate
import warnings

logger = logging.getLogger(__name__)


class RioData(Data):
    """Class to manipulate and display riometer data."""

    def __init__(
        self,
        project=None,
        site=None,
        channels=None,
        start_time=None,
        end_time=None,
        sample_start_time=np.array([], dtype="datetime64[s]"),
        sample_end_time=np.array([], dtype="datetime64[s]"),
        integration_interval=None,
        nominal_cadence=None,
        data=None,
        units=None,
        sort=None,
        processing=None,
    ):
        super().__init__(
            project=project,
            site=site,
            channels=channels,
            start_time=start_time,
            end_time=end_time,
            sample_start_time=sample_start_time,
            sample_end_time=sample_end_time,
            integration_interval=integration_interval,
            nominal_cadence=nominal_cadence,
            data=data,
            units=units,
            sort=sort,
            processing=processing,
        )

    @classmethod
    def calculate_qdc_start_time(cls, t: np.datetime64, project: str, site: str) -> np.datetime64:
        """Calculate the standard QDC start time for a given date"""
        qdc_cadence = get_site_info(project, site, "qdc_cadence")
        qdc_offset = get_site_info(project, site, "qdc_offset")
        return dt64.floor(t, qdc_cadence) + qdc_offset

    @classmethod
    def calculate_qdc_create_times(
        cls, t: np.datetime64, project: str, site: str
    ) -> Tuple[np.datetime64, np.datetime64, np.datetime64]:
        """Calculate the nominal start time, and the default create start and end times when creating a QDC"""
        site_info = get_site_info(project, site)
        qdc_cadence = site_info["qdc_cadence"]
        qdc_offset = site_info["qdc_offset"]
        qdc_create_duration = site_info.get("qdc_create_duration", qdc_cadence)
        qdc_create_offset = site_info.get("qdc_create_offset", qdc_offset)
        qdc_t = dt64.floor(t, qdc_cadence) + qdc_offset
        qdc_load_st = dt64.floor(t, qdc_cadence) + qdc_create_offset
        qdc_load_et = qdc_load_st + qdc_create_duration
        return qdc_t, qdc_load_st, qdc_load_et

    def data_description(self):
        return "Riometer data"

    def assert_valid(self):
        Data.assert_valid(self)
        self.get_site_info("beams")

    def get_beam_index(self, channels: List[str] = None):
        """Get the index into the site's beams list for the channels in this object"""
        if channels is None:
            channels = self.channels
        if isinstance(channels, str):
            channels = [channels]
        r = []
        all_beams = list(self.get_site_info("beams"))
        for c in channels:
            r.append(all_beams.index(c))
        return r


class RioRawPower(RioData):
    """Class to manipulate and display riometer unlinearized received power data."""

    def __init__(
        self,
        project=None,
        site=None,
        channels=None,
        start_time=None,
        end_time=None,
        sample_start_time=np.array([], dtype="datetime64[s]"),
        sample_end_time=np.array([], dtype="datetime64[s]"),
        integration_interval=None,
        nominal_cadence=None,
        data=None,
        units=None,
        processing=(),
        sort=None,
    ):
        super().__init__(
            project=project,
            site=site,
            channels=channels,
            start_time=start_time,
            end_time=end_time,
            sample_start_time=sample_start_time,
            sample_end_time=sample_end_time,
            integration_interval=integration_interval,
            nominal_cadence=nominal_cadence,
            data=data,
            units=units,
            processing=processing,
            sort=sort,
        )

    def data_description(self):
        return "Riometer unlinearized received power"

    def linearize(self):
        r = RioPower(
            project=self.project,
            site=self.site,
            channels=copy.copy(self.channels),
            start_time=copy.copy(self.start_time),
            end_time=copy.copy(self.end_time),
            sample_start_time=copy.copy(self.sample_start_time),
            sample_end_time=copy.copy(self.sample_end_time),
            integration_interval=copy.copy(self.integration_interval),
            nominal_cadence=copy.copy(self.nominal_cadence),
            data=np.zeros_like(self.data),
            units=self.units,
            processing=copy.copy(self.processing),
        )
        interp_function = self.get_site_info().get("RioRawPower_linearization_function")
        if interp_function is not None:
            r = interp_function(self)
            r.add_processing(f"Linearized by {interp_function.__name__}")
            return r

        interp_data = self.get_site_info().get("RioRawPower_linearization_table")
        if interp_data is not None:
            for cn in range(len(self.channels)):
                beam_index = self.get_beam_index([self.channels[cn]])[0]
                r.data[cn] = scipy.interpolate.interp1d(
                    interp_data[beam_index + 1], interp_data[0], bounds_error=False
                )(self.data[cn])
            r.add_processing(f"Linearized from site data")
            return r

        raise DataProcessingError(f"No way to linearize {self.format_project_site()} data")


class RioPower(RioData):
    """Class to manipulate and display riometer received power data."""

    def __init__(
        self,
        project=None,
        site=None,
        channels=None,
        start_time=None,
        end_time=None,
        sample_start_time=np.array([], dtype="datetime64[s]"),
        sample_end_time=np.array([], dtype="datetime64[s]"),
        integration_interval=None,
        nominal_cadence=None,
        data=None,
        units=None,
        processing=(),
        sort=None,
    ):
        super().__init__(
            project=project,
            site=site,
            channels=channels,
            start_time=start_time,
            end_time=end_time,
            sample_start_time=sample_start_time,
            sample_end_time=sample_end_time,
            integration_interval=integration_interval,
            nominal_cadence=nominal_cadence,
            data=data,
            units=units,
            processing=processing,
            sort=sort,
        )

    @staticmethod
    def load_from_raw_power(
        file_name: str,
        archive_data: dict,
        project: str,
        site: str,
        data_type: Data,
        start_time: np.datetime64,
        end_time: np.datetime64,
        archive: str,
        channels,
        **kwargs,
    ):
        assert data_type == "RioPower", "incorrect data type to use for load_from_raw_power()"
        raw_power = generic_load_converter(
            file_name, archive_data, project, site, "RioRawPower", start_time, end_time, archive, channels, **kwargs
        )
        return raw_power.linearize()

    def data_description(self):
        return "Riometer received power"

    def plot_with_qdc(self, qdc, fit_err_func=None, channels=None, **kwargs):
        if qdc is not None:
            if channels is None:
                channels = self.channels
            # Only plot channels in QDC and data, in order
            channels = list(set(channels) & set(qdc.channels))
            channels = [c for c in self.channels if c in channels]
            self.plot(channels=channels, **kwargs)
            qdc.align(self, fit_err_func=fit_err_func).plot(figure=plt.gcf(), channels=channels, **kwargs)

    def make_qdc(self, algorithm, channels=None, base_qdc=None, cadence=np.timedelta64(5, "s"), smooth=True):
        if channels is None:
            channels = self.get_channels()

        if self.get_cadence() == cadence:
            rio_power = self
        else:
            rio_power = self.set_cadence(cadence, inplace=False)

        sidt_num_samples = math.ceil(dt64.get_sidereal_day("ms") / cadence)
        sample_times = dt64.mean(rio_power.get_sample_start_time(), rio_power.get_sample_end_time())
        sidt_offset, sidt_day = dt64.get_sidereal_time_offset(
            sample_times, rio_power.get_site_info("longitude"), rio_power.get_start_time()
        )
        offsets: int = np.floor(sidt_offset / cadence).astype(int) % sidt_num_samples
        sidt_days = list(range(np.max(sidt_day) + 1))

        qdc_sample_start_time = np.array(list(dt64.dt64_range(cadence * 0, dt64.get_sidereal_day("ms"), cadence)))
        qdc_sample_end_time = qdc_sample_start_time + cadence
        qdc_data = np.ones([len(channels), sidt_num_samples], dtype=float)
        qdc = RioQDC(
            project=rio_power.get_project(),
            site=rio_power.get_site(),
            channels=channels,
            start_time=rio_power.get_start_time(),
            end_time=rio_power.get_end_time(),
            sample_start_time=qdc_sample_start_time,
            sample_end_time=qdc_sample_end_time,
            # integration_interval=None,
            nominal_cadence=cadence,
            data=qdc_data,
            units=rio_power.get_units(),
            processing=algorithm.get_processing(),
        )
        for c in channels:
            cidx = rio_power.get_channel_index(c)
            data_list = []
            for d in sidt_days:
                sid_data = np.ones([sidt_num_samples], dtype=float) * math.nan
                sidt_d_idx = d == sidt_day  # Elements from the correct sidereal day
                sid_data[offsets[sidt_d_idx]] = rio_power.get_data()[cidx, sidt_d_idx]
                if np.isnan(sid_data[-1]):
                    # No data for last sample, so use the mean of the previous and the first (ie surrounding samples) to
                    # help ensure everything lines up.
                    sid_data[-1] = np.nanmean(sid_data[[0, -2]])
                data_list.append(sid_data)
            # Assign into the output QDC. Must use the channel index from the output object as channels can differ.
            qdc_data[qdc.get_channel_index(c), :] = algorithm.process(data_list)
        qdc.data = qdc_data
        return qdc


class RioAbs(RioData):
    """Class to manipulate and display riometer absorption data."""

    def __init__(
        self,
        project=None,
        site=None,
        channels=None,
        start_time=None,
        end_time=None,
        sample_start_time=np.array([], dtype="datetime64[s]"),
        sample_end_time=np.array([], dtype="datetime64[s]"),
        integration_interval=None,
        nominal_cadence=None,
        data=None,
        units=None,
        sort=None,
        processing=None,
    ):
        super().__init__(
            project=project,
            site=site,
            channels=channels,
            start_time=start_time,
            end_time=end_time,
            sample_start_time=sample_start_time,
            sample_end_time=sample_end_time,
            integration_interval=integration_interval,
            nominal_cadence=nominal_cadence,
            data=data,
            units=units,
            sort=sort,
            processing=processing,
        )

    def data_description(self):
        return "Riometer absorption"


class RioQDC(RioData):
    """Class to load and manipulate Riometer quiet-day curves (QDC)."""

    def __init__(
        self,
        project=None,
        site=None,
        channels=None,
        start_time=None,
        end_time=None,
        sample_start_time=np.array([]),
        sample_end_time=np.array([]),
        integration_interval=np.array([]),
        nominal_cadence=None,
        data=np.array([]),
        units=None,
        sort=None,
        processing=None,
    ):
        super().__init__(
            project=project,
            site=site,
            channels=channels,
            start_time=start_time,
            end_time=end_time,
            sample_start_time=sample_start_time,
            sample_end_time=sample_end_time,
            integration_interval=integration_interval,
            nominal_cadence=nominal_cadence,
            data=data,
            units=units,
            sort=sort,
            processing=processing,
        )

    def plot(
        self,
        channels=None,
        figure=None,
        axes=None,
        subplot=None,
        units_prefix=None,
        title=None,
        start_time=None,
        end_time=None,
        time_units=None,
        **kwargs,
    ):

        if start_time is None:
            # start_time = self.start_time
            start_time = self.get_sample_start_time()[0]
        if end_time is None:
            # end_time = self.end_time
            end_time = self.get_sample_end_time()[-1]

        r = RioData.plot(
            self,
            channels=channels,
            figure=figure,
            axes=axes,
            subplot=subplot,
            units_prefix=units_prefix,
            title=title,
            start_time=start_time,
            end_time=end_time,
            time_units=time_units,
            **kwargs,
        )
        return r

    def data_description(self):
        return "Riometer QDC"
