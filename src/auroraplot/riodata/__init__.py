from __future__ import annotations
import copy
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

import auroraplot as ap
from auroraplot import get_site_info
from auroraplot.data import Data, DataProcessingError, generic_load_converter
import auroraplot.dt64tools as dt64
import math
import scipy.interpolate

logger = logging.getLogger(__name__)


def load_qdc(
    project,
    site,
    time,
    archive=None,
    channels=None,
    path=None,
    tries=1,
    realtime=False,
    load_function=None,
    full_output=False,
) -> Optional[Union[RioQDC, dict]]:
    data_type = "RioQDC"
    archive, ad = ap.get_archive_info(project, site, data_type, archive=archive)
    if channels is not None:
        # Ensure it is a 1D numpy array
        channels = np.array(channels).flatten()
        for c in channels:
            if c not in ad["channels"]:
                raise ValueError("Unknown channel (%s)" % str(c))
    else:
        channels = ad["channels"]

    if path is None:
        path = ad["path"]

    if load_function is None:
        load_function = ad.get("load_function", None)

    if tries is None:
        tries = 1

    if load_function:
        # Pass responsibility for loading to some other
        # function. Parameters have already been checked.
        return load_function(
            project,
            site,
            data_type,
            time,
            archive=archive,
            channels=channels,
            path=path,
            tries=tries,
            full_output=full_output,
        )
    data = []

    t = dt64.get_start_of_month(time)

    if realtime:
        # For realtime use the QDC for the month is (was) not
        # available, so use the previous month's QDC
        t = dt64.get_start_of_previous_month(t)

        # Early in the month the previous motnh's QDC was probably not
        # computed, so use the month before that
        qdc_rollover_day = ad.get("qdc_rollover_day", 4)
        if dt64.get_day_of_month(time) < qdc_rollover_day:
            t = dt64.get_start_of_previous_month(t)

    for n in range(tries):
        try:
            if hasattr(path, "__call__"):
                # Function: call it with relevant information to get the path
                file_name = path(t, project=project, site=site, data_type=data_type, archive=archive, channels=channels)
            else:
                file_name = dt64.strftime(t, path)

            logger.info("loading " + file_name)

            r = ad["load_converter"](
                file_name,
                ad,
                project=project,
                site=site,
                data_type=data_type,
                start_time=np.timedelta64(0, "h"),
                end_time=np.timedelta64(24, "h"),
                archive=archive,
                channels=channels,
                path=path,
            )
            if r is not None:
                r.extract(inplace=True, channels=channels)
                return dict(rioqdc=r, tries=n + 1, maxtries=tries) if full_output else r

        finally:
            # Go to start of previous month
            t = dt64.get_start_of_month(t - np.timedelta64(24, "h"))

    return None


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

    def calculate_absorption(
        self, qdc: Optional[RioQDC] = None, obliquity_factors=None, qdc_archive=None, qdc_tries=1
    ) -> RioAbs:
        if qdc is None:
            qdc = load_qdc(
                project=self.get_project(),
                site=self.get_site(),
                time=self.get_start_time(),
                archive=qdc_archive,
                channels=self.get_channels(),
            )
        aligned_qdc = qdc.align(self)

        r = RioAbs(
            project=self.get_project(),
            site=self.get_site(),
            channels=self.get_channels(),
            start_time=self.get_start_time(),
            end_time=self.get_end_time(),
            sample_start_time=self.get_sample_start_time(),
            sample_end_time=self.get_sample_end_time(),
            integration_interval=self.get_integration_interval(),
            nominal_cadence=self.get_nominal_cadence(),
            data=None,
            units="dB",
            sort=False,
            processing=None,
        )
        r.data = aligned_qdc.get_data() - self.get_data()
        if obliquity_factors is not None:
            if np.isscalar(obliquity_factors):
                r.data /= obliquity_factors
            elif isinstance(obliquity_factors, np.ndarray):
                if obliquity_factors.ndim == 1 and obliquity_factors.size == len(self.get_channels()):
                    for cn in range(len(self.get_channels())):
                        r.data[cn, :] /= obliquity_factors[cn]
                elif obliquity_factors.shape == r.data.shape:
                    r.data /= obliquity_factors
                else:
                    raise ValueError(
                        f"obliquity_factors shape ({obliquity_factors.shape}) incompatible with results shape {r.data.shape}"
                    )
        return r


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
        if title is None:
            title = self.make_title(sidereal_day=True)

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

    def align(
        self,
        align_to: Data = None,
    ) -> RioPower:

        if isinstance(align_to, RioQDC) or not isinstance(align_to, Data):
            raise TypeError(f"Cannot align a {type(self)} type to a {type(align_to)} type")

        align_to_mst = align_to.get_mean_sample_time()
        # time_units = dt64.get_units(align_to_mst)
        time_units = "ms"
        sidereal_day = dt64.get_sidereal_day(time_units)
        xi = dt64.get_sidereal_time_offset(align_to_mst, align_to.get_site_info("longitude"), units=time_units)
        xi = xi.astype(f"timedelta64[{time_units}]").astype("float64")
        print(f"xi: {xi}   {xi.shape}   {xi.dtype}")

        # Get a numpy array of all the mean sample times
        mst = self.get_mean_sample_time()
        # Want an array where the ends wrap
        mst_idx = list(range(len(mst)))
        mst_idx.append(0)
        mst_idx.insert(0, len(mst) - 1)
        x = mst[mst_idx]
        # Fix the copied values
        x[0] -= sidereal_day
        x[-1] += sidereal_day
        x = x.astype(f"timedelta64[{time_units}]").astype("float64")
        print(f"x: {x}   {x.shape}   {x.dtype}")

        r = RioPower(
            project=self.get_project(),
            site=self.get_site(),
            channels=self.get_channels(),
            start_time=align_to.get_start_time(),
            end_time=align_to.get_end_time(),
            sample_start_time=align_to_mst,
            sample_end_time=align_to_mst,
            units=self.units,
            data=np.ndarray([len(self.get_channels()), len(align_to_mst)], dtype="float"),
        )
        r.data[:] = np.nan
        print(r)
        print(f"x: {x.shape}")
        print(f"xi: {xi.shape}")

        for cn in range(len(self.get_channels())):
            # y = [self.data[cn, -1], self.data[cn, :], self.data[cn, 0]]
            y = self.data[cn, mst_idx]
            print(f"y: {y.shape}")
            interp_func = scipy.interpolate.interp1d(x, y, kind="linear")

            r.data[cn, :] = interp_func(xi)

        return r
