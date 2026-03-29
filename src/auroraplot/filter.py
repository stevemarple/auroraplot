import copy
import numpy as np
from numpy.typing import ArrayLike
import scipy.ndimage
from typing import Callable, List, Tuple, Union, Literal

from . import dt64tools as dt64
from .data import Data

Scalar = Union[int, float]


class FilterBase(object):
    @classmethod
    def get_name(cls):
        return cls.__name__

    def __init__(self):
        pass

    def __call__(self, data: Data, inplace=False) -> Data:
        raise NotImplementedError("Derived class should implement this method")

    def get_processing(self, data: Data) -> Union[str, List[str]]:
        raise NotImplementedError("Derived class should implement this method")


class WindowedFilterBase(FilterBase):
    @staticmethod
    def get_window_samples(window: np.timedelta64, cadence: np.timedelta64) -> int:
        w = round(window / cadence)
        if w % 2 != 1:
            raise ValueError("Window size must be odd multiple of cadence")
        return w

    def __init__(
        self,
        window: np.timedelta64,
        cadence: np.timedelta64,
    ):
        super().__init__()
        self.window = np.timedelta64(window)
        self.cadence = None if cadence is None else np.timedelta64(cadence)
        self.window_samples = self.get_window_samples(window, cadence)

    def __call__(self, data: Data, inplace=False) -> Data:
        raise NotImplementedError("Derived class should implement this method")

    def _prepare_result(self, data: Data, inplace=False) -> Data:
        result = data if inplace else copy.deepcopy(data)
        data_cadence = data.get_cadence()
        if data_cadence is None or data_cadence != self.cadence:
            result.set_cadence(self.cadence, inplace=True)
        return result


class VectorizedFilter(WindowedFilterBase):
    def __init__(
        self,
        function: Callable[[ArrayLike, Union[int, Tuple[int, ...]]], Scalar],
        window: np.timedelta64,
        cadence: np.timedelta64,
        mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "constant",
        cval: Scalar = np.nan,
    ) -> None:
        super().__init__(window=window, cadence=cadence)
        self.function = function
        self.mode = mode
        self.cval = cval

    def __call__(self, data: Data, inplace=False) -> Data:
        # result = data if inplace else copy.deepcopy(data)
        # data_cadence = data.get_cadence()
        # if data_cadence is None or data_cadence != self.cadence:
        #     result.set_cadence(self.cadence, inplace=True)
        result = self._prepare_result(data, inplace)

        # for cn in range(result.data.shape[0]):
        #     # See https://stackoverflow.com/a/75114333
        #     result.data[cn, :] = np.nanmedian(
        #         np.lib.stride_tricks.sliding_window_view(result.data[cn, :], (self.window_samples,)), axis=1
        #     )
        result.data = scipy.ndimage.vectorized_filter(
            input=result.data,
            function=self.function,
            size=(1, self.window_samples),
            # mode=self.mode,
        )
        result.add_processing(self.get_processing(data))
        return result

    def get_processing(self, data: Data) -> str:
        return f"{__class__.__name__} with function {self.function.__name__ }, window {dt64.to_str(self.window)}, ({self.window_samples} samples)"


class SlidingMedianFilter(VectorizedFilter):
    def __init__(
        self,
        window: np.timedelta64,
        cadence: np.timedelta64,
        mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "constant",
        cval: Scalar = np.nan,
    ) -> None:
        super().__init__(function=np.nanmedian, window=window, cadence=cadence, mode=mode, cval=cval)


class SavitzkyGolayFilter(WindowedFilterBase):
    def __init__(
        self,
        polyorder: int,
        window: np.timedelta64,
        cadence: np.timedelta64,
        mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "constant",
        cval: Scalar = np.nan,
        deriv: int = 0,
        delta: float = 1.0,
    ) -> None:
        super().__init__(
            window=window,
            cadence=cadence,
        )
        self.polyorder = int(polyorder)
        self.mode = mode
        self.cval = cval
        self.deriv = int(deriv)
        self.delta = float(delta)

    def __call__(self, data: Data, inplace=False) -> Data:
        result = self._prepare_result(data, inplace)
        result.data = scipy.signal.savgol_filter(
            result.data,
            self.window_samples,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            mode=self.mode,
            cval=self.cval,
            axis=1,
        )
        result.add_processing(self.get_processing(data))
        return result

    def get_processing(self, data: Data) -> str:
        return f"{__class__.__name__}(polyorder={self.polyorder}, window={dt64.to_str(self.window)}, cadence={self.cadence}, window_samples={self.window_samples})"


class ChainedFilter(FilterBase):
    def __init__(self, *filters: FilterBase) -> None:
        super().__init__()
        self.filters = filters

    def __call__(self, data: Data, inplace=False) -> Data:
        result = data if inplace else copy.deepcopy(data)
        for f in self.filters:
            result = f(result, inplace=True)
        # Each filter will have amended the processing themselves, do not repeat here.
        return result

    def get_processing(self, data: Data) -> Union[str, List[str]]:
        return [f.get_processing(data) for f in self.filters]
