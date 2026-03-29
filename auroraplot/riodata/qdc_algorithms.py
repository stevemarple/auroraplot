import numpy as np
from typing import List, Optional, NewType


class QdcAlgorithmBase(object):
    def __init__(self):
        pass

    def get_processing(self) -> str:
        raise NotImplementedError("Class must be derived and the process() function implemented")

    def process(self, data: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("Class must be derived and the process() function implemented")


class UpperEnvelope(QdcAlgorithmBase):
    """Create QDC from the upper envelope of all data

    :param rows: list of integers corresponding to the sorted rows, with 0 being the highest values at each sidereal
               time. When the list contains multiple values those rows are combined using numpy.mean.
    """

    def __init__(self, rows: Optional[List[int]] = None):
        super().__init__()
        self.rows: List[int] = [1, 2] if rows is None else list(rows)
        assert len(self.rows) > 0, "rows must not be empty"

    def get_processing(self) -> str:
        return f"Created by UpperEnvelope, rows={self.rows}"

    def process(self, data: List[np.ndarray]) -> np.ndarray:
        """
        Convert riometer data to a single quiet day curve

        :param data: list of numpy arrays. The data is passed as data[sidereal_day_index] = timeseries: np.ndarray

        :return: single quiet day curve, timeseries: np.ndarray

        This function must be called separately for each channel (riometer beam).
        """
        # Combine into a 2D array and sort; Numpy sorts in ascending order, with NaNs last. Prefer descending order with
        # NaNs last. Use *= -1 to get the desired order with NaNs mixed with the smallest values.
        combined_data: np.ndarray = np.array(data) * -1
        combined_data.sort(axis=0)
        combined_data *= -1  # Restore the correct sign
        return np.nanmean(combined_data[self.rows, :], axis=0)


def get_algorithms() -> List:
    return [UpperEnvelope]


def get_default_algorithm():
    return UpperEnvelope
