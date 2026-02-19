"""Module predicting.py"""
import collections

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import tensorflow_probability.python.sts.components as tfc

import src.elements.master as mr


class Predicting:
    """
    Predicting
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments

    def __times(self, start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
        """

        :param start: The starting time point
        :param periods: The number of time points
        :return:
        """

        return pd.date_range(
            start=start, periods=periods, freq=self.__arguments.get('frequency'), inclusive='left')

    @staticmethod
    def __estimates(data: np.ndarray, times: pd.DatetimeIndex, p_distribution_samples: np.ndarray) -> pd.DataFrame:
        """

        :param data:
        :param times:
        :param p_distribution_samples:
        :return:
        """

        return pd.DataFrame(data={
            'datetime': times,
            'observation': data,
            'lower_q': np.quantile(p_distribution_samples, q=0.25, axis=0),
            'upper_q': np.quantile(p_distribution_samples, q=0.75, axis=0),
            'median': np.quantile(p_distribution_samples, q=0.5, axis=0),
            'lower_w': np.quantile(p_distribution_samples, q=0.10, axis=0),
            'upper_w': np.quantile(p_distribution_samples, q=0.90, axis=0)})

    def exc(self, master: mr.Master, model: tfc.Sum, v_posterior_samples: collections.OrderedDict) -> pd.DataFrame:
        """

        :param master:
        :param model:
        :param v_posterior_samples:
        :return:
        """

        # The value of each future observation is unknown, hence np.nan
        data = pd.concat((master.training['measure'], master.testing['measure']), ignore_index=True).values
        data = np.concat((data, np.repeat(np.nan, self.__arguments.get('ahead'))))

        # Times
        times = self.__times(start=master.training['date'].min(), periods=data.shape[0])

        # Distribution of predictions
        p_distribution = tfp.sts.one_step_predictive(
            model=model,
            observed_time_series=data,
            parameter_samples=v_posterior_samples)
        p_distribution_samples: np.ndarray = p_distribution.distribution.sample(
            self.__arguments.get('n_samples')).numpy()

        # Hence
        estimates = self.__estimates(data=data, times=times, p_distribution_samples=p_distribution_samples)
        estimates['timestamp'] = estimates['datetime'].astype(np.int64) / 1000
        estimates.drop(columns='datetime', inplace=True)

        return estimates
