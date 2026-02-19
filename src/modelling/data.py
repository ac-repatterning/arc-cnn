"""Module data.py"""
import datetime

import dask.dataframe as ddf
import numpy as np
import pandas as pd


class Data:
    """
    Data
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        # Focus
        self.__dtype = {'timestamp': np.float64, 'ts_id': np.float64, 'measure': float}

        # seconds, milliseconds
        as_from: datetime.datetime = (datetime.datetime.now()
                                      - datetime.timedelta(days=round(arguments.get('spanning')*365)))
        self.__as_from = as_from.timestamp() * 1000

    def __get_data(self, listing: list[str]):
        """

        :param listing:
        :return:
        """

        try:
            block: pd.DataFrame = ddf.read_csv(
                listing, header=0, usecols=list(self.__dtype.keys()), dtype=self.__dtype).compute()
        except ImportError as err:
            raise err from err

        block.reset_index(drop=True, inplace=True)
        block.sort_values(by='timestamp', ascending=True, inplace=True)
        block.drop_duplicates(subset='timestamp', keep='first', inplace=True)

        return block

    @staticmethod
    def __set_missing(data: pd.DataFrame) -> pd.DataFrame:
        """
        Forward filling.  In contrast, the variational model inherently deals with missing data, hence
                          it does not include this type of step.

        :param data:
        :return:
        """

        data['measure'] = data['measure'].ffill().values

        return data

    def exc(self, listing: list[str]) -> pd.DataFrame:
        """
        Append a date of the format datetime64[]
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

        :param listing:
        :return:
        """

        # The data
        data = self.__get_data(listing=listing)
        data = self.__set_missing(data=data.copy())

        # Filter
        data = data.copy().loc[data['timestamp'] >= self.__as_from, :]

        return data
