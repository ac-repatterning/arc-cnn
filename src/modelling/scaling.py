"""Module scaling.py"""
import typing

import numpy as np
import pandas as pd
import sklearn

import src.elements.master as mr
import src.elements.intermediary as itr


class Scaling:
    """
    Minimum Maximum Scaling
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__features = arguments.get('scaling').get('features')

    def __restructure(self, structure: pd.DataFrame, transforms: np.ndarray) -> pd.DataFrame:
        """

        :param structure: The dataframe the transforms apply to
        :param transforms: The scaled forms of the fields of self.__features
        :return:
        """

        __data = structure.drop(columns=self.__features)
        __data.loc[:, self.__features] = transforms

        return __data

    def preimage(self, blob: pd.DataFrame) -> typing.Tuple[pd.DataFrame, sklearn.preprocessing.MinMaxScaler]:
        """

        :param blob: The training data
        :return:
            frame: A data frame wherein the features in focus have been scaled.
            scaler: The scaler object vis-à-vis the training data.
        """

        scaler = sklearn.preprocessing.MinMaxScaler()

        # Creating and using a scaler
        transforms = scaler.fit_transform(blob.copy()[self.__features])
        frame = self.__restructure(structure=blob, transforms=transforms)

        return frame, scaler

    def image(self, blob: pd.DataFrame, scaler: sklearn.preprocessing.MinMaxScaler) -> pd.DataFrame:
        """
        For transforming data sets associated with the training data that built the scaler

        :param blob: Includes that need to be transformed
        :param scaler: A scaler object
        """

        transforms = scaler.transform(blob.copy()[self.__features])

        return self.__restructure(structure=blob, transforms=transforms)

    def exc(self, master: mr.Master) -> itr.Intermediary:
        """

        :param master: Refer to src.elements.master
        :return:
        """

        training, scaler = self.preimage(blob=master.training)
        testing = self.image(blob=master.testing, scaler=scaler)

        intermediary = itr.Intermediary(training=training, testing=testing, scaler=scaler)

        return intermediary
