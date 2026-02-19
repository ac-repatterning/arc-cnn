"""Module valuations.py"""
import logging
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf


class Valuations:
    """
    This class extracts the raw predictions of a developed model w.r.t. (with respect to) the model's training data,
    and the model development task's testing data.  Subsequently, the predictions are re-scaled via the applicable scaling
    object's inverse transform function.  Consequently, predictions & original values can be compared.
    """

    def __init__(self, model: tf.keras.models.Sequential, scaler: sklearn.preprocessing.MinMaxScaler, arguments: dict):
        """

        :param model:
        :param scaler:
        :param arguments:
        """

        self.__model = model
        self.__scaler = scaler
        self.__arguments = arguments

        # Scaling Arguments
        self.__features = self.__arguments.get('scaling').get('features')

        # Modelling Arguments
        _, self.__targets, self.__disjoint = self.__get_modelling_arguments()

        # Renaming
        self.__rename = { arg: f'e_{arg}' for arg in self.__targets}

    def __get_modelling_arguments(self):
        """

        :return:
        """

        elements: dict = self.__arguments.get('modelling')
        fields: list = elements.get('fields')
        targets: list = elements.get('targets')

        # The variables present within the [input] fields, but not the targets
        disjoint: list = list(set(fields).difference(set(targets)))

        return fields, targets, disjoint

    def __reconfigure(self, structure: pd.DataFrame) -> pd.DataFrame:
        """

        :param structure:
        :return:
        """


        data: np.ndarray = self.__scaler.inverse_transform(structure.values)
        frame = pd.DataFrame()
        frame.loc[:, self.__features] = data
        frame.rename(columns=self.__rename, inplace=True)

        return frame

    def exc(self, x_matrix: np.ndarray, design: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """

        :param x_matrix: A matrix of sequences
        :param design: Where applicable, the feature fields are scaled
        :param original: cf. design
        :return:
        """

        logging.info('x_matrix shape: %s', x_matrix.shape)

        # Predict
        predictions: np.ndarray = self.__model.predict(x=x_matrix)

        # Points
        n_points = x_matrix.shape[0]

        # The expected inverse transform structure
        structure = design.copy()[self.__disjoint][-n_points:]
        structure.loc[:, self.__targets] = predictions
        structure = structure.copy()[self.__features]

        # Reconfiguring
        frame = self.__reconfigure(structure=structure)

        # Original & Estimates
        __original = original[-n_points:]
        instances = pd.concat([__original.copy().reset_index(drop=True), frame[list(self.__rename.values())]],
                              axis=1)

        return instances
