"""Module artefacts.py"""
import logging
import os

import pandas as pd
import sklearn
import tensorflow as tf

import src.functions.objects
import src.functions.streams
import src.modelling.timings


class Artefacts:
    """
    Artefacts
    """

    def __init__(self, model: tf.keras.models.Sequential, scaler: sklearn.preprocessing.MinMaxScaler,
                 arguments: dict, path: str):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__model = model
        self.__scaler = scaler
        self.__arguments = arguments
        self.__path = path

        # Times
        self.__starting = src.modelling.timings.Timings(arguments=self.__arguments).starting()

        # Instances
        self.__streams = src.functions.streams.Streams()
        self.__objects = src.functions.objects.Objects()

    def __modelling(self, hyperparameters: dict) -> str:
        """

        :param hyperparameters:
        :return:
        """

        elements: dict = self.__arguments.get('modelling')

        values = {
            'fields': elements.get('fields'), 'targets': elements.get('targets'),
            'n_sequence': self.__arguments.get('n_sequence'),
            'epochs': self.__model.history.params.get('epochs'),
            'units': hyperparameters.get('units'),
            'batch_size': hyperparameters.get('batch_size'),
            'training_starts': {
                'epoch_milliseconds': self.__starting.epoch_milliseconds,
                'string': self.__starting.string
            }
        }

        return self.__objects.write(nodes=values, path=os.path.join(self.__path, 'modelling.json'))

    def __history(self) -> str:
        """

        :return:
        """

        history = pd.DataFrame(data=self.__model.history.history)

        return self.__streams.write(blob=history, path=os.path.join(self.__path, 'history.csv'))

    def __scaling(self) -> str:
        """

        :return:
        """

        values = {
            'data_max_': list(self.__scaler.data_max_), 'data_min_': list(self.__scaler.data_min_),
            'data_range_': list(self.__scaler.data_range_),
            'feature_names_in_': list(self.__scaler.feature_names_in_),
            'n_features_in_': self.__scaler.n_features_in_,
            'feature_names_out': list(self.__scaler.get_feature_names_out()),
            'n_samples_seen_': self.__scaler.n_samples_seen_
        }

        return self.__objects.write(nodes=values, path=os.path.join(self.__path, 'scaling.json'))

    def exc(self, hyperparameters: dict) -> None:
        """

        :param hyperparameters:
        :return:
        """

        try:
            self.__model.save(filepath=os.path.join(self.__path, 'model'), overwrite=True, zipped=False)
        except OSError as err:
            raise err from err

        self.__modelling(hyperparameters=hyperparameters)
        self.__history()
        self.__scaling()

        logging.info('%s artefacts: succeeded', os.path.basename(self.__path))
