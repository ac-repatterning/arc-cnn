"""Module estimates.py"""
import logging
import os

import pandas as pd
import tensorflow as tf

import src.elements.intermediary as itr
import src.elements.master as mr
import src.elements.sequences as sq
import src.functions.streams
import src.modelling.valuations


class Estimates:
    """
    Estimates
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__arguments = arguments

        # Instances
        self.__streams = src.functions.streams.Streams()

    def __persist(self, blob: pd.DataFrame, path: str) -> None:
        """

        :param blob:
        :param path:
        :return:
        """

        message = self.__streams.write(blob=blob, path=path)

        logging.info(message)

    def exc(self, model: tf.keras.models.Sequential, sequences: sq.Sequences,
            intermediary: itr.Intermediary, master: mr.Master) -> None:
        """

        :param model:
        :param sequences:
        :param intermediary:
        :param master:
        :return:
        """

        valuations = src.modelling.valuations.Valuations(model=model, scaler=intermediary.scaler, arguments=self.__arguments)

        # training
        training = valuations.exc(x_matrix=sequences.x_tr, design=intermediary.training, original=master.training)
        self.__persist(blob=training, path=os.path.join(master.path, 'e_training.csv'))

        # testing
        testing = valuations.exc(x_matrix=sequences.x_te, design=intermediary.testing, original=master.testing)
        self.__persist(blob=testing, path=os.path.join(master.path, 'e_testing.csv'))
