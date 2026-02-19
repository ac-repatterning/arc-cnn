"""Module architecture.py"""
import numpy as np
import tensorflow as tf

import src.elements.intermediary as itr
import src.elements.master as mr
import src.elements.sequences as sq
import src.modelling.artefacts
import src.modelling.estimates
import src.modelling.sequencing
import src.modelling.scaling


class Architecture:
    """
    Architecture
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments
        self.__patience = self.__arguments.get('modelling').get('patience')
        self.__epochs = self.__arguments.get('modelling').get('epochs')

        # Instances
        self.__estimates = src.modelling.estimates.Estimates(arguments=self.__arguments)
        self.__scaling = src.modelling.scaling.Scaling(arguments=arguments)

    def __get_sequences(self, intermediary: itr.Intermediary) -> sq.Sequences:
        """

        :param intermediary:
        :return:
        """

        seq = src.modelling.sequencing.Sequencing(arguments=self.__arguments)
        x_tr, y_tr = seq.exc(blob=intermediary.training)
        x_te, y_te = seq.exc(blob=intermediary.testing)

        return sq.Sequences(x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)

    # noinspection PyUnresolvedReferences
    def __model(self, x_tr: np.ndarray, y_tr: np.ndarray, units: int, batch_size: int) -> tf.keras.models.Sequential:
        """

        :param x_tr:
        :param y_tr:
        :param units:
        :param batch_size:
        :return:
        """

        architecture = tf.keras.models.Sequential()
        architecture.add(tf.keras.layers.Input(shape=(x_tr.shape[1], x_tr.shape[2])))
        architecture.add(tf.keras.layers.LSTM(units=units, return_sequences=False))
        architecture.add(tf.keras.layers.Dense(units=1))

        # error w.r.t. training data
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.__arguments.get('modelling').get('monitor'), patience=self.__patience, mode='min',
            min_delta=self.__arguments.get('modelling').get('min_delta'))

        architecture.compile(
            loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

        architecture.fit(
            x=x_tr, y=y_tr, epochs=self.__epochs, batch_size=batch_size, callbacks=[early_stopping])

        return architecture

    # pylint: disable=R0915
    def exc(self, master: mr.Master) -> str:
        """

        :param master:
        :return:
        """

        # scaling
        intermediary: itr.Intermediary = self.__scaling.exc(master=master)

        # get sequential structure
        sequences = self.__get_sequences(intermediary=intermediary)

        # Modelling
        j = -1
        model = None
        hyperparameters = {}
        for units in self.__arguments.get('modelling').get('units'):

            for batch_size in self.__arguments.get('modelling').get('batch_size'):

                j = j + 1

                # noinspection PyUnresolvedReferences
                cell: tf.keras.models.Sequential = self.__model(
                    x_tr=sequences.x_tr, y_tr=sequences.y_tr, units=units, batch_size=batch_size)
                latest = min(cell.history.history['loss'])

                if j == 0:
                    model = cell
                    hyperparameters = {'units': units, 'batch_size': batch_size}
                    continue

                previous = min(model.history.history['loss'])
                if latest < previous:
                    model = cell
                    hyperparameters = {'units': units, 'batch_size': batch_size}

        # Hence
        src.modelling.artefacts.Artefacts(
            model=model, scaler=intermediary.scaler, arguments=self.__arguments, path=master.path).exc(
            hyperparameters=hyperparameters)
        self.__estimates.exc(
            model=model, sequences=sequences, intermediary=intermediary, master=master)

        return '/'.join(master.path.rsplit(sep='/', maxsplit=2)[-2:])
