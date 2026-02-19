"""Module split.py"""
import os

import pandas as pd

import config
import src.elements.master as mr
import src.elements.partitions as pr
import src.functions.directories
import src.functions.streams


class Split:
    """
    The training & testing splits.
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__arguments = arguments
        self.__n_exclude = self.__arguments.get('n_points_testing') + self.__arguments.get('n_sequence')

        self.__configurations = config.Config()
        self.__directories = src.functions.directories.Directories()
        self.__streams = src.functions.streams.Streams()

    def __training(self, blob: pd.DataFrame) -> pd.DataFrame:
        """
        <b>Note:</b><br>
        n_points_training = training.shape[0] - self.__arguments.get('n_sequence')<br><br>

        :param blob:
        :return:
        """

        return blob.copy()[:-self.__n_exclude]

    def __testing(self, blob: pd.DataFrame) -> pd.DataFrame:
        """
        ascertains the testing-data split has the appropriate number of instances

        :param blob:
        :return:
        """

        return blob.copy()[-self.__n_exclude:]

    def __persist(self, blob: pd.DataFrame, pathstr: str) -> None:
        """

        :param blob:
        :param pathstr:
        :return:
        """

        self.__streams.write(blob=blob, path=pathstr)

    def exc(self, data: pd.DataFrame, partition: pr.Partitions) -> mr.Master:
        """

        :param data: The data set consisting of the attendance numbers of <b>an</b> institution/hospital.
        :param partition: The time series & catchment identification codes of a gauge.
        :return:
        """

        frame = data.copy()
        frame.sort_values(by='timestamp', ascending=True, inplace=True)

        # Split
        training = self.__training(blob=frame)
        testing = self.__testing(blob=frame)

        # Path
        path = os.path.join(self.__configurations.pathway_, str(partition.catchment_id), str(partition.ts_id))
        self.__directories.create(path=path)

        # Persist
        for instances, name in zip([frame, training, testing], ['data.csv', 'training.csv', 'testing.csv']):
            self.__persist(blob=instances, pathstr=os.path.join(path, name))

        return mr.Master(training=training, testing=testing, path=path)
