"""Module interface.py"""
import sys

import pandas as pd

import src.assets.foci
import src.assets.gauges
import src.assets.partitions
import src.elements.partitions as pr
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.cache


class Interface:
    """
    Notes<br>
    ------<br>

    Reads-in the assets.
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.
        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__service = service
        self.__s3_parameters = s3_parameters
        self.__arguments = arguments

    @staticmethod
    def __structure(partitions: pd.DataFrame) -> list[pr.Partitions]:
        """

        :param partitions: The time series partitions.
        :return:
        """

        values: list[dict] = partitions.copy().reset_index(drop=True).to_dict(orient='records')

        return [pr.Partitions(**value) for value in values]

    def exc(self):
        """

        :return:
        """

        # Applicable time series metadata, i.e., gauge, identification codes
        gauges = src.assets.gauges.Gauges(
            service=self.__service, s3_parameters=self.__s3_parameters, arguments=self.__arguments).exc()
        if gauges.empty:
            src.functions.cache.Cache().exc()
            sys.exit('There are no data sets for model development.')

        partitions, listings = src.assets.partitions.Partitions(
            gauges=gauges, arguments=self.__arguments).exc()

        return self.__structure(partitions=partitions), listings
