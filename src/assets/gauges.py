"""Module gauges.py"""
import numpy as np
import pandas as pd

import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.s3.prefix


class Gauges:
    """
    Retrieves the catchment & time series codes of the gauges in focus.
    """

    def __init__(self, service: sr.Service, s3_parameters: s3p.S3Parameters, arguments: dict):
        """

        :param service:
        :param s3_parameters:
        :param arguments:
        """

        self.__service = service
        self.__s3_parameters = s3_parameters
        self.__arguments = arguments

        # An instance for interacting with objects within an Amazon S3 prefix
        self.__pre = src.s3.prefix.Prefix(
            service=self.__service, bucket_name=self.__s3_parameters.internal)

    @staticmethod
    def __get_elements(objects: list[str]) -> pd.DataFrame:
        """

        :param objects:
        :return:
        """

        # A set of S3 uniform resource locators
        values = pd.DataFrame(data={'uri': objects})

        # Splitting locators
        rename = {0: 'endpoint', 1: 'catchment_id', 2: 'ts_id', 3: 'name'}
        splittings = values['uri'].str.rsplit('/', n=3, expand=True)
        splittings.rename(columns=rename, inplace=True)
        splittings['date'] = splittings['name'].str.replace('.csv', '')

        # Collating
        values = values.copy().join(splittings, how='left')

        return values

    def __get_keys(self) -> list[str]:
        """
        cf. self.__pre.objects(prefix=paths[0], delimiter=''),
        self.__pre.objects(prefix=paths[0], delimiter='/')

        :return:
        """

        paths = self.__pre.objects(
            prefix=(self.__s3_parameters.path_internal_data + f"{self.__arguments['s3']['affix']}/"),
            delimiter='/')

        computations = []
        for path in paths:
            listings = self.__pre.objects(prefix=path, delimiter='')
            computations.append(listings)
        keys: list[str] = sum(computations, [])

        return keys

    def exc(self) -> pd.DataFrame:
        """

        :return: The fields &Rarr; <br>
            <ul><li>uri</li>
                <li>catchment_id</li>
                <li>ts_id</li>
                <li>date</li>
            </ul>
        """

        keys = self.__get_keys()
        if len(keys) > 0:
            objects = [f's3://{self.__s3_parameters.internal}/{key}' for key in keys]
        else:
            return pd.DataFrame()

        # The variable objects is a list of uniform resource locators.  Each locator includes a 'ts_id',
        # 'catchment_id', 'datestr' substring; the function __get_elements extracts these items.
        values = self.__get_elements(objects=objects)

        # Types
        values['catchment_id'] = values['catchment_id'].astype(dtype=np.int64)
        values['ts_id'] = values['ts_id'].astype(dtype=np.int64)
        values['date'] = pd.to_datetime(values['date'], format='%Y-%m-%d')
        values.drop(columns=['endpoint', 'name'], inplace=True)

        return values
