"""Module timings.py"""
import collections
import datetime
import time


class Timings:
    """
    Timings
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments

        # Set up a collection
        self.__starting = collections.namedtuple(
            typename='starting', field_names=['epoch_milliseconds', 'string'])

    def starting(self):
        """

        :return:
        """

        spanning = self.__arguments.get('spanning')
        as_from = datetime.date.today() - datetime.timedelta(days=round(spanning*365.25))
        epoch_milliseconds = 1000 * time.mktime(as_from.timetuple())

        beginning = time.gmtime(epoch_milliseconds/1000)
        string = time.strftime('%Y-%m-%d %H:%M:%S', beginning)

        return self.__starting(epoch_milliseconds=epoch_milliseconds, string=string)
