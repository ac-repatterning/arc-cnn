"""Module intermediary.py"""
import typing

import pandas as pd
import sklearn


class Intermediary(typing.NamedTuple):
    """
    The data type class ⇾ Intermediary<br><br>

    Attributes<br>
    ----------<br>
    <b>training</b> : pandas.DataFrame
        The training data of a gauge<br>

    <b>testing</b> : pandas.DataFrame
        The testing data of a gauge<br>

    <b>scaler</b> : sklearn.preprocessing.MinMaxScaler
        The scaler object<br>

    """

    training: pd.DataFrame
    testing: pd.DataFrame
    scaler: sklearn.preprocessing.MinMaxScaler = None
