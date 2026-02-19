"""Module sequences.ph"""
import typing

import numpy as np


class Sequences(typing.NamedTuple):
    """
    The data type class ⇾ Sequences<br><br>

    Attributes<br>
    ----------<br>
    <b>x_tr</b> : numpy.ndarray
        The input sequences vis-à-vis training data.<br>

    <b>y_tr</b> : numpy.ndarray
        The target sequences vis-à-vis training data.<br>

    <b>x_te</b> : numpy.ndarray
        The input sequences vis-à-vis testing data.<br>

    <b>y_te</b> : numpy.ndarray
        The target sequences vis-à-vis testing data.<br>

    """

    x_tr: np.ndarray
    y_tr: np.ndarray
    x_te: np.ndarray
    y_te: np.ndarray
