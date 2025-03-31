import numpy as np
import math
from numpy import (pi, sin, cos, tan, arcsin, arccos, arctan, arctan2)
from pprint import pprint
from datetime import datetime, timedelta


def calc_roll_angle(st_vvlh):
    x, y, z = st_vvlh
    return np.degrees(- arctan2(y, z))


def calc_pitch_angle(st_vvlh):
    x, y, z = st_vvlh
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.degrees(arcsin(x / r))


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm_product)
    return np.degrees(angle)


def str2datetime(time_string):
    dt = datetime.strptime(time_string, "%d %b %Y %H:%M:%S.%f")
    return dt


def datetime2str(datetime):
    time_string = datetime.strftime("%d %b %Y %H:%M:%S.%f")
    return time_string[:-3]


def next_dt(datetime, delta=0.25):
    delta = timedelta(seconds=delta)
    next_dt = datetime + delta
    return next_dt
