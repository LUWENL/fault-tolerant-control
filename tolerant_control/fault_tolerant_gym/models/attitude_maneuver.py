import numpy as np
from numpy import (pi, sin, cos, arcsin, arctan2)
from numpy.linalg import norm, inv
import random
from metadata import METADATA
from tolerant_control.fault_tolerant_gym.models.model_utils import str2datetime
from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from astropy import units as u
from astropy.time import Time


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm_product)
    return np.degrees(angle)


def rotate_matrix_x(theta):
    return np.array([[1, 0, 0],
                     [0, cos(theta), sin(theta)],
                     [0, -sin(theta), cos(theta)]])


def rotate_matrix_y(theta):
    return np.array([[cos(theta), 0, -sin(theta)],
                     [0, 1, 0],
                     [sin(theta), 0, cos(theta)]])


def rotate_matrix_z(theta):
    return np.array([[cos(theta), sin(theta), 0],
                     [-sin(theta), cos(theta), 0],
                     [0, 0, 1]])


def get_Reo(true_anomaly, argp, incl, raan, is_radians=False):
    if not is_radians:
        true_anomaly = np.radians(true_anomaly)
        argp = np.radians(argp)
        incl = np.radians(incl)
        raan = np.radians(raan)

    matrix1 = rotate_matrix_x(- pi / 2)
    matrix2 = rotate_matrix_z(true_anomaly + argp + pi / 2)
    matrix3 = rotate_matrix_x(incl)
    matrix4 = rotate_matrix_z(raan)

    matrix = matrix1 @ matrix2 @ matrix3 @ matrix4

    return matrix


def eci2orbit(true_anomaly, argp, incl, raan, xyz_eci, is_radians=False):
    if not is_radians:
        true_anomaly = np.radians(true_anomaly)
        argp = np.radians(argp)
        incl = np.radians(incl)
        raan = np.radians(raan)

    matrix1 = rotate_matrix_x(- pi / 2)
    matrix2 = rotate_matrix_z(true_anomaly + argp + pi / 2)
    matrix3 = rotate_matrix_x(incl)
    matrix4 = rotate_matrix_z(raan)

    matrix = matrix1 @ matrix2 @ matrix3 @ matrix4

    return matrix @ xyz_eci


def orbit2eci(true_anomaly, argp, incl, raan, xyz_orbit, is_radians=False):
    if not is_radians:
        true_anomaly = np.radians(true_anomaly)
        argp = np.radians(argp)
        incl = np.radians(incl)
        raan = np.radians(raan)

    matrix1 = rotate_matrix_x(- pi / 2)
    matrix2 = rotate_matrix_z(true_anomaly + argp + pi / 2)
    matrix3 = rotate_matrix_x(incl)
    matrix4 = rotate_matrix_z(raan)

    matrix = inv(matrix1 @ matrix2 @ matrix3 @ matrix4)

    return matrix @ xyz_orbit


def eci2lla(eci, dt):

    x, y, z = eci
    tt = Time(dt, format='datetime')

    gcrs = GCRS(CartesianRepresentation(x=x * u.km, y=y * u.km, z=z * u.km), obstime=tt)

    itrs = gcrs.transform_to(ITRS(obstime=tt))

    el = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)

    lon, lat, alt = el.to_geodetic('WGS84')

    return np.array([lon.value, lat.value, alt.value])


def lla2eci(lla, dt):

    lon, lat, alt = lla
    earth_location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=alt * u.km)

    tt = Time(dt, format='datetime')

    gcrs = earth_location.get_gcrs(obstime=tt)
    eci = gcrs.represent_as(CartesianRepresentation)

    return np.array([eci.x.value, eci.y.value, eci.z.value])


# roll-pitch-yaw
def calc_roll_angle(st_orbit):
    x, y, z = st_orbit
    return np.degrees(- arctan2(y, z))


def calc_pitch_angle(st_orbit):
    x, y, z = st_orbit
    r = norm(st_orbit)
    return np.degrees(arcsin(x / r))


def get_Rob(roll_angle, pitch_angle, yaw_angle, is_radians=False):
    if not is_radians:
        roll_angle = np.radians(roll_angle)
        pitch_angle = np.radians(pitch_angle)
        yaw_angle = np.radians(yaw_angle)

    matrix = rotate_matrix_x(roll_angle) @ rotate_matrix_y(pitch_angle) @ rotate_matrix_z(yaw_angle)

    return matrix


def get_Rbo(roll_angle, pitch_angle, yaw_angle, is_radians=False):
    if not is_radians:
        roll_angle = np.radians(roll_angle)
        pitch_angle = np.radians(pitch_angle)
        yaw_angle = np.radians(yaw_angle)

    matrix = inv(rotate_matrix_x(roll_angle) @ rotate_matrix_y(pitch_angle) @ rotate_matrix_z(yaw_angle))

    return matrix


def body2orbit(s_body, roll_angle, pitch_angle, yaw_angle, is_radians=False):
    # rotation sequence Roll-Pitch-Yaw
    if not is_radians:
        roll_angle = np.radians(roll_angle)
        pitch_angle = np.radians(pitch_angle)
        yaw_angle = np.radians(yaw_angle)

    matrix = inv(rotate_matrix_z(yaw_angle) @ rotate_matrix_y(pitch_angle) @ rotate_matrix_x(roll_angle))
    s_orbit = matrix @ s_body

    return s_orbit


def orbit2body(s_orbit, roll_angle, pitch_angle, yaw_angle, is_radians=False):
    # rotation sequence Roll-Pitch-Yaw
    if not is_radians:
        roll_angle = np.radians(roll_angle)
        pitch_angle = np.radians(pitch_angle)
        yaw_angle = np.radians(yaw_angle)

    matrix = rotate_matrix_z(yaw_angle) @ rotate_matrix_y(pitch_angle) @ rotate_matrix_x(roll_angle)
    s_body = matrix @ s_orbit

    return s_body


# def geodetic2eci_(lla, time, is_time_str=False):
#     if is_time_str:
#         time = str2datetime(time)
#
#     latitude, longitude, altitude = lla
#     # longitude, latitude, altitude = lla
#
#     # the unit of return is meters
#     x, y, z = geodetic2eci(latitude, longitude, altitude, time)
#
#     # change m to km
#     eci = np.array([x, y, z]) / 1000
#
#     return eci


# def eci2geodetic_(t_eci, time, is_time_str=False):
#     if is_time_str:
#         time = str2datetime(time)
#
#     # change km to m
#     t_eci *= 1000
#     x, y, z = t_eci
#
#     # the unit of input is meters
#     lla = eci2geodetic(x, y, z, time)
#     latitude, longitude, altitude = lla
#     altitude /= 1000
#
#     return np.array([latitude, longitude, altitude])


# def matrix2quaternion(roll_angle, pitch_angle, yaw_angle, is_radians=False):
#     # rotation sequence Roll-Pitch-Yaw
#     if not is_radians:
#         roll_angle = np.radians(roll_angle)
#         pitch_angle = np.radians(pitch_angle)
#         yaw_angle = np.radians(yaw_angle)
#
#     matrix = inv(rotate_matrix_z(yaw_angle) @ rotate_matrix_y(pitch_angle) @ rotate_matrix_x(roll_angle))
#
#     q_4 = np.sqrt(1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2
#     q_1 = (matrix[1, 2] - matrix[2, 1]) / (4 * q_4)
#     q_2 = (matrix[2, 0] - matrix[0, 2]) / (4 * q_4)
#     q_3 = (matrix[0, 1] - matrix[1, 0]) / (4 * q_4)
#
#     return np.array([q_1, q_2, q_3, q_4])


def matrix2quaternion(matrix):
    q_4 = np.sqrt(1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2
    q_1 = (matrix[1, 2] - matrix[2, 1]) / (4 * q_4)
    q_2 = (matrix[2, 0] - matrix[0, 2]) / (4 * q_4)
    q_3 = (matrix[0, 1] - matrix[1, 0]) / (4 * q_4)

    # q_1 =  np.sqrt(1 + matrix[0,0] - matrix[1,1] - matrix[2,2]) / 2
    # q_2 = (matrix[0, 1] + matrix[1, 0]) / (4 * q_1)
    # q_3 = (matrix[0, 2] + matrix[2, 0]) / (4 * q_1)
    # q_4 = (matrix[1, 2] - matrix[2, 1]) / (4 * q_1)

    return np.array([q_1, q_2, q_3, q_4])


def quaternion2matrix(quaternion):
    q_1, q_2, q_3, q_4 = quaternion

    matrix = np.array([
        [q_1 ** 2 - q_2 ** 2 - q_3 ** 2 + q_4 ** 2, 2 * (q_1 * q_2 + q_3 * q_4), 2 * (q_1 * q_3 - q_2 * q_4)],
        [2 * (q_1 * q_2 - q_3 * q_4), -q_1 ** 2 + q_2 ** 2 - q_3 ** 2 + q_4 ** 2, 2 * (q_2 * q_3 + q_1 * q_4)],
        [2 * (q_1 * q_3 + q_2 * q_4), 2 * (q_2 * q_3 - q_1 * q_4), -q_1 ** 2 - q_2 ** 2 + q_3 ** 2 + q_4 ** 2]
    ])

    return matrix


def get_conjugate_quaternion(quaternion):
    q_1, q_2, q_3, q_4 = quaternion
    return np.array([-q_1, -q_2, -q_3, q_4])


def quaternion_multiply(quaternionA, quaternionB):
    qA_1, qA_2, qA_3, qA_4 = quaternionA
    qB_1, qB_2, qB_3, qB_4 = quaternionB

    matrixA = np.array([
        [qA_4, -qA_3, qA_2, qA_1],
        [qA_3, qA_4, -qA_1, qA_2],
        [-qA_2, qA_1, qA_4, qA_3],
        [-qA_1, -qA_2, -qA_3, qA_4]
    ])

    vectorB = np.array([qB_1, qB_2, qB_3, qB_4])

    return matrixA @ vectorB


def generate_random_quaternion():
    while True:
        q1 = random.uniform(-1, 1)
        q2 = random.uniform(-1, 1)
        q3 = random.uniform(-1, 1)
        q4 = random.uniform(-1, 1)

        norm_ = np.linalg.norm([q1, q2, q3, q4])

        q1 /= norm_
        q2 /= norm_
        q3 /= norm_
        q4 /= norm_

        if q4 >= 0.8:
            return np.array([q1, q2, q3, q4])


def generate_random_omega(max_omega=METADATA['max_omega']):
    omega = np.random.uniform(-max_omega, max_omega, size = 3)

    return omega


if __name__ == '__main__':
    eci = np.array([-4942.032604, 1795.799244, 3616.736937])
    lla = np.array([88.72370363821628, 34.57800720832746, 10.66803617928042])

    datetime = str2datetime('1 Oct 2023 04:08:00.000')

    print(lla2eci(lla[0], lla[1], lla[2], datetime))
    print(eci2lla(eci[0], eci[1], eci[2], datetime))

    # # 29 May 2023 04:50:00.000
    # t_eci = np.array([5816.375388, 433.224864, 2598.658744])
    #
    # s_eci = np.array([6125.227267, -544.622345, 2564.362020])
    # s_orbit = np.array([-0.000000, 0.000000, -6662.655262])
    #
    # true_anomaly, argp, incl, raan = [287.345, 126.422, 28.500, 304.741]
    # st_eci = t_eci - s_eci
    # st_orbit = eci2orbit(true_anomaly, argp, incl, raan, st_eci, is_radians=False)
    #
    # roll_angle = calc_roll_angle(st_orbit)
    # pitch_angle = calc_pitch_angle(st_orbit)
    # yaw_angle = 177.344
    # yaw_angle = 0
    #
    # print(roll_angle, pitch_angle, yaw_angle)
    #
    # s_body = st_orbit
    # # print(s_body, s_body / norm(s_body))
    #
    # s_body_z = body2orbit(np.array([0, 0, 1]), roll_angle, pitch_angle, yaw_angle, ) * norm(s_body)
    # s_body_x = body2orbit(np.array([1, 0, 0]), roll_angle, pitch_angle, yaw_angle, ) * norm(s_body)
    # s_body_y = body2orbit(np.array([0, 1, 0]), roll_angle, pitch_angle, yaw_angle, ) * norm(s_body)
    #
    # print(s_body_x, s_body_y, s_body_z)
    # print(angle_between_vectors(s_body, s_body_z))
    #
    # # print(get_Rob(roll_angle, pitch_angle, yaw_angle))
    #
    # # print(matrix2quaternion(get_Rob(roll_angle, pitch_angle, yaw_angle)))
    # # print(st_orbit, s_body_z)
    # #
    # # print(s_body_z / norm(s_body_z),
    # #       orbit2eci(true_anomaly, argp, incl, raan, s_body_z / norm(s_body_z), is_radians=False))
    #
    # # print()
    # # print(eci2orbit(true_anomaly, argp, incl, raan, st_eci / norm(st_eci), is_radians=False))
