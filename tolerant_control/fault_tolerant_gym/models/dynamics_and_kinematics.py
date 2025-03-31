import numpy as np
from numpy import cos, sin
from metadata import METADATA


def runge_kutta(y, x, dx, f):
    """ y is the initial value for y
        x is the initial value for x
        dx is the time step in x
        f is derivative of function y(t)
    """
    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
    k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
    k4 = dx * f(y + k3, x + dx)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.


# def generate_external_disturbance(t):
#     disturbance_x = 1 + sin(np.pi * t / 125) + cos(np.pi * t / 200)
#     disturbance_y = 2 + sin(np.pi * t / 150) + cos(np.pi * t / 185)
#     disturbance_z = -1 + sin(np.pi * t / 225) + cos(np.pi * t / 175)

#     return 0.001 * np.array([disturbance_x, disturbance_y, disturbance_z])

def generate_external_disturbance(t, omega, constant = False):
    if constant:
        return (10 ** -5) * np.array([])
    omega_x, omega_y, omega_z = omega
    disturbance_x = 3 * cos(10 * omega_x * t) + 4 * sin(3 * omega_x * t) - 10
    disturbance_y = 1.5 * sin(2 * omega_y * t) + cos(5 * omega_y * t) + 15
    disturbance_z = 3 * sin(10 * omega_z * t) + 8 * sin(4 * omega_z * t) + 10

    return (10 ** -5) * np.array([disturbance_x, disturbance_y, disturbance_z])

    # if constant:
    #     return np.array([-0.005, 0.005, -0.005])
    #
    # disturbance_x = -0.005 * sin(t)
    # disturbance_y = 0.005 * sin(t)
    # disturbance_z = -0.005 * sin(t)
    #
    # return np.array([disturbance_x, disturbance_y, disturbance_z])



def integrate_omega(omega, dt, tau, I, t=0):
    def dynamics_func(omega, t):
        omega_dot = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega) )
        return omega_dot

    new_omega = runge_kutta(omega, t, dt, dynamics_func)
    # print(new_omega - omega, np.linalg.inv(I) @ (tau - get_cross_matrix(omega) @ I @ omega) * dt)

    return new_omega


def get_omega_matrix(omega):
    x, y, z = omega
    matrix = np.array([[0, z, -y, x],
                       [-z, 0, x, y],
                       [y, -x, 0, z],
                       [-x, -y, -z, 0]])

    return matrix


def get_quaternion_dot(quaternion, omega):
    omega_matrix = get_omega_matrix(omega)
    quaternion_dot = 0.5 * omega_matrix @ quaternion

    return quaternion_dot


def integrate_quaternion(quaternion, dt, omega, t=0):
    # q4 is the scalar vector

    def runge_kutta(y, x, dx, f):
        """ y is the initial value for y
            x is the initial value for x
            dx is the time step in x
            f is derivative of function y(t)
        """
        k1 = dx * f(y, x)
        k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
        k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
        k4 = dx * f(y + k3, x + dx)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.

    def quaternion_func(quaternion, t):
        omega_matrix = get_omega_matrix(omega)
        quaternion_dot = 0.5 * omega_matrix @ quaternion

        # print(quaternion_dot)

        return quaternion_dot

    new_quaternion = runge_kutta(quaternion, t, dt, quaternion_func)

    # norm
    new_quaternion /= np.linalg.norm(new_quaternion)

    return new_quaternion


if __name__ == '__main__':
    # # tau = np.array([0, 0.025, 0.025])
    # tau = np.array([0.00266535, 0.00399783, 0.00028299])
    # tau = np.array([0.02, 0, 0])
    # # tau = np.array([0.05, 0, 0])
    #
    # dt = 3
    #
    # # Ixx, Iyy, Izz = 0.872, 0.115, 0.797
    #
    # # I = np.array([
    # #     [Ixx, 0, 0],
    # #     [0, Iyy, 0],
    # #     [0, 0, Izz]
    # # ])
    #
    # I = np.array([
    #     [10, 0.02, 0.01],
    #     [0.02, 15, -0.01],
    #     [0.01, -0.01, 20]
    # ])
    #
    # # I = np.array([
    # #     [55.91, 8.92, 12.24],
    # #     [8.92, 53.26, 6.92],
    # #     [12.24, 6.92, 56.29]
    # # ])
    #
    # tau = np.array([0, 0, 1.74402513])
    #
    # # omega = np.array([0, 0, 0])
    # omega = np.array([0, 0, 0])
    #
    #
    # print(omega)
    #
    # quaternion = np.array([-0.3, 0.6, 0, 0.75])
    # new_omega = integrate_omega(omega, t=0, dt=dt, tau=tau, I=I)
    #
    #
    # new_quaternion = integrate_quaternion(quaternion, t=0, dt=dt, omega=new_omega)
    #
    # print(new_omega, new_quaternion)
    pass
