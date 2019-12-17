#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import quaternion  # noqa # pylint: disable=unused-import


def rotation_to_quaternion(r: np.array):
    r"""
    ref: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    tr = r[0][0] + r[1][1] + r[2][2]

    if (tr > 0):
        S = np.sqrt(tr+1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (r[2][1] - r[1][2]) / S
        qy = (r[0][2] - r[2][0]) / S
        qz = (r[1][0] - r[0][1]) / S
    elif ((r[0][0] > r[1][1]) & (r[0][0] > r[2][2])):
        S = np.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2  # S=4*qx
        qw = (r[2][1] - r[1][2]) / S
        qx = 0.25 * S
        qy = (r[0][1] + r[1][0]) / S
        qz = (r[0][2] + r[2][0]) / S
    elif (r[1][1] > r[2][2]):
        S = np.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2  # S=4*qy
        qw = (r[0][2] - r[2][0]) / S
        qx = (r[0][1] + r[1][0]) / S
        qy = 0.25 * S
        qz = (r[1][2] + r[2][1]) / S
    else:
        S = np.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2  # S=4*qz
        qw = (r[1][0] - r[0][1]) / S
        qx = (r[0][2] + r[2][0]) / S
        qy = (r[1][2] + r[2][1]) / S
        qz = 0.25 * S

    return np.quaternion(qw, qx, qy, qz)


def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j ** 2 + q_k ** 2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i ** 2 + q_k ** 2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i ** 2 + q_j ** 2),
            ],
        ],
        dtype=np.float32,
    )
    return rotation_mat


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion

    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate

    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format
    """
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def polar_to_cartesian(phi, rho=1):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

'''
def heading_to_rotation(heading):
    x, y = polar_to_cartesian(heading)
    heading_vector = [y, 0, -x]
    rotation_matrix = np.array([[-x, 0, -y], [0, -1, 0], [-y, 0, x]])
    quat = rotation_to_quaternion(rotation_matrix)
    return [quat.real] + quat.imag.tolist()
'''
def heading_to_rotation(heading):
    if heading < 0:
        heading = np.pi * 2 + heading
    cy = 1
    sy = 0
    cp = np.cos(heading * 0.5)
    sp = np.sin(heading * 0.5)
    cr = 1
    sr = 0
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    coeffs = [x,y,z,w]
    return  coeffs
