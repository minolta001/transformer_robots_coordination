import math
import numpy as np
from pyquaternion import Quaternion


PI = np.pi
EPS = np.finfo(float).eps * 4.


def sample_quat(low=0, high=2*np.pi):
    """Samples quaternions of random rotations along the z-axis."""
    rot_angle = np.random.uniform(high=high, low=low)
    return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]


# code from stanford robosuite
def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q: a 4-dim numpy array corresponding to a quaternion
        to: a string, either 'xyzw' or 'wxyz', determining
            which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def norm(x):
    return x / np.linalg.norm(x)


def lookat_to_quat(forward, up):
    vector = norm(forward)
    vector2 = norm(np.cross(norm(up), vector))
    vector3 = np.cross(vector, vector2)
    m00 = vector2[0]
    m01 = vector2[1]
    m02 = vector2[2]
    m10 = vector3[0]
    m11 = vector3[1]
    m12 = vector3[2]
    m20 = vector[0]
    m21 = vector[1]
    m22 = vector[2]

    num8 = (m00 + m11) + m22
    quaternion = np.zeros(4)
    if num8 > 0:
        num = np.sqrt(num8 + 1)
        quaternion[3] = num * 0.5
        num = 0.5 / num
        quaternion[0] = (m12 - m21) * num
        quaternion[1] = (m20 - m02) * num
        quaternion[2] = (m01 - m10) * num
        return quaternion

    if ((m00 >= m11) and (m00 >= m22)):
        num7 = np.sqrt(((1 + m00) - m11) - m22)
        num4 = 0.5 / num7
        quaternion[0] = 0.5 * num7
        quaternion[1] = (m01 + m10) * num4
        quaternion[2] = (m02 + m20) * num4
        quaternion[3] = (m12 - m21) * num4
        return quaternion

    if m11 > m22:
        num6 = np.sqrt(((1 + m11) - m00) - m22)
        num3 = 0.5 / num6
        quaternion[0] = (m10+ m01) * num3
        quaternion[1] = 0.5 * num6
        quaternion[2] = (m21 + m12) * num3
        quaternion[3] = (m20 - m02) * num3
        return quaternion

    num5 = np.sqrt(((1 + m22) - m00) - m11)
    num2 = 0.5 / num5
    quaternion[0] = (m20 + m02) * num2
    quaternion[1] = (m21 + m12) * num2
    quaternion[2] = 0.5 * num5
    quaternion[3] = (m01 - m10) * num2
    return quaternion


# https://www.gamedev.net/forums/topic/56471-extracting-direction-vectors-from-quaternion/
def forward_vector_from_quat(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 2 * (qx * qy + qw * qz)#
    y = 1 - 2 * (qx * qx + qz * qz)
    z = 2 * (qy * qz - qw * qx)#
    return np.array([x, y, z])


def up_vector_from_quat(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 2 * (qx * qz - qw * qy)
    y = 2 * (qy * qz + qw * qx)
    z = 1 - 2 * (qx * qx + qy * qy)
    return np.array([x, y, z])


def right_vector_from_quat(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    x = 1 - 2 * (qy * qy + qz * qz)
    y = 2 * (qx * qy - qw * qz)
    z = 2 * (qx * qz + qw * qy)#
    return np.array([x, y, z])


def quat_dist(quat1, quat2):
    q1 = Quaternion(axis=quat1[:-1], angle=quat1[-1])
    q2 = Quaternion(axis=quat2[:-1], angle=quat2[-1])
    return Quaternion.sym_distance(q1, q2)


def l2_dist(a, b):
    return np.linalg.norm(a - b)


def cos_dist(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

# given two forward vector at two time steps
# return > 0 if rotate right
# return < 0 if rotate left
# return 0 if move straight
def rotate_direction(a, b):
    rotate = np.cross(a, b)[2]
    if(rotate > 0):
        return 1
    elif(rotate < 0):
        return -1
    else:   # moving straight
        return 0

# Given pos of the object, pos of the vehicle, and the forward vector of the vehicle, 
# check if the heading of the vehicle is pointing toward the object
# return 1 - normalized radian
# The closer to 1, the more correct the heading
def movement_heading_difference(ob_pos, car_pos, car_forward_vec):
    direction_vec = ob_pos - car_pos
    direction_vec_normalized = direction_vec / np.linalg.norm(direction_vec)
    dot_product = np.dot(car_forward_vec, direction_vec_normalized)
    
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return abs(1 - (angle_rad / math.pi))

# Given forward vector of the object and the vehicle, check if their headings are parallel to each one
# 
def alignment_heading_difference(ob_forward_vec, car_forward_vec):
    ob_forward_normalized = ob_forward_vec / np.linalg.norm(ob_forward_vec)
    car_forward_nomalized = car_forward_vec / np.linalg.norm(car_forward_vec)

    dot_product = np.dot(ob_forward_normalized, car_forward_nomalized)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

    if(angle_rad < math.pi):
        angle_diff = min(abs(angle_rad - math.pi), angle_rad)
    else:
        angle_diff = min(abs(angle_rad - math.pi), abs(2 * math.pi - angle_rad))

    if(angle_diff > math.pi / 2 or angle_diff < 0):
        print("\nangle_diff error!!")
        return ValueError
    return abs(1 - (angle_diff / math.pi / 2))


# Given position before, position after and forward vector before,
# return if the object move forward or backward
# return 1 if forward, -1 if backward, 0 if perpendicular
def forward_backward(pos_before, pos_after, forward_before):
    displacement = pos_after - pos_before
    displacement_normalized = displacement / np.linalg.norm(displacement)
    
    forward_normalized = forward_before / np.linalg.norm(forward_before)
    dot_product = np.dot(displacement_normalized, forward_normalized)

    if dot_product > 0:
        return 1    # move forward
    elif dot_product < 0:
        return -1   # move backward
    else:
        return 0    # move perpendicular

