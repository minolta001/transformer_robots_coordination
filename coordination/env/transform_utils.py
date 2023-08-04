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
    return np.array([x, -y, z])


def quat_dist(quat1, quat2):
    q1 = Quaternion(axis=quat1[:-1], angle=quat1[-1])
    q2 = Quaternion(axis=quat2[:-1], angle=quat2[-1])
    return Quaternion.sym_distance(q1, q2)


def l2_dist(a, b):
    return np.linalg.norm(a - b)


def cos_dist(a, b):
    a_normalized = a / np.linalg.norm(a)
    b_normalized = b / np.linalg.norm(b)

    dot_product = np.dot(a_normalized, b_normalized)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return angle_rad / np.pi

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
# Also, could also check if the tail of robot is pointing toward the object
# return 1 - normalized radian
# The closer to 1, the more correct the heading
def movement_heading_difference(ob_pos, car_pos, car_forward_vec, f_or_b):
    assert(len(ob_pos) == 3 and len(car_forward_vec) == len(ob_pos))
    ob_pos = ob_pos[0:2]
    car_pos = car_pos[0:2]
    assert(len(ob_pos) == len(car_pos))
    car_forward_vec = car_forward_vec[0:2]

    direction_vec = ob_pos - car_pos
    suggested_vec_normalized = direction_vec / np.linalg.norm(direction_vec)

    if f_or_b == "forward":
        car_vec_normalized = car_forward_vec / np.linalg.norm(car_forward_vec)
    elif f_or_b == "backward":
        car_forward_vec = -car_forward_vec
        car_vec_normalized = car_forward_vec / np.linalg.norm(car_forward_vec)

    dot_product = np.dot(suggested_vec_normalized, car_vec_normalized)
    
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return (1 - (angle_rad / np.pi)) - 0.5

# Given forward vector of the object and the vehicle, check if their headings are parallel to each one
# NOTE: No matter forward or backward vector is parallel with the forward vector of the box, it is OK.
def alignment_heading_difference(ob_forward_vec, car_forward_vec):
    ob_forward_vec = ob_forward_vec[0:2]
    car_forward_vec = car_forward_vec[0:2]

    ob_forward_normalized = ob_forward_vec / np.linalg.norm(ob_forward_vec)
    car_forward_nomalized = car_forward_vec / np.linalg.norm(car_forward_vec)

    dot_product = np.dot(ob_forward_normalized, car_forward_nomalized)
    angle_rad_forward = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    car_backward_nomalized = (-car_forward_vec) / np.linalg.norm(-car_forward_vec)
    dot_product = np.dot(ob_forward_normalized, car_backward_nomalized)
    angle_rad_backward = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # the minimum of max(1 - angle_rad_*) is 0.5 (perpendicular), the maximum is 1.0 (parallel).
    # therefore, we reduce the value and re-scale it to (0, 1) range.
    return (max(1 - angle_rad_forward/np.pi, 
               1 - angle_rad_backward/np.pi) - 0.5) / 0.5

    '''
    if(angle_rad < np.pi):
        angle_diff = min(abs(angle_rad - np.pi), angle_rad)
    else:
        angle_diff = min(abs(angle_rad - np.pi), abs(2 * np.pi - angle_rad))

    if(angle_diff > np.pi / 2 or angle_diff < 0):
        print("\nangle_diff error!!")
        return ValueError
    '''



# Give qvel of an object, check if the object is moving forward or backward
def forward_backward(qvel):
    qvel = qvel[6:10]
    res = sum(qvel) / 4
    if(res > 0):
        return 1
    elif(res < 0):
        return -1
    else:
        return 0
