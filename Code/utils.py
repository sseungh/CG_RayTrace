import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *



def get_camera_basis(look_from, look_at, cam_up, fin_rot):
    R = np.eye(4)
    R[:3, :3] = fin_rot
    camera_orig = np.array([look_from[0],look_from[1],look_from[2],1])
    camera_orig = np.dot(R, camera_orig)[:3]

    dest = np.array([look_at[0], look_at[1], look_at[2], 1])
    dest = np.dot(R, dest)[:3]

    up = np.array([cam_up[0],cam_up[1],cam_up[2],1])
    up = np.dot(R, up)[:3]

    camera_look = dest - camera_orig
    camera_x = np.cross(camera_look, up)
    camera_y = np.cross(camera_x, camera_look)

    camera_look = camera_look / np.linalg.norm(camera_look)
    camera_x = camera_x / np.linalg.norm(camera_x)
    camera_y = camera_y / np.linalg.norm(camera_y)

    return camera_orig, camera_look, camera_x, camera_y


import numpy as np

def square_intersecting(rectangle_points, origin, direction):
    # 사각형 평면을 정의하는데 사용되는 세 개의 점을 선택합니다.
    p1, p2, p3 = rectangle_points[:3]
    v1, v2 = p2 - p1, p3 - p1

    # 사각형 평면의 법선 벡터를 계산합니다.
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # 선과 사각형 평면이 평행한지 확인합니다.
    if np.dot(normal, direction) == 0:
        return np.inf, False  # 선과 사각형 평면이 평행하면 교차하지 않습니다.

    # 선과 사각형 평면이 교차하는 점을 계산합니다.
    A = np.stack((v1, v2, direction))
    a, b, k = np.linalg.solve(A, o)
    intersection_point = a * v1 + b * v2
    k = -k

    if a < 0 and b < 0:
        return False, origin, normal
    else:
        return True, intersection_point, normal
