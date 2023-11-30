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

    # 사각형 평면의 법선 벡터를 계산합니다.
    normal_vector = np.cross(p2 - p1, p3 - p1)
    normal_vector /= np.linalg.norm(normal_vector)

    # 선과 사각형 평면이 평행한지 확인합니다.
    if np.dot(normal_vector, direction) == 0:
        return np.inf, False  # 선과 사각형 평면이 평행하면 교차하지 않습니다.

    # 선과 사각형 평면이 교차하는 점을 계산합니다.
    t = np.dot(normal_vector, p1 - origin) / np.dot(normal_vector, direction)
    intersection_point = origin + t * direction

    # 교차점이 사각형 내부에 있는지 확인합니다.
    return is_inside_rectangle(intersection_point, rectangle_points), intersection_point, normal_vector

def is_inside_rectangle(point, rectangle_points):
    # 사각형의 모든 변의 외적을 계산하여 사각형 내부에 있는지 확인합니다.
    for i in range(4):
        p1, p2 = rectangle_points[i], rectangle_points[(i + 1) % 4]
        edge_vector = p2 - p1
        normal_vector = np.cross(edge_vector, point - p1)

        if np.dot(normal_vector, p1 - point) < 0:
            return False  # 교차점이 사각형 외부에 있으면 False 반환

    return True  # 교차점이 사각형 내부에 있으면 True 반환
