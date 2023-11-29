import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *



def get_camera_direction():
    # 현재의 모델뷰 행렬을 가져옴
    modelview_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)

    # 모델뷰 행렬의 역행렬 계산
    inv_modelview_matrix = np.linalg.inv(modelview_matrix[:3, :3])

    # 카메라가 바라보는 방향 벡터는 역행렬의 세 번째 열
    camera_direction = -inv_modelview_matrix[:, 2]

    return camera_direction
