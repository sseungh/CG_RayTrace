from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# 초기화 함수
def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)  # 배경색 설정
    glShadeModel(GL_SMOOTH)            # 부드러운 쉐이딩 모드 사용

    # 광원 설정
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.8, 0.8, 0.8, 1.0))  # Ambient light
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))  # Diffuse light
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0)) # Specular light
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.8)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.001)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    #glEnable(GL_DEPTH_TEST)

# 광원 위치 업데이트
def update_light_position():
    light_position = [-1, 2, -1, 1]  # 광원 위치 (w=0이면 방향 광원)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

def draw_box():
    glPushMatrix()  # 현재 변환 행렬 상태 저장
    glTranslatef(0.15, 0.15, 0.15)
    glutSolidCube(0.3) 
    glPopMatrix()

def draw_sphere():
    glPushMatrix()  # 현재 변환 행렬 상태 저장
    glTranslatef(0.2, 0.0, 0.2)
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialfv(GL_FRONT, GL_SHININESS, 100.0)
    glutSolidSphere(0.3, 32, 32)  # 반지름 0.3, 32개의 경도와 위도 분할로 구 그리기
    glPopMatrix()

# 화면에 그리는 함수
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    update_light_position()  # 광원 위치 업데이트

    #draw_box()               # 박스 그리기
    draw_sphere()
    glutSwapBuffers()

# 메인 함수
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(500, 500)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Python OpenGL Example")
    init()
    glutDisplayFunc(display)
    glutIdleFunc(display)  # 화면이 유휴 상태일 때 display 함수 호출
    glutMainLoop()

# 프로그램 시작
if __name__ == '__main__':
    main()