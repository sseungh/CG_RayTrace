import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image # conda install pillow
from math import sin

from utils import *



OBJ_TYPE = {
    'BASIC': 0,
    'REFLECTOR': 1,
    'REFRACTOR': 2,
}


def ray_trace(o, d):
    intersects = []
    # collision이 일어날 것으로 예측되는 obj들 중 가장 가까운 것을 선택하기 때문에 intersect 단에서 ray의 진행방향과 반대에 있는 obj는 걸러주어야 함
    for i, obj in enumerate(SubWindow.obj_list):
        is_intersect, intersect_point, changed_d = obj.intersect(o, d)
        if is_intersect:
            distance = np.linalg.norm(intersect_point - o)
            intersects.append([i, intersect_point, changed_d, distance])

    if len(intersects) == 0:    # intersecting obj가 존재하지 않으면 black으로 mapping
        return (0, 0, 0)
    intersects.sort(key=lambda l: l[-1])
    idx, intersect_point, changed_d, _ = intersects[0]

    collision = SubWindow.obj_list[idx]
    if collision.obj_type == OBJ_TYPE['BASIC']:
        ret = collision.get_pixel(o, d)
        print("ret:", ret)
        return ret
    return ray_trace(intersect_point, changed_d)


class Object:
    cnt = 0
    recurse = 0

    def __init__(self, obj_type=OBJ_TYPE["BASIC"]):
        # Do NOT modify: Object's ID is automatically increased
        self.id = Object.cnt
        Object.cnt += 1
        # self.mat needs to be updated by every transformation
        self.mat = np.eye(4)
        self.obj_type = obj_type

    def draw(self):
        raise NotImplementedError
    
    def intersect(self, o, d):
        raise NotImplementedError
    
    def get_pixel(self, o, d):
        raise NotImplementedError



class Sphere(Object):
    def __init__(self, obj_type=OBJ_TYPE["BASIC"]):
        super().__init__(obj_type)

    def draw(self):
        glPushMatrix()
        glMultMatrixf(self.mat.T)
        glColor3f(0.0, 1.0, 0.0)
        glutSolidSphere(0.1, 32, 32)
        glColor3f(1.0, 1.0, 1.0)
        glPopMatrix()

    # Responsible to check if a ray intersects with the sphere 
    # o origin of ray, d direction
    def intersect(self, o, d):
        n_sphere = 2.419 # refrective index for diomand as an example
        n_air = 1.00
        center = self.mat[:3,3]
        r = 0.1
        dot = np.dot(d, center - o)
        # Check if the ray is pointing away from the sphere
        # If smaller than zero, no intersaction
        if dot < 0:

            # Return False along with the original ray parameters
            return False, o, d
        
        # Calculate the point on the ray closest to the sphere's center
        v = d / np.linalg.norm(d) * dot
        
        # Check if the closest point is inside the sphere
        if np.linalg.norm(o + v - center) <= r:
            b = np.dot(v, o - center) / np.dot(v, v)
            sqrt = np.sqrt(b*b - np.dot(o-center, o-center) + r*r) # discriminant
            
            # Two potential solutions for the intersection parameter 'k'
            k1, k2 = - b - sqrt, - b + sqrt
            
            # Choose the intersection point based on the sign of k1 and k2
            if k1 * k2 < 0:
                k = max(k1, k2)
            else:
                k = min(k1, k2)
            
            # Calculate the intersection point
            intersect_points = o + k * v
            normal = intersect_points - center
            # Ensure the normal points outward from the sphere
            if np.dot(normal, d) > 0:
                normal = -normal
        
            # cos and sin of the angle between the incident ray direction and the surface normal 
            # considering that it is a 3D surface [2D->> cos_theta = -np.dot(d, normal)]
            cos_theta = np.dot(d, normal) / (np.linalg.norm(d) * np.linalg.norm(normal))
            sin_theta = np.sqrt(1.0 - cos_theta**2)


            if n_sphere * sin_theta > 1.0:
                # Total internal reflection, the material does not allow light to pass through
                return None  
            else:
                # Using Snell's Law to calculate the refracted ray direction
                sin_theta2 = (n_air / n_sphere) * sin_theta
                cos_theta2 = np.sqrt(1.0 - sin_theta2**2)
                refracted_d = (n_air / n_sphere) * d + (n_air / n_sphere * cos_theta - cos_theta2) * normal
                refracted_d = refracted_d / np.linalg.norm(refracted_d)

            return True, intersect_points, refracted_d
        
        return False, o, d
    
    def get_pixel(self, o, d):
        return (0, 255, 0)

class Env(Object):
    _v = [[0., 0., 0.],
        [2., 0., 0.],
        [2., 2., 0.],
        [0., 2., 0.],
        [0., 0., 2],
        [2., 0., 2.],
        [2., 2., 2.],
        [0., 2., 2.]]
    _f = [[0, 1, 2, 3],
        [0, 1, 5, 4],
        [0, 3, 7, 4]]

    def __init__(self):
        super().__init__()
        self.loadTexture()
    
    def loadTexture(self):
        self.check, self.black = glGenTextures(2)
        if self.check == 0:
            raise Exception("텍스처 생성 실패")
        glBindTexture(GL_TEXTURE_2D, self.check)
        image = Image.open("checkered_texture.jpg")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.size[0], image.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glDisable(GL_TEXTURE_2D)

    
    def draw(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_TEXTURE_2D)  # 텍스처 활성화
        glBindTexture(GL_TEXTURE_2D, self.check)
        glPushMatrix()
        glMultMatrixf(self.mat.T)

        n = 0
        for face in Env._f:
            glBegin(GL_QUADS)
            for i, v in enumerate(face):
                if n==1:
                    glColor3f(0.3, 0.3, 0.3)
                    x, y, z = Env._v[v]
                    glVertex3f(x, y, z)
                    glColor3f(1.0, 1.0, 1.0)
                else:
                    if i == 0:
                        glTexCoord2f(0, 0)  # 첫 번째 정점의 텍스처 좌표
                    elif i == 1:
                        glTexCoord2f(0, 1)  # 두 번째 정점의 텍스처 좌표
                    elif i == 2:
                        glTexCoord2f(1, 1)  # 세 번째 정점의 텍스처 좌표
                    elif i == 3:
                        glTexCoord2f(1, 0)  # 네 번째 정점의 텍스처 좌표
                    x, y, z = Env._v[v]
                    glVertex3f(x, y, z)
            n += 1
            glEnd()

        glPopMatrix()
        glDisable(GL_TEXTURE_2D)

    # Responsible to check if a ray intersects with the environment 
    def intersect(self, o, d):
        intersects = []
        for f in Env._f:
            is_intersect, intersect_point, normal = square_intersecting(Env._v[f], o, d)
            dot = np.dot(intersect_point - o, d)
            if is_intersect and dot > 0:
                distance = np.linalg.norm(intersect_point - o)
                if np.dot(d, normal) > 0:
                    normal = -normal
                intersects.append([intersect_point, normal, distance])
        if len(intersects) == 0:
            return False, o, d
        intersects.sort(key=lambda l: l[-1])
        intersect_point, normal, _ = intersects[0]
        changed_d = d - 2 * np.dot(d, normal) * normal
        changed_d = changed_d / np.linalg.norm(changed_d)
        return True, intersect_point, changed_d
    
    def get_pixel(self, o, d):
        return (255, 0, 0)

class SubWindow:
    """
    SubWindow Class.\n
    Used to display objects in the obj_list, with different camera configuration.
    """

    windows = []
    obj_list = []
    light = None

    def __init__(self, win, x, y, width, height):
        # identifier for the subwindow
        self.id = glutCreateSubWindow(win, x, y, width, height)
        # projection matrix
        self.projectionMat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # view matrix, you do not need to modify the matrix for now
        self.viewMat = np.eye(4)
        # shader program used to pick objects, and its associated value. DO NOT MODIFY.=
        self.width = width
        self.height = height
        #SubWindow.light = Light()
        sphere = Sphere()
        sphere.mat[:3,3] = [0.9, 0.3, 0.9]
        # print(sphere.mat)
        SubWindow.obj_list.append(sphere)
        env = Env()
        SubWindow.obj_list.append(env)

        self.fov = 3
        self.init_from = np.array([3, 1, 3])
        temp = 1/(np.tan(self.fov*np.pi/360)*np.sqrt(3))
        self.init_from = self.init_from/np.linalg.norm(self.init_from)*temp
        self.look_from = self.init_from
        self.look_at = np.array([0.,0.3,0.])
        self.cam_up = np.array([0.,1.,0.])
        
        self.cur_mat = np.eye(3)
        self.fin_rot = np.eye(3)
        self.pos_init = []
        self.track_ball = False
        self.radius = 1.
        self.ratio = width/height
        

    def display(self):
        """
        Display callback function for the subwindow.
        """
        glutSetWindow(self.id)

        self.drawScene()

        glutSwapBuffers()
    
    def press_d(self):
        self.look_from = self.init_from
        self.look_at = np.array([0.,0.3,0.])
        self.cam_up = np.array([0.,1.,0.])
        self.cur_mat = np.eye(3)
        self.fin_rot = np.eye(3)
    
    def render(self):
        canvas = np.zeros((500, 500, 3))
        width, height = self.width, self.height
        '''data = glReadPixels(500-252, 500-332, 1, 1, GL_RGB, GL_FLOAT)
        print(data)'''
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT)
        ind_set = []
        for x in range(width):
            for y in range(height):
                color = data[x][y]
                r, g, b = color[0], color[1], color[2]
                if r == 0.0 and g > 0.1 and b == 0.0:
                    #print(f"녹색 픽셀 위치: ({500-y}, {500-x})")
                    ind_set.append((height-y, width-x))
        '''for i, (x, y) in enumerate(ind_set):
            print(f"녹색 픽셀 위치 {i}: ({x}, {y})")'''
        print("총 녹색 픽셀 수:", len(ind_set))


        # depth_info = list(map(lambda arg: glReadPixels(500-arg[0], 500-arg[1], 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT), ind_set))   # depth 값 read가 안됨
        camera_orig, camera_look, camera_x, camera_y = get_camera_basis(self.look_from, self.look_at, self.cam_up, self.fin_rot)
        for i, (mouse_x, mouse_y) in enumerate(ind_set):
            _x = mouse_x - self.width // 2
            _y = self.height // 2 - mouse_y
            world_x, world_y, world_z = camera_orig
            d = camera_look + np.tan(self.fov * np.pi / 360) * (_x / 250) * camera_x + np.tan(self.fov * np.pi / 360) * (_y / 250) * camera_y
            canvas[_x, _y] = ray_trace(world_x, world_y, world_z, d)
            break


        # for i, (x, y) in enumerate(ind_set):
            # RGB = glReadPixels(500-x, 500-y, 1, 1, GL_RGB, GL_FLOAT) #인풋 좌표는 또 바뀜
            # assert(RGB[0][0][0]==0 and RGB[0][0][1]>0.1 and RGB[0][0][2]==0)

    def drawScene(self):
        """
        Draws scene with objects to the subwindow.
        """
        glutSetWindow(self.id)

        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(self.projectionMat.T)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(self.viewMat.T)

        if self.id == 2:
            gluPerspective(self.fov, 1., 1e-10, 100.0)
            #glOrtho(-1, 1, -1, 1, 0.0001, 100.0)
            R = np.eye(4)
            R[:3, :3] = self.fin_rot
            a = np.array([self.look_from[0],self.look_from[1],self.look_from[2],1])
            b = np.array([self.look_at[0], self.look_at[1], self.look_at[2], 1])
            c = np.array([self.cam_up[0],self.cam_up[1],self.cam_up[2],1])
            a = np.dot(R, a)
            b = np.dot(R, b)
            c = np.dot(R, c)
            gluLookAt(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])

        #self.drawAxes()
        #SubWindow.light.update_light_position()

        for obj in SubWindow.obj_list:
            obj.draw()

    def mouse(self, button, state, x, y):
        """
        Mouse callback function.
        """
        # button macros: GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON
        print(f"Display #{self.id} mouse press event: button={button}, state={state}, x={x}, y={y}")
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            self.track_ball = True
            self.pos_init = [x, y]
            self.cur_mat = self.fin_rot

            obj_id = self.pickObject(x, y)
            if obj_id != 0xFFFFFF:
                print(f"{obj_id} selected")
            else:
                print("Nothing selected")
        elif button == GLUT_LEFT_BUTTON and state == GLUT_UP:
            self.track_ball = False

        if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
            print(f"Add teapot at ({x}, {y})")

        self.button = button
        self.modifier = glutGetModifiers()

        glutPostRedisplay()
    
    def win_to_cam(self, start_x, start_y, dest_x, dest_y):
        # 로드리게스 회전 매트릭스 공식을 사용하여, 초기 위치에서 현재 위치를 기준으로 회전축과 각도를 계산
        cur_loc = self.look_at
        if self.ratio > 1:
            start_x = (2.0*start_x/self.width-1.0)*self.ratio+cur_loc[0]
            start_y = (1.0-2.0*start_y/self.height)+cur_loc[1]
        else:
            start_x = (2.0*start_x/self.width-1.0)+cur_loc[0]
            start_y = (1.0-2.0*start_y/self.height)/self.ratio+cur_loc[1]
        start_l = np.sqrt(start_x**2+start_y**2)
        if start_l < self.radius:
            start_z = np.sqrt(self.radius**2 - start_l**2)
        else:
            start_z = 0.
            start_x = self.radius*start_x/start_l
            start_y = self.radius*start_y/start_l
        start_v = np.array([start_x, start_y, start_z])

        if self.ratio > 1:
            dest_x = (2.0*dest_x/self.width-1.0)*self.ratio+cur_loc[0]
            dest_y = (1.0-2.0*dest_y/self.height)+cur_loc[1]
        else:
            dest_x = (2.0*dest_x/self.width-1.0)+cur_loc[0]
            dest_y = (1.0-2.0*dest_y/self.height)/self.ratio+cur_loc[1]
        dest_l = np.sqrt(dest_x**2+dest_y**2)
        if dest_l < self.radius:
            dest_z = np.sqrt(self.radius**2 - dest_l**2)
        else:
            dest_z = 0.
            dest_x = self.radius*dest_x/dest_l
            dest_y = self.radius*dest_y/dest_l
        dest_v = np.array([dest_x, dest_y, dest_z])

        temp = np.cross(start_v, dest_v)
        if np.any(temp > 0.002) | np.any(temp < -0.002):
            temp = np.dot(self.cur_mat, temp)
            axis = temp/np.linalg.norm(temp)
            angle = np.arccos(np.dot(start_v, dest_v))
        else:
            return np.eye(3)
        
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R

    def motion(self, x, y):
        """
        Motion (Dragging) callback function.
        """
        print(f"Display #{self.id} mouse move event: x={x}, y={y}, modifer={self.modifier}")
        if self.track_ball:
            #트랙볼중. 현재위치를 받아서 cur_mat에 저장된 시작시의 매트릭스에 계속 업데이트해줌
            rot = self.win_to_cam(self.pos_init[0], self.pos_init[1], x, y)
            self.fin_rot = np.dot(rot.T, self.cur_mat)
        
        if self.button == GLUT_LEFT_BUTTON:
            if self.modifier & GLUT_ACTIVE_ALT:
                print("Rotation")
            elif self.modifier & GLUT_ACTIVE_SHIFT:
                print("Scaling")
            else:
                print("Translation")

        glutPostRedisplay()

    def pickObject(self, x, y):
        """
        Object picking function.\n
        obj_id can be used to identify which object is clicked, as each object is assigned with unique id.
        """

        data = glReadPixels(x, self.height - y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)

        obj_id = data[0] + data[1] * (2**8) + data[2] * (2**16)

        self.drawScene()

        return obj_id


class Viewer:
    width, height = 500, 500

    def __init__(self):
        pass

    def light(self):
        """
        Light used in the scene.
        """
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        # feel free to adjust light colors
        lightAmbient = [0.5, 0.5, 0.5, 1.0]
        lightDiffuse = [1.0, 1.0, 1.0, 1.0]
        lightSpecular = [0.5, 0.5, 0.5, 1.0]
        '''glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.8)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.001)'''
        lightPosition = [1, 1, -1, 1]  # vector: point at infinity
        glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular)
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition)
        glEnable(GL_LIGHT0)

    def idle(self):
        """
        Idle callback function.\n
        Used to update all the subwindows.
        """
        self.display()
        for subWindow in SubWindow.windows:
            subWindow.display()

    def display(self):
        """
        Display callback function for the main window.
        """
        glutSetWindow(self.mainWindow)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1, 1, 1, 1)

        glutSwapBuffers()

    def reshape(self, w, h):
        """
        Reshape callback function.\n
        Does notihing as of now.
        """
        print(f"reshape to width: {w}, height: {h}")

        glutPostRedisplay()

    def keyboard(self, key, x, y):
        """
        Keyboard callback function.
        """
        print(f"Display #{glutGetWindow()} keyboard event: key={key}, x={x}, y={y}")
        if glutGetModifiers() & GLUT_ACTIVE_SHIFT:
            print("shift pressed")
        if glutGetModifiers() & GLUT_ACTIVE_ALT:
            print("alt pressed")
        if glutGetModifiers() & GLUT_ACTIVE_CTRL:
            print("ctrl pressed")
        
        if key==b'd':
            #d를 눌렀을 때.
            self.subWindow.press_d()
        
        if key==b'r':
            #r를 눌렀을 때.
            self.subWindow.render()

        glutPostRedisplay()

    def special(self, key, x, y):
        """
        Special key callback function.
        """
        print(f"Display #{glutGetWindow()} special key event: key={key}, x={x}, y={y}")

        glutPostRedisplay()

    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.mainWindow = glutCreateWindow(b"CS471 Computer Graphics #1")

        # 메인 윈도우의 디스플레이 콜백 설정
        glutDisplayFunc(self.display)
        
        # 단일 SubWindow 생성 및 설정
        self.subWindow = SubWindow(self.mainWindow, 0, 0, self.width, self.height)
        glutSetWindow(self.subWindow.id)
        glutDisplayFunc(self.subWindow.display)  # 서브 윈도우의 디스플레이 콜백 설정
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.special)
        glutMouseFunc(self.subWindow.mouse)
        glutMotionFunc(self.subWindow.motion)

        self.light()

        glutMainLoop()


if __name__ == "__main__":
    viewer = Viewer()
    viewer.run()
