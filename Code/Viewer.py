from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from numpy.linalg import inv
from PIL import Image # conda install pillow
#import threading



def load_bunny():
    Vn = []
    V = []
    F = []
    with open("./bunny_1024.obj", 'r') as f:
        for line in f.readlines():
            t, a, b, c = line.strip().split(' ')
            if t == 'vn':
                Vn += [[float(a), float(b), float(c)]]
            elif t == 'v':
                V += [[float(a) * 0.2, float(b) * 0.2, float(c) * 0.2]]
            elif t == 'f':
                F += [[int(a.split('/')[0])-1, int(b.split('/')[0])-1, int(c.split('/')[0])-1]]

    V = np.array(V, dtype='float32')
    V = V - V.mean(axis=0)
    Vn = np.array(Vn, dtype='float32')
    F = np.array(F, dtype='uint')

    return Vn, V, F


def build_bvh(parent, nodes, total_nodes):
    if len(nodes) == 1:
        node = nodes[0]
        node.parent = parent
        return node

    new_node = Node()
    total_nodes.append(new_node)

    axis = Node.axis % 3
    Node.axis += 1

    nodes.sort(key=lambda l: l.center[axis])
    l_nodes = nodes[:len(nodes) // 2]
    r_nodes = nodes[len(nodes) // 2:]

    lnode = build_bvh(new_node, l_nodes, total_nodes)
    rnode = build_bvh(new_node, r_nodes, total_nodes)

    new_node.parent = parent
    new_node.left = lnode
    new_node.right = rnode
    new_node.combine_aabb()

    return new_node


class Node:
    axis = 0
    def __init__(self):
        self.parent = None
        self.left = None
        self.right = None

        self.Vs = None
        self.aabb_min = None
        self.aabb_max = None
        self.center = None

    def init(self, Vs):
        self.Vs = Vs
        self.aabb_min = Vs.min(axis=0)
        self.aabb_max = Vs.max(axis=0)
        self.center = Vs.mean(axis=0)

    def combine_aabb(self):
        self.aabb_min = np.min([self.left.aabb_min, self.right.aabb_min], axis=0)
        self.aabb_max = np.max([self.left.aabb_max, self.right.aabb_max], axis=0)


class Object:
    cnt = 0

    def __init__(self):
        # Do NOT modify: Object's ID is automatically increased
        self.id = Object.cnt
        Object.cnt += 1
        # self.mat needs to be updated by every transformation
        self.mat = np.eye(4)

    def draw(self):
        raise NotImplementedError

class Sphere(Object):
    def __init__(self):
        super().__init__()
        self.r = 0.3

    def draw(self):
        glPushMatrix()
        glMultMatrixf(self.mat.T)
        glColor3f(0.0, 1.0, 0.0)
        glutSolidSphere(self.r, 32, 32)
        glColor3f(1.0, 1.0, 1.0)
        glPopMatrix()


class Bunny(Object):
    Vn, V, F = load_bunny()
    _Vn = np.concatenate(Vn, axis=0)
    _V = np.concatenate(V, axis=0)
    _F = np.concatenate(F, axis=0)

    def __init__(self):
        super().__init__()
        self.nodes = []
        for fa, fb, fc in Bunny.F:
            Vs = Bunny.V[[fa, fb, fc]]
            node = Node()
            node.init(Vs)
            self.nodes += [node]
        
        self.bvh = build_bvh(None, self.nodes.copy(), self.nodes)
        self.aabb_min = self.bvh.aabb_min
        self.aabb_max = self.bvh.aabb_max

    def draw(self):
        glColor3f(0., 1., 0.)
        glPushMatrix()
        glMultMatrixf(self.mat.T)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, Bunny._V)
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, Bunny._Vn)
        glDrawElements(GL_TRIANGLES, len(Bunny._F), GL_UNSIGNED_INT, Bunny._F)
        glPopMatrix()
        glColor3f(1., 1., 1.)


class Line(Object):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
    
    def draw(self):
        glPushMatrix()
        glMultMatrixf(self.mat.T)

        glLineWidth(2.5)  # 선의 두께 설정
        glColor3f(1.0, 0.0, 0.0)  # 선의 색상 설정 (빨간색)
        glBegin(GL_LINES)
        glVertex3fv(self.start)  # 시작점
        glVertex3fv(self.end)   # 끝점
        glEnd()

        glPopMatrix()

class Bullet(Object):
    radius = 0.01
    def __init__(self, start, dir):
        super().__init__()
        self.start = start
        self.dir = dir
        self.inside = False
        self.coord = None
        self.order = None
    
    def draw(self):
        glPushMatrix()
        glMultMatrixf(self.mat.T)
        glutSolidSphere(Bullet.radius, 16, 16)
        glPopMatrix()

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

class SubWindow:
    """
    SubWindow Class.\n
    Used to display objects in the obj_list, with different camera configuration.
    """

    windows = []
    obj_list = []
    bullet_list = []
    pixel_list = []
    shade_list = []
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
        # sphere = Sphere()
        # sphere.mat[:3,3] = [0.9, 0.3, 0.9]
        # SubWindow.obj_list.append(sphere)
        bunny = Bunny()
        bunny.mat[:3, 3] = [0.9, 0.3, 0.9]
        SubWindow.obj_list.append(bunny)
        env = Env()
        SubWindow.obj_list.append(env)

        self.fov = 3
        self.init_from = np.array([-15., -6., -15.])
        self.look_from = self.init_from
        self.look_at = np.array([0.,0.,0.])
        look_direction = self.look_at - (self.look_from*-1)
        look_direction /= np.linalg.norm(look_direction)
        cam_right = np.cross(look_direction, np.array([0.,1.,0.]))
        cam_right /= np.linalg.norm(cam_right)
        self.cam_up = np.cross(cam_right, look_direction)
        self.up_init = self.cam_up
        self.cur_mat = np.eye(3)
        self.fin_rot = np.eye(3)
        self.pos_init = []
        self.track_ball = False
        self.radius = 1.
        self.ratio = width/height

        # self.sphere_r = sphere.r
        # self.sphere_loc = sphere.mat[:3,3]
        self.bunny_loc = bunny.mat[:3,3]
        self.RenderPixel = False
    
    def draw_center_dot(self):
        for ind, color, order in SubWindow.pixel_list:
            glWindowPos2i(ind[0], ind[1])
            temp_color = int(SubWindow.shade_list[order]*color[0])
            color = [temp_color, temp_color, temp_color]
            glDrawPixels(1, 1, GL_RGB, GL_UNSIGNED_BYTE, (GLubyte * len(color))(*color))
    
    def handler(self):
        if len(SubWindow.bullet_list) > 0:
            for bullet in SubWindow.bullet_list:
                bullet_loc = bullet.mat[:3, 3]
                dist = np.linalg.norm(bullet_loc - self.sphere_loc)
                normal = (bullet_loc - self.sphere_loc) / np.linalg.norm(bullet_loc - self.sphere_loc)
                bullet_dir = bullet.dir

                if bullet.inside and dist > self.sphere_r:
                    bullet.inside = False
                    bullet.dir = self.calculate_refraction(bullet_dir, normal, 1.2, 1.0)
                elif not bullet.inside and dist < self.sphere_r:
                    bullet.inside = True
                    bullet.dir = self.calculate_refraction(bullet_dir, -normal, 1.0, 1.2)

                bullet.dir /= np.linalg.norm(bullet.dir)
                trans = np.eye(4)
                trans[:3, 3] = bullet.dir * 0.3
                bullet.mat = np.dot(trans, bullet.mat)

                # 충돌 및 경계 조건 처리
                if np.any(bullet_loc < 0.) and np.all(bullet_loc < 2.0):
                    ind = np.argmin(bullet_loc)
                    bullet_loc -= bullet.dir * (bullet_loc[ind] / bullet.dir[ind])
                    #print(bullet_loc, "에서 충돌")
                    if ind == 1:
                        SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [77, 77, 77], bullet.order])
                    elif ind == 0:
                        if bullet_loc[1]%0.5 < 0.25:
                            if bullet_loc[2]%0.5 < 0.25:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [0, 0, 0], bullet.order])
                            else:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [255, 255, 255], bullet.order])
                        else:
                            if bullet_loc[2]%0.5 < 0.25:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [255, 255, 255], bullet.order])
                            else:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [0, 0, 0], bullet.order])
                    else:
                        if bullet_loc[1]%0.5 < 0.25:
                            if bullet_loc[0]%0.5 < 0.25:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [0, 0, 0], bullet.order])
                            else:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [255, 255, 255], bullet.order])
                        else:
                            if bullet_loc[0]%0.5 < 0.25:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [255, 255, 255], bullet.order])
                            else:
                                SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [0, 0, 0], bullet.order])
                    SubWindow.bullet_list.remove(bullet)
                elif np.any(bullet_loc < 0.):
                    ind = np.argmin(bullet)
                    if bullet.dir[ind] < 0.:
                        SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [0, 0, 255], bullet.order])
                        SubWindow.bullet_list.remove(bullet)
                elif np.any(bullet_loc > 23.):
                    SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [0, 0, 255], bullet.order])
                    SubWindow.bullet_list.remove(bullet)       
            #threading.Timer(0.01, self.handler).start()
            self.handler()
        else:
            self.RenderPixel = True
    
    def calculate_refraction(self, bullet_dir, normal, n_i, n_t):
        cos_theta_i = np.dot(bullet_dir, normal)
        sin_theta_i = np.sqrt(1 - cos_theta_i ** 2)
        sin_theta_t = (n_i / n_t) * sin_theta_i
        if sin_theta_t > 1:
            # 전반사 조건
            return bullet_dir - 2 * cos_theta_i * normal
        else:
            # 굴절 조건
            cos_theta_t = np.sqrt(1 - sin_theta_t ** 2)
            return (n_t / n_i) * bullet_dir + ((n_t / n_i) * cos_theta_i - cos_theta_t) * normal
    
    def fire(self, start, dir, x_ind, y_ind, ith):
        speed = 0.1
        bullet = Bullet(start, dir)
        bullet.coord = [int(x_ind), int(y_ind)]
        init_mat = np.eye(4)
        init_mat[:3,3] = start
        bullet.mat = init_mat
        bullet.order = ith
        SubWindow.bullet_list.append(bullet)

    def display(self):
        """
        Display callback function for the subwindow.
        """
        glutSetWindow(self.id)
        self.drawScene()
        
        if self.RenderPixel:
            self.draw_center_dot()
        glutSwapBuffers()
    
    def press_d(self):
        self.look_from = self.init_from
        self.look_at = np.array([0.,0.,0.])
        self.cam_up = self.up_init
        self.cur_mat = np.eye(3)
        self.fin_rot = np.eye(3)
    
    def render(self):
        width, height = self.width, self.height
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT)
        print(data.max(), data.min())
        ind_set = []
        for x in range(width):
            for y in range(height):
                color = data[x][y]
                r, g, b = color[0], color[1], color[2]
                if r == 0.0 and g > 0.1 and b == 0.0:
                    SubWindow.shade_list.append(g)
                    ind_set.append((height-y, width-x))
        print("총 녹색 픽셀 수:", len(ind_set))
        
        world_coords = [self.start_pos(500-x, y) for x, y in ind_set]
        '''cam_start, direction = world_coords[5000][0], world_coords[5000][1]
        self.fire(cam_start, direction, 500-ind_set[5000][0], 500-ind_set[5000][1])'''
        
        iters = len(ind_set)
        for i in range(iters):
            cam_start, direction = world_coords[i][0], world_coords[i][1]
            self.fire(cam_start, direction, 500-ind_set[i][0], 500-ind_set[i][1], i)
        #threading.Timer(0.03, self.handler).start()
        self.handler()
    
    def start_pos(self, x_ind, y_ind):
        cam_loc = self.look_from*-1
        R = np.eye(4)
        R[:3, :3] = self.fin_rot
        a = np.array([cam_loc[0], cam_loc[1], cam_loc[2],1])
        cam_loc = np.dot(R, a)[:3]

        direction = np.array([0.,0.,0.]) - cam_loc
        direction /= np.linalg.norm(direction) #cam의 z벡터
        
        b = np.array([self.cam_up[0], self.cam_up[1], self.cam_up[2],1])
        up = np.dot(R, b)[:3]   #cam의 y벡터
        cam_right = np.cross(up, direction)

        cam_start = cam_loc + (2.*x_ind/self.width-1.)*cam_right + (1.-2.*y_ind/self.height)*up
        return cam_start, direction

    def drawScene(self):
        """
        Draws scene with objects to the subwindow.
        """
        glutSetWindow(self.id)

        glClearColor(0, 0, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(self.projectionMat.T)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(self.viewMat.T)

        if self.id == 2:
            #gluPerspective(self.fov, 1., 0.01, 100.0)
            glOrtho(-1, 1, -1, 1, 0.01, 100.0)
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
        for bullet in SubWindow.bullet_list:
            bullet.draw()
        
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
            self.RenderPixel = False
            SubWindow.shade_list = []
            SubWindow.pixel_list = []
            obj_id = self.pickObject(x, y)
            if obj_id != 0xFFFFFF:
                print(f"{obj_id} selected")
            else:
                print("Nothing selected")
        elif button == GLUT_LEFT_BUTTON and state == GLUT_UP:
            self.track_ball = False

        if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
            print(f"Add teapot at ({x}, {y})")
            #self.addTeapot(x, y)

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
            start_x = (1.0-2.0*start_x/self.width)+cur_loc[0]
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
            dest_x = (1.0-2.0*dest_x/self.width)+cur_loc[0]
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

    def addTeapot(self, x, y):
        # this function should be implemented
        teapot = Teapot()
        # update teapot.mat, etc. to complete your tasks
        SubWindow.obj_list.append(teapot)


class Viewer:
    width, height = 500, 500

    def __init__(self):
        self.x = 50.0
        self.y = 80.0
        self.z = 50.0

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
        lightPosition = [self.x, self.y, self.z, 1]  # vector: point at infinity
        #print(lightPosition)
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
        self.subWindow.display()

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
        
        if key==b'l':
            self.light()

        if key==b'r':
            #r을 눌렀을 때.
            self.subWindow.render()
        
        if key==b'x':
            if self.x > 20.0:
                self.x -= 10
                self.light()
        elif key==b'y':
            if self.y > 20:
                self.y -= 10
                self.light()
        elif key==b'z':
            if self.z > 20:
                self.z -= 10
                self.light()
        elif key==b'X':
            if self.x < 100:
                self.x += 10
                self.light()
        elif key==b'Y':
            if self.y < 100:
                self.y += 10
                self.light()
        elif key==b'Z':
            if self.z < 100:
                self.z += 10
                self.light()
        
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
        glutIdleFunc(self.idle)

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
