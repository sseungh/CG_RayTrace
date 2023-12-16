from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from numpy.linalg import inv
from PIL import Image # conda install pillow
#import threading
from tqdm import tqdm

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

class AABBNode:
    vertices = np.empty((0, 3), dtype='float32')
    def __init__(self):
        self.min = [float('inf'), float('inf'), float('inf')]
        self.max = [float('-inf'), float('-inf'), float('-inf')]
        self.children = []
        self.f = None
        self.ind = None
    
    def edge_node(self, f, _v, ind):
        # edge 노드를 위한 함수
        self.f = f
        self.ind = ind
        for v_id in f:
            v = _v[v_id]
            for i in range(3):
                self.min[i] = min(self.min[i], v[i])
                self.max[i] = max(self.max[i], v[i])
    
    def merge(self, ch1, ch2):
        # parent노드를 위한 함수
        for i in range(3):
            self.min[i] = min(ch1.min[i], ch2.min[i])
            self.max[i] = max(ch1.max[i], ch2.max[i])
        self.children = [ch1, ch2]


class Glass(Object):
    loaded = False
    _v = [] # vertices
    _f = [] # v_id array, vn id array
    _n = []
    root = None

    def __init__(self):
        super().__init__()
        if not Glass.loaded:
            self._load_obj("glass.obj")
            Glass.loaded = True
    
    def tree_check(self, aabb, n):
        if n < 10:
            if len(aabb.children)!=2:
                print("binary error", n)
                return
            self.tree_check(aabb.children[0], n+1)
            self.tree_check(aabb.children[1], n+1)
        else:
            if len(aabb.children)!=0:
                print("leaf error", len(aabb.children))
                return

    def _load_obj(self, filename):
        #문제발생 가능. 동시에 호출될 수도 있음
        print("Glass loaded first time")
        file = open(filename, 'r')
        minx, miny, minz = float("inf"), float("inf"), float("inf")
        maxx, maxy, maxz = float("-inf"), float("-inf"), float("-inf")
        for line in file:
            words = line.split()
            if len(line) == 0:
                continue
            if words[0] == 'v':
                values = []
                for n in words[1:]:
                    values.append(float(n))
                minx = min(minx, values[0])
                maxx = max(maxx, values[0])
                miny = min(miny, values[1])
                maxy = max(maxy, values[1])
                minz = min(minz, values[2])
                maxz = max(maxz, values[2])
                Glass._v.append(values)
            elif words[0] == 'f':
                face_vertices = []
                for part in words[1:]:
                    face_vertices.append(int(part) - 1)
                #face_vertices.sort()
                Glass._f.append((face_vertices))  # v_id와 vn_id가 동일
        
        c_x, c_y, c_z = (maxx+minx)/2, (maxy+miny)/2, (maxz+minz)/2
        size = len(Glass._v)
        for i in range(size):
            Glass._v[i][0] = (Glass._v[i][0]-c_x)*2
            Glass._v[i][1] = (Glass._v[i][1]-c_y)*2
            Glass._v[i][2] = (Glass._v[i][2]-c_z)*2
        
        for face in Glass._f:
            v0 = np.array(Glass._v[face[0]])
            v1 = np.array(Glass._v[face[1]])
            v2 = np.array(Glass._v[face[2]])
            vec1 = v1 - v0
            vec2 = v2 - v0
            normal = np.cross(vec1, vec2)
            normal = normal / np.linalg.norm(normal)
            Glass._n.append(normal)
        
        # AABB 트리 설계, _f의 길이가 1024는 검증 완료
        queue = np.array([], dtype=object)
        ind = 0
        for f in Glass._f:
            node = AABBNode()
            node.edge_node(f, Glass._v, ind)
            ind += 1
            queue = np.append(queue, node)
        # 1024 = 2^10 임으로 완벽하게 짝이 맞음
        while queue.size > 1:
            ch1 = queue[0]
            ch2 = queue[1]
            queue = queue[2:]
            parent = AABBNode()
            parent.merge(ch1, ch2)
            queue = np.append(queue, parent)
        Glass.root = queue[0]
        #self.tree_check(Glass.root, 0)
        
        file.close()

    def draw(self):
        vertices = np.array(Glass._v, dtype='float32').flatten()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPushMatrix()
        glMultMatrixf(self.mat.T)
        glColor3f(0.0, 1.0, 0.0)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)

        # 각 면을 개별적으로 그립니다.
        for face in Glass._f:
            glBegin(GL_POLYGON)  # 면의 형태에 따라 GL_TRIANGLES, GL_QUADS 등으로 변경 가능
            for vertex_index in face:
                glVertex3fv(vertices[3 * vertex_index:3 * vertex_index + 3])
            glEnd()
        glColor3f(1.0, 1.0, 1.0)
        glPopMatrix()


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
        self.prev_ind = -1
    
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
        self.glass = Glass()
        self.glass.mat[:3,3] = [0.9, 0.35, 0.9]
        SubWindow.obj_list.append(self.glass)
        env = Env()
        SubWindow.obj_list.append(env)

        self.fov = 3
        self.init_from = np.array([-10., -4., -10.])
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

        #self.sphere_r = sphere.r
        #self.sphere_loc = sphere.mat[:3,3]
        self.RenderPixel = False
    
    def draw_center_dot(self):
        for ind, color, order in SubWindow.pixel_list:
            glWindowPos2i(ind[0], ind[1])
            temp = SubWindow.shade_list[order]
            color = [int(temp*color[0]), int(temp*color[1]), int(temp*color[2])]
            glDrawPixels(1, 1, GL_RGB, GL_UNSIGNED_BYTE, (GLubyte * len(color))(*color))
    
    def handler(self):
        if len(SubWindow.bullet_list) >0:
            for bullet in SubWindow.bullet_list:
                bullet_loc = bullet.mat[:3, 3]
                bullet_dir = bullet.dir
                coord = bullet_dir*0.3
                check, norm, _ind = self.bounce_glass(self.glass.mat, bullet_loc, coord)
                if check and bullet.prev_ind!=_ind:
                    bullet.prev_ind = _ind
                    if bullet.inside:
                        bullet.inside = False
                        bullet.dir = self.calculate_refraction(bullet_dir, norm, 1.2, 1.0)
                    else:
                        bullet.inside = True
                        bullet.dir = self.calculate_refraction(bullet_dir, -norm, 1.0, 1.2)

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
                    ind = np.argmin(bullet_loc)
                    if bullet.dir[ind] < 0.:
                        SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [100, 100, 255], bullet.order])
                        SubWindow.bullet_list.remove(bullet)
                elif np.any(bullet_loc > 23.):
                    SubWindow.pixel_list.append([[bullet.coord[0], bullet.coord[1]], [100, 100, 255], bullet.order])
                    SubWindow.bullet_list.remove(bullet)       
            #threading.Timer(0.01, self.handler).start()
            return False
            #self.handler()
        else:
            self.RenderPixel = True
            return True
    
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
        # bullet
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
        '''
        cam_start, direction = world_coords[50][0], world_coords[50][1]
        self.fire(cam_start, direction, 500-ind_set[50][0], 500-ind_set[50][1], 50)
        '''
        iters = len(ind_set)
        for i in tqdm(range(iters)):
            cam_start, direction = world_coords[i][0], world_coords[i][1]
            self.fire(cam_start, direction, 500-ind_set[i][0], 500-ind_set[i][1], i)
            while not self.handler():
                continue
        #threading.Timer(0.03, self.handler).start()
        
    
    def ver_move(self, v_init, mat):
        v = np.array(v_init + [1.0])
        v_moved = np.dot(mat, v)
        return v_moved[:3].tolist()

    def bounce_glass(self, mat, org, diff):
        coord = org + diff
        stack = [Glass.root]
        while stack:
            aabb = stack.pop()
            min1 = self.ver_move(aabb.min, mat)
            max1 = self.ver_move(aabb.max, mat)
            outer = True
            for i in range(3):
                if min1[i]>max1[i]:
                    temp = min1[i]
                    min1[i] = max1[i]
                    max1[i] = temp
            # org or coord is inside the aabb
            in_org = True
            in_coord = True
            panet = True
            half = org + diff/2
            for i in range(3):
                if (min1[i] <= org[i] <= max1[i]) == False:
                    in_org = False
                if (min1[i] <= coord[i] <= max1[i]) == False:
                    in_coord = False
                if (min1[i] <= half[i] <= max1[i]) == False:
                    panet = False
            
            if in_org or in_coord or panet:
                if aabb.f:
                    #print("candidate")
                    f = aabb.f
                    n = Glass._n[aabb.ind]
                    '''
                    d = -np.dot(n, vertices[0])
                    if np.dot(n, diff)*-1 > 1e-6:
                        t = -(np.dot(n, org) + d) / np.dot(n, diff)
                        ret = org + t * diff
                        c1 = np.cross(vertices[1]-vertices[0], ret-vertices[0])
                        c2 = np.cross(vertices[2]-vertices[1], ret-vertices[1])
                        c3 = np.cross(vertices[0]-vertices[2], ret-vertices[2])
                        if np.dot(c1,c2)>=0 and np.dot(c2,c3)>=0:
                            return True, ret, n
                    '''
                    return True, n, aabb.ind
                else:
                    stack.append(aabb.children[0])
                    stack.append(aabb.children[1])
        return False, None, None
    
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

        glClearColor(0.39, 0.39, 1, 1)
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
