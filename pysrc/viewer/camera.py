import nanogui
import numpy as np

from nanogui import Color, Screen, Window, GroupLayout, BoxLayout, \
    ToolButton, Label, Button, Widget, \
    PopupButton, CheckBox, MessageDialog, VScrollPanel, \
    ImagePanel, ImageView, ComboBox, ProgressBar, Slider, \
    TextBox, ColorWheel, Graph, GridLayout, \
    Alignment, Orientation, TabWidget, IntBox, GLShader, GLCanvas, gl, glfw

class Camera:

    def __init__(self, resolution):
        self.lookAt = np.array([0, 0, -1])
        self.up = np.array([0, 1, 0])
        self.right = np.array([1, 0, 0])
        self.permanent_up = np.array([0, 1, 0])

        self.fov = 45
        self.far = 5000
        self.near = 0.1
        self.pos = np.array([0, 0, 20])
        self.resolution = resolution
        self.aspect = self.resolution[0] / self.resolution[1]
        self.updateViewProj()


    def getProj(self):
        height = self.near * np.tan(self.fov / 2 / 360 * 2 * np.pi)
        width = height * self.aspect
        return nanogui.frustum(-width, width, -height, height, self.near, self.far)


    def updateViewProj(self):        
        rot = np.matrix(
            [[self.right[0], self.up[0], -self.lookAt[0], 0],
             [self.right[1], self.up[1], -self.lookAt[1], 0],
             [self.right[2], self.up[2], -self.lookAt[2], 0],
             [0, 0, 0, 1]])

        view = np.matrix(
            [[1, 0, 0, -self.pos[0]],
             [0, 1, 0, -self.pos[1]],
             [0, 0, 1, -self.pos[2]],
             [0, 0, 0, 1]],
            dtype=np.float32)
        self.view = rot.transpose().dot(view)
        self.viewproj = self.getProj().dot(self.view)
    

    def setLookAt(self, origin, target):
        self.pos = origin
        self.target = target
        d = target - origin
        self.lookAt = d / np.sqrt(np.dot(d, d))

        self.right = np.cross(self.lookAt, self.permanent_up)
        self.right = self.right / np.sqrt(np.dot(self.right, self.right))
        self.up = np.cross(self.right, self.lookAt)
        self.up = self.up / np.sqrt(np.dot(self.up, self.up))
        self.pos = origin
        self.updateViewProj()


    def setTarget(self, target):
        self.target = target
        self.setLookAt(self.pos, target)
        self.updateViewProj()


    def rotMat(self, u, angle):
        cosTheta = np.cos(angle)
        sinTheta = np.sin(angle)
        return np.matrix(
            [[cosTheta + u[0]**2 * (1 - cosTheta), u[0] * u[1] * (1 - cosTheta) - u[2] * sinTheta, u[0] * u[2] * (1 - cosTheta) + u[1] * sinTheta],
             [u[1] * u[0] * (1 - cosTheta) + u[2] * sinTheta, cosTheta + u[1] ** 2 *
              (1 - cosTheta), u[1] * u[2] * (1 - cosTheta) - u[0] * sinTheta],
             [u[2] * u[0] * (1 - cosTheta) - u[1] * sinTheta, u[2] * u[1] * (1 - cosTheta) + u[0] * sinTheta, cosTheta + u[2]**2 * (1 - cosTheta)]]
        )


    def rotate(self, relOffset):
        relOffset = 0.0075 * relOffset
        targetVec = -self.target + self.pos
        desiredDist = np.sqrt(np.dot(targetVec, targetVec))
        targetVec = targetVec / desiredDist

        matUp = self.rotMat(self.right, -relOffset[1])
        matRight = self.rotMat(self.permanent_up, -relOffset[0])

        rotMat = matUp.dot(matRight)
        newVec = np.ravel(rotMat.dot(np.transpose(targetVec)))
        newVec = newVec / np.sqrt(np.sum(newVec ** 2))

        if np.abs(newVec.dot(self.permanent_up)) < 0.999:
            newPos = self.target + newVec * desiredDist
            self.setLookAt(newPos, self.target)
        self.updateViewProj()


    def translate(self, relOffset):
        relOffset = 0.001 * relOffset * np.sqrt(np.sum((self.target - self.pos) ** 2))
        offset = relOffset[1] * self.up - relOffset[0] * self.right
        self.setLookAt(self.pos + offset, self.target + offset)
        self.updateViewProj()


class CameraController:


    def __init__(self, camera):
        self.camera = camera
        self.selectedPoint = np.array([0,0,0])


    def mouseButtonEvent(self, p, button, action, modifier):
        return False


    def scrollEvent(self, p, rel):
        cam_speed = np.maximum(0.5, np.sqrt(np.sum((self.camera.pos - self.camera.target) ** 2)) / 10)
        if rel[1] > 0:
            self.camera.pos = self.camera.pos + self.camera.lookAt * cam_speed
            self.camera.updateViewProj()
            return True

        if rel[1] < 0:
            self.camera.pos = self.camera.pos - self.camera.lookAt * cam_speed
            self.camera.updateViewProj()
            return True
        return True


    def keyboardEvent(self, key, scancode, action, modifiers):
        cam_speed = 0.8 if modifiers == glfw.MOD_SHIFT else 0.2

        if key == glfw.KEY_W:
            self.camera.pos = self.camera.pos + self.camera.lookAt * cam_speed
        if key == glfw.KEY_S:
            self.camera.pos = self.camera.pos - self.camera.lookAt * cam_speed
        if key == glfw.KEY_D:
            self.camera.pos = self.camera.pos + self.camera.right * cam_speed
        if key == glfw.KEY_A:
            self.camera.pos = self.camera.pos - self.camera.right * cam_speed
        if key == glfw.KEY_E:
            self.camera.pos = self.camera.pos + self.camera.up * cam_speed
        if key == glfw.KEY_C:
            self.camera.pos = self.camera.pos - self.camera.up * cam_speed
        if key == glfw.KEY_F and action == glfw.PRESS:
            self.camera.setLookAt(self.camera.pos, self.selectedPoint)
        self.camera.updateViewProj()
        return False


    def mouseMotionEvent(self, p, rel, button, modifier):
        if modifier == glfw.MOD_ALT and button == 1:
            self.camera.rotate(rel)
            return True
        if modifier == glfw.MOD_ALT and button == 4:
            self.camera.translate(rel)
            return True
        if modifier == glfw.MOD_ALT and button == 2:
            self.camera.pos = self.camera.pos - self.camera.lookAt * \
                rel[1] * np.maximum(0.005, np.sqrt(np.sum((self.camera.pos - self.camera.target) ** 2)) / 100)
            self.camera.updateViewProj()
            return True
        return False
