import os
import queue
import subprocess
import sys
sys.path.append("../binvox")

from multiprocessing import Queue

import numpy as np
import trimesh

from OpenGL.GL import *
from OpenGL.GLU import *

import nanogui
from nanogui import (Alignment, BoxLayout, Button, CheckBox, Color, ColorWheel,
                     ComboBox, GLCanvas, GLShader, Graph, GridLayout,
                     GroupLayout, ImagePanel, ImageView, IntBox, Label,
                     MessageDialog, Orientation, PopupButton, ProgressBar,
                     Screen, Slider, TabWidget, TextBox, ToolButton,
                     VScrollPanel, Widget, Window, gl, glfw, GLFramebuffer)
from viewer import camera
from viewer.datasources import (BinvoxGrid, ScatterGrid, TriangleMesh,
                                VectorField, VoxelGrid, PointCloud)
from viewer.render import *

import gc

import matplotlib.pyplot as plt


class GLTexture:

    def __init__(self, data):
        self.id = glGenTextures(1)
        channels = data.shape[2] if data.ndim >= 3 else 1
        w = data.shape[1]
        h = data.shape[0]
        fmt = [GL_RED, GL_RG, GL_RGB, GL_RGBA][channels - 1]
        if data.dtype == np.uint8:
            internal_fmt = [GL_R8, GL_RG8, GL_RGB8, GL_RGBA8][channels - 1]
            dtype = GL_UNSIGNED_BYTE
        elif data.dtype == np.float32 or data.dtype == np.float64:
            internal_fmt = [GL_R32F, GL_RG32F, GL_RGB32F, GL_RGBA32F][channels - 1]
            dtype = GL_FLOAT

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, w, h, 0, fmt, dtype, data)
        glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, w, h, 0, fmt, dtype, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    def __del__(self):
        glDeleteTextures(self.id)


class RenderContext:
    def __init__(self):
        try:
            self.shader = GLShader()
            load_shader(self.shader, 'viewer/shaders/simple', 'a_simple_shader')
            self.shader_points = GLShader()
            load_shader(self.shader_points, 'viewer/shaders/point', 'a_point_shader')
            self.shader_dashed = GLShader()
            load_shader(self.shader, 'viewer/shaders/simple', 'a_simple_shader2')
            self.shader_simple = GLShader()
            load_shader(self.shader_dashed, 'viewer/shaders/dashed', 'a_dashed_line_shader')
            self.shader_raymarch = GLShader()
            load_shader(self.shader_raymarch, 'viewer/shaders/raymarch', 'a_raymarching_shader')
            self.shader_scatter = GLShader()
            load_shader(self.shader_scatter, 'viewer/shaders/scatter', 'a_scatter_viz_shader')
            self.shader_basic_surface = GLShader()
            load_shader(self.shader_basic_surface, 'viewer/shaders/basic_surface', 'a_basic_surface_shader')
            self.shader_alpha_surface = GLShader()
            load_shader(self.shader_alpha_surface, 'viewer/shaders/alpha_surface', 'a_transparent_surface_shader')
            self.shader_colored_surface = GLShader()
            load_shader(self.shader_colored_surface, 'viewer/shaders/colored_surface', 'a_colored_surface_shader')
            # self.shader_facetted = GLShader()
            # load_shader(self.shader_facetted, 'viewer/shaders/facetted', 'a_facetted_shader')
            self.shader_quad = GLShader()
            load_shader(self.shader_quad, 'viewer/shaders/quad', 'a_quad_shader')
        except ImportError:
            print("Shader compilation failed")
            quit()
        self.trilinear_interpolation = True
        self.its_loc = np.array([0, 0, 0], dtype=np.float32)

        # Get desktop scaling factor
        if 'XDG_CURRENT_DESKTOP' in os.environ and os.environ['XDG_CURRENT_DESKTOP']:
            self.resolution_scale = float(subprocess.check_output(
                'kreadconfig5 --group KScreen --key ScaleFactor', shell=True).decode('utf-8'))
        else:  # This is currently hardcoded for OSX
            self.resolution_scale = 2.0


class ViewerApp(Screen):
    """Main window for scatter viewer application"""

    def __init__(self):
        super(ViewerApp, self).__init__((1280, 1024), "Scattering Viewer", True, nSamples=8, depthBits=32)

        # Generic event queue. Threads can deposit events here which then have to be handled by overrideing handleEvent method
        self.event_queue = Queue()

        self.camera = camera.Camera(self.size())
        self.camera.setLookAt(np.array([0, 20, 20]), np.array([0, 0, 0]))
        self.camera_controller = camera.CameraController(self.camera)
        self.render_context = RenderContext()
        self.setBackground(Color(0.2, 0))

        self.fb = GLFramebuffer()
        self.fb.init(self.size() * self.render_context.resolution_scale, 1)

        self.show_grid = True

    def handleEvent(self, event):
        '''Generic event handling method'''
        pass

    def handleEvents(self):
        '''Handles at most one event per frame of drawing. Event handling should be fast to prevent lag'''
        try:
            self.handleEvent(self.event_queue.get(False))
        except:
            pass

    def resizeEvent(self, new_size):
        if new_size[0] < 32 or new_size[1] < 32:  # prevent 0 x 0 window
            self.setSize(np.maximum(new_size, [64, 64]))
            return True
        self.camera.aspect = new_size[0] / new_size[1]
        self.camera.resolution = new_size
        self.camera.updateViewProj()

        self.fb.free()
        self.fb = GLFramebuffer()
        self.fb.init(self.size() * self.render_context.resolution_scale, 1)

        super(ViewerApp, self).drawAll()
        return True

    def scrollEvent(self, p, rel):
        if super(ViewerApp, self).scrollEvent(p, rel):
            return True
        return self.camera_controller.scrollEvent(p, rel)

    def exitEvent(self):
        pass

    def keyboardEvent(self, key, scancode, action, modifiers):
        if super(ViewerApp, self).keyboardEvent(key, scancode, action, modifiers):
            return True

        if (key == glfw.KEY_ESCAPE or key == glfw.KEY_Q) and action == glfw.PRESS:
            self.exitEvent()
            self.setVisible(False)
            return True
        return self.camera_controller.keyboardEvent(key, scancode, action, modifiers)

    def mouseMotionEvent(self, p, rel, button, modifier):
        if super(ViewerApp, self).mouseMotionEvent(p, rel, button, modifier):
            return True

        return self.camera_controller.mouseMotionEvent(p, rel, button, modifier)

    def drawContents(self):
        super(ViewerApp, self).drawContents()
        self.handleEvents()
        self.fb.bind()
        color = self.background()
        glClearColor(color.r, color.g, color.b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        self.render_context.shader.bind()

        gl.Enable(gl.DEPTH_TEST)
        if self.show_grid:
            draw_grid(self.render_context.shader, self.camera.viewproj)

    def drawContentsPost(self):
        pass

    def drawContentsFinalize(self):
        self.fb.release()
        self.fb.blit()
        glClear(GL_DEPTH_BUFFER_BIT)

        self.drawContentsPost()


class VoxelGridViewer(ViewerApp):
    """Viewer to visualize a given fixed voxelgrid"""

    def __init__(self, voxel_grids, normals=None, mesh_file=None):
        super(VoxelGridViewer, self).__init__()
        self.disp_idx = 0
        self.voxel_grids = voxel_grids
        self.normals = normals

        if mesh_file is not None:
            self.mesh = TriangleMesh(mesh_file)
        else:
            self.mesh = None

        window = Window(self, "VoxelGridViewer")
        window.setPosition((15, 15))
        window.setLayout(GroupLayout())

        tools = Widget(window)
        tools.setLayout(BoxLayout(Orientation.Horizontal,
                                  Alignment.Middle, 0, 5))

        self.label = Label(window, self.voxel_grids[self.disp_idx][0])

        self.selected_voxel = None
        self.performLayout()

    def keyboardEvent(self, key, scancode, action, modifiers):
        if super(VoxelGridViewer, self).keyboardEvent(key, scancode, action, modifiers):
            return True

        num_keys = [glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5,
                    glfw.KEY_6, glfw.KEY_7, glfw.KEY_8, glfw.KEY_9, glfw.KEY_0]

        if action == glfw.PRESS and key in num_keys:
            idx = num_keys.index(key)
            self.disp_idx = np.minimum(len(self.voxel_grids) - 1, idx)
            self.label.setCaption(self.voxel_grids[self.disp_idx][0])
            self.performLayout()
            return True
        return False

    def update_displayed_scattering(self, p):
        d = get_view_ray_dir(p, self.size(), self.camera)
        if self.mesh:
            intersector = self.mesh.mesh.ray
            its_loc, ray_idx, tri_idx = intersector.intersects_location(
                self.camera.pos[np.newaxis, :], d[np.newaxis, :])
            if len(its_loc) > 0:
                dist_to_cam = np.sum((its_loc - self.camera.pos) ** 2, axis=1)
                its_loc = np.array(its_loc[np.argmin(dist_to_cam), :], dtype=np.float32)
                voxel_idx = self.voxel_grids[0][1].point_to_vox_idx(its_loc)

                # Update displayed voxel
                self.render_context.its_loc = its_loc
                self.selected_voxel = voxel_idx

                print('Voxel Index: {}'.format(voxel_idx))
                print('Selected point {}'.format(its_loc))

    def mouseButtonEvent(self, p, button, action, modifier):
        if super(VoxelGridViewer, self).mouseButtonEvent(p, button, action, modifier):
            return True
        if button == 0 and action == glfw.PRESS and modifier == 0 and self.mesh is not None:
            self.update_displayed_scattering(p)
            return True
        return False

    def drawContents(self):
        super(VoxelGridViewer, self).drawContents()

        self.render_context.shader.bind()
        gl.Enable(gl.DEPTH_TEST)
        draw_box(self.render_context.shader, self.voxel_grids[self.disp_idx][1].vxbb_min,
                 self.voxel_grids[self.disp_idx][1].vxbb_max, self.camera.viewproj, np.array([1, 1, 0], dtype=np.float32))

        if self.mesh is not None:
            self.mesh.draw_contents(self.camera, self.render_context, self.voxel_grids[self.disp_idx][1])
        self.voxel_grids[self.disp_idx][1].draw_contents(self.camera, self.render_context)
        if self.normals is not None:
            self.normals.draw_contents(self.camera, self.render_context)

        if self.selected_voxel is not None:
            vxgrid = self.voxel_grids[0][1]
            voxel_diag_ws = (vxgrid.vxbb_max - vxgrid.vxbb_min) / vxgrid.voxel_grid_res
            # Round position to index

            minvox = (self.selected_voxel / vxgrid.voxel_grid_res) * \
                (vxgrid.vxbb_max - vxgrid.vxbb_min) + vxgrid.vxbb_min

            maxvox = minvox + voxel_diag_ws
            draw_box(self.render_context.shader, minvox, maxvox,
                     self.camera.viewproj, np.array([1, 0, 0], dtype=np.float32))


def compare_voxel_grids(names, data):
    nanogui.init()

    app = VoxelGridViewer([x for x in zip(names, data)])
    app.drawAll()
    app.setVisible(True)

    nanogui.mainloop()
    del app
    gc.collect()
    nanogui.shutdown()
