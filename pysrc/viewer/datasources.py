import gc
import os

import numpy as np
import scipy
import trimesh
from OpenGL.GL import *
from OpenGL.GLU import *

import binvox.binvox_rw
import nanogui
from viewer.camera import *
from nanogui import (Alignment, BoxLayout, Button, CheckBox, Color, ColorWheel,
                     ComboBox, GLCanvas, GLShader, Graph, GridLayout,
                     GroupLayout, ImagePanel, ImageView, IntBox, Label,
                     MessageDialog, Orientation, PopupButton, ProgressBar,
                     Screen, Slider, TabWidget, TextBox, ToolButton,
                     VScrollPanel, Widget, Window, gl, glfw)


from utils.math import reshape_sparse
import voxels
from voxels import read_voxelgrid_metadata
from viewer.render import get_color_map, draw_lines

from utils.printing import printg


class TriangleMesh:
    """Class representing a simple triangle mesh"""

    def __init__(self, filename, verts=None, faces=None):
        self.texture = glGenTextures(1)  # TODO: This should be done once we are sure that the OpenGL context exists
        self.texture2 = glGenTextures(1)
        self.cmap_img = get_color_map()

        self.translation = np.zeros(3)
        self.scale = np.ones(3)

        glBindTexture(GL_TEXTURE_3D, self.texture)
        glBindTexture(GL_TEXTURE_1D, self.texture2)

        if filename is not None:
            if os.path.isfile(filename):
                self.mesh = trimesh.load_mesh(filename)
                self.mesh_positions = self.mesh.vertices.astype(np.float32).T
                self.mesh_faces = self.mesh.faces.astype(np.int32).T
                self.mesh_normal = self.mesh.vertex_normals.astype(np.float32).T

                # Meshes seem to be loaded flipped (culling, not axes). Therefore reverse order of triangles
                # tmp = np.array(self.mesh_faces[0, :], dtype=np.int32)
                # self.mesh_faces[0, :] = self.mesh_faces[2, :]
                # self.mesh_faces[2, :] = tmp
            else:
                print('Error: Mesh {} not found'.format(filename))
        else:
            self.mesh = trimesh.Trimesh(verts, faces)

            # self.mesh_positions = verts.astype(np.float32).T
            # self.mesh_faces = faces.astype(np.int32).T
            self.mesh_positions = self.mesh.vertices.astype(np.float32).T
            self.mesh_normal = self.mesh.vertex_normals.astype(np.float32).T
            self.mesh_faces = self.mesh.faces.astype(np.int32).T

    def draw_contents(self, camera, context, scatter_data, in_color=None, 
                      bb_min=np.array([-1, -1, -1], np.float32), bb_max=np.array([1, 1, 1], np.float32), vertex_colors=None):
        """Draws the triangle mesh using the camera and shader which are passed in"""

        if vertex_colors is not None:
            shader = context.shader_colored_surface
        elif scatter_data is not None:
            shader = context.shader_scatter
        else:
            # shader = context.shader_basic_surface
            shader = context.shader_alpha_surface
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        shader.bind()

        s = self.scale
        model_matrix = trimesh.transformations.translation_matrix(self.translation) @ np.diag([s[0], s[1], s[2], 1.0])

        if scatter_data is not None:
            glUniform1i(shader.uniform("tex"), 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self.texture)

            if type(scatter_data) == ScatterGrid:
                bb_min = scatter_data.vxbb_min
                bb_max = scatter_data.vxbb_max
                glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, scatter_data.surf_voxel_data.shape[0], scatter_data.surf_voxel_data.shape[0],
                             scatter_data.surf_voxel_data.shape[0], 0, GL_RED, GL_FLOAT, scatter_data.surf_voxel_data)

                # glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, scatter_data.voxel_data.shape[0], scatter_data.voxel_data.shape[0],
                #         scatter_data.voxel_data.shape[0], 0, GL_RED, GL_FLOAT, scatter_data.voxel_data)
            else:
                glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, scatter_data.shape[0], scatter_data.shape[0],
                             scatter_data.shape[0], 0, GL_RED, GL_FLOAT, scatter_data)

            interp = GL_LINEAR if context.trilinear_interpolation else GL_NEAREST
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, interp)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, interp)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

            glUniform1i(shader.uniform("cmap"), 1)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_1D, self.texture2)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, self.cmap_img.shape[0], 0,
                         GL_RGB, GL_FLOAT, self.cmap_img)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            shader.setUniform("refPosition", context.its_loc)
            shader.setUniform("bbMin", bb_min)
            shader.setUniform("bbMax", bb_max)
        else:
            if in_color is None:
                in_color = np.array([1, 1, 1, 1], dtype=np.float32)

            if vertex_colors is not None: 
                shader.uploadAttrib("color", vertex_colors.T)
            else:
                shader.setUniform("invTModelView", np.linalg.inv(camera.view @ model_matrix).T)
                shader.setUniform("in_color", in_color)
                shader.uploadAttrib("normal", self.mesh_normal)

        shader.setUniform("modelViewProj", camera.viewproj @ model_matrix)
        shader.uploadAttrib("position", self.mesh_positions)
        shader.uploadIndices(self.mesh_faces)
        shader.drawIndexed(gl.TRIANGLES, 0, self.mesh_faces.shape[1])



class FacetedTriangleMesh:
    """Class representing a mesh histogram"""

    def __init__(self, verts, faces, face_colors):
        self.mesh = trimesh.Trimesh(verts, faces)
        self.mesh_positions = self.mesh.vertices.astype(np.float32).T
        self.mesh_normal = self.mesh.vertex_normals.astype(np.float32).T
        self.mesh_faces = self.mesh.faces.astype(np.int32).T

        face_colors = face_colors / (np.max(face_colors) + 1e-8)
        face_colors = np.clip(face_colors, 0.0, 1.0);

        self.face_colors = face_colors

        # Split triangles and assign face colors as vertex colors
        self.new_vertices = np.zeros((3, 3 * self.mesh_faces.shape[1]))
        self.vertex_colors = np.zeros((3, 3 * self.mesh_faces.shape[1]))

        for i in range(self.mesh_faces.shape[1]):
            self.new_vertices[:, 3 * i + 0] = self.mesh_positions[:, self.mesh_faces[0, i]]
            self.new_vertices[:, 3 * i + 1] = self.mesh_positions[:, self.mesh_faces[1, i]]
            self.new_vertices[:, 3 * i + 2] = self.mesh_positions[:, self.mesh_faces[2, i]]
            self.vertex_colors[:, 3 * i + 0] = face_colors[i]
            self.vertex_colors[:, 3 * i + 1] = face_colors[i]
            self.vertex_colors[:, 3 * i + 2] = face_colors[i]

        self.new_faces = np.reshape(np.arange(0, 3 * self.mesh_faces.shape[1]), [self.mesh_faces.shape[1], 3]).T
        self.mesh_positions = self.new_vertices.astype(np.float32)
        self.mesh_faces = self.new_faces.astype(np.int32)
        

    def draw_contents(self, camera, context):
        """Draws the triangle mesh using the camera and shader which are passed in"""
        
        shader = context.shader
        shader.bind()
        shader.setUniform("modelViewProj", camera.viewproj)
        shader.uploadAttrib("position", self.mesh_positions)
        shader.uploadAttrib("color", self.vertex_colors)
        shader.uploadIndices(self.mesh_faces)
        shader.drawIndexed(gl.TRIANGLES, 0, self.mesh_faces.shape[1])


class BoxCloud:
    """Class to repreresent a whole collection of boxes"""

    def box_vertices(self, p0, p1):
        positions = np.zeros([3, 24], np.float32)
        vertices = []
        vertices.append(np.array([p0[0], p0[1], p0[2]]))
        vertices.append(np.array([p1[0], p0[1], p0[2]]))
        vertices.append(np.array([p0[0], p1[1], p0[2]]))
        vertices.append(np.array([p0[0], p0[1], p1[2]]))
        vertices.append(np.array([p1[0], p1[1], p0[2]]))
        vertices.append(np.array([p0[0], p1[1], p1[2]]))
        vertices.append(np.array([p1[0], p0[1], p1[2]]))
        vertices.append(np.array([p1[0], p1[1], p1[2]]))
        idx = 0
        for v in vertices:
            for i in range(0, 3):
                newVert = np.array(v)
                if p1[i] > newVert[i]:
                    newVert[i] = p1[i]
                    positions[:, idx] = v
                    positions[:, idx + 1] = newVert
                    idx += 2
        return np.array(positions, dtype=np.float32)

    def __init__(self, box_min, box_max):
        all_vertices = []
        for p0, p1 in zip(box_min, box_max):
            all_vertices.append(self.box_vertices(p0, p1))

        self.positions = np.concatenate(all_vertices, 1)

    def draw_contents(self, camera, context, scatter_data, color=[1, 0, 0]):
        """Draws the triangle mesh using the camera and shader which are passed in"""
        shader = context.shader
        shader.bind()
        shader.setUniform("modelViewProj", camera.viewproj)
        shader.uploadAttrib("position", self.positions)
        shader.uploadAttrib("color", np.tile(np.array([[color[0]], [color[1]], [color[2]]],
                                                      dtype=np.float32), [1, self.positions.shape[1]]))
        shader.drawArray(gl.LINES, 0, self.positions.shape[1])


class PointCloud:
    """Class representing a simple triangle mesh"""

    def __init__(self, points, colors=None):
        points = np.atleast_2d(points)
        self.points = points.astype(np.float32).T
        if colors is not None and colors.shape[0] == points.shape[0]:
            self.colors = colors.astype(np.float32).T
        else:
            self.colors = None

    def draw_contents(self, camera, context, scatter_data, color=[1, 0, 0], 
                      disable_ztest=False, use_depth=False, depth_map=None,
                      use_ref_point=False, ref_point=np.zeros(3), ref_radius=1.0, cull_occlusions=False):
        """Draws the triangle mesh using the camera and shader which are passed in"""
        glPointSize(10.0)

        if use_depth:
            shader = context.shader_points
            shader.bind()
            # glDisable(GL_DEPTH_TEST)

            shader.setUniform('useRefPoint', np.int32(use_ref_point))
            shader.setUniform('cullOcclusions', np.int32(cull_occlusions))
            shader.setUniform('refPoint', np.array([ref_point[0], ref_point[1], ref_point[2]]).astype(np.float32))
            shader.setUniform('refRadius', np.float32(ref_radius))
            shader.setUniform("modelViewProj", camera.viewproj)
            shader.setUniform('far', np.float32(camera.far))
            shader.setUniform('near', np.float32(camera.near))
            shader.setUniform("screenSize", np.array(camera.resolution, dtype=np.float32) * context.resolution_scale)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, depth_map)
            glUniform1i(shader.uniform("renderedTexture"), 0)
            shader.uploadAttrib("position", self.points)

            if self.colors is not None:
                shader.uploadAttrib("color", self.colors)
            else:
                if type(color) == np.ndarray and color.shape[0] == self.points.shape[1]:
                    shader.uploadAttrib("color", color.T.astype(np.float32))
                else:
                    shader.uploadAttrib("color", np.tile(np.array([[color[0]], [color[1]], [color[2]]],
                                                                dtype=np.float32), [1, self.points.shape[1]]))
            shader.drawArray(gl.POINTS, 0, self.points.shape[1])
            glEnable(GL_DEPTH_TEST)
            return

        shader = context.shader
        shader.bind()

        if disable_ztest:
            glDisable(GL_DEPTH_TEST)
        shader.setUniform("modelViewProj", camera.viewproj)
        shader.uploadAttrib("position", self.points)
        if type(color) == np.ndarray and color.shape[0] == self.points.shape[1]:
            shader.uploadAttrib("color", color.T.astype(np.float32))
        else:
            shader.uploadAttrib("color", np.tile(np.array([[color[0]], [color[1]], [color[2]]],
                                                          dtype=np.float32), [1, self.points.shape[1]]))
        shader.drawArray(gl.POINTS, 0, self.points.shape[1])
        if disable_ztest:
            glEnable(GL_DEPTH_TEST)


class VoxelGrid:
    """Class representing a regular voxel grid"""

    def __init__(self, metadata_file, filename, voxel_data=None, vxbb_min=None, vxbb_max=None, normalize=True):

        if voxel_data is not None:
            self.voxel_grid_res = voxel_data.shape
            self.voxel_data = voxel_data[:, :, :, np.newaxis]
            self.vxbb_min = vxbb_min.astype(np.float32)
            self.vxbb_max = vxbb_max.astype(np.float32)

        else:
            self.voxel_grid_res, self.vxbb_min, self.vxbb_max, self.sigmaA, self.sigmaS = read_voxelgrid_metadata(
                metadata_file)

            data = np.genfromtxt(filename, dtype=np.float32)

            data3d = np.reshape(data, [self.voxel_grid_res[0], self.voxel_grid_res[1],
                                       self.voxel_grid_res[2]], order='F')
            data3d = np.swapaxes(data3d, 0, 1)
            self.voxel_data = data3d[:, :, :, np.newaxis]

        if normalize and np.max(self.voxel_data) > 0:
            self.voxel_data = self.voxel_data / np.max(self.voxel_data)

        self.texture = -1
        self.texture2 = -1
        self.cmap_img = get_color_map()

    def point_to_vox_idx(self, point):
        idx = ((point - self.vxbb_min) / (self.vxbb_max - self.vxbb_min) * self.voxel_grid_res).astype(int)
        return idx

    def draw_contents(self, camera, context):
        if self.texture < 0:
            self.texture = glGenTextures(1)
        if self.texture2 < 0:
            self.texture2 = glGenTextures(1)
        shader = context.shader_raymarch
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, -1]]).T.astype(float)
        indices = np.array(
            [[3, 3, 6, 3, 5, 7, 1, 4, 3, 4, 2, 5],
             [1, 2, 2, 7, 6, 4, 5, 0, 0, 7, 6, 1],
             [0, 1, 3, 6, 7, 5, 4, 1, 4, 3, 5, 2]],
            dtype=np.int32)

        positions = np.array(
            [[self.vxbb_min[0], self.vxbb_min[0], self.vxbb_max[0], self.vxbb_max[0],
              self.vxbb_min[0], self.vxbb_min[0], self.vxbb_max[0], self.vxbb_max[0]],
             [self.vxbb_max[1], self.vxbb_max[1], self.vxbb_max[1], self.vxbb_max[1],
              self.vxbb_min[1], self.vxbb_min[1], self.vxbb_min[1], self.vxbb_min[1]],
             [self.vxbb_max[2], self.vxbb_min[2], self.vxbb_min[2], self.vxbb_max[2],
              self.vxbb_max[2], self.vxbb_min[2], self.vxbb_min[2], self.vxbb_max[2]]],
            dtype=np.float32)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        shader.bind()

        glUniform1i(shader.uniform("tex"), 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.texture)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, self.voxel_data.shape[0], self.voxel_data.shape[0],
                     self.voxel_data.shape[0], 0, GL_RED, GL_FLOAT, self.voxel_data)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glUniform1i(shader.uniform("cmap"), 1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_1D, self.texture2)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, self.cmap_img.shape[0], 0,
                     GL_RGB, GL_FLOAT, self.cmap_img)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT)

        shader.setUniform("modelViewProj", camera.viewproj)
        shader.setUniform("cameraPos", camera.pos.astype(np.float32))
        shader.setUniform("bbMin", self.vxbb_min)
        shader.setUniform("bbMax", self.vxbb_max)
        res = np.array(self.voxel_data.shape, np.float32)[:3]
        shader.setUniform("voxelRes", res)

        shader.uploadAttrib("position", positions)
        shader.uploadIndices(indices)
        shader.drawIndexed(gl.TRIANGLES, 0, 12)


class VectorField:
    """Class representing a grid of normals """

    def __init__(self, metadata_file, filename, voxel_data=None, vxbb_min=None, vxbb_max=None, line_length=0.05, surf_voxels=None):
        if voxel_data is not None:
            self.voxel_grid_res = voxel_data.shape[0:3]
            self.vxbb_min = vxbb_min
            self.vxbb_max = vxbb_max
            data = np.reshape(voxel_data, [-1, 3])
            data4d = np.reshape(data, [self.voxel_grid_res[0], self.voxel_grid_res[1], self.voxel_grid_res[2], 3])
            # data4d = np.swapaxes(data4d, 0, 1)
            data = np.reshape(data4d, [np.prod(data4d.shape[:3]), 3])
        else:
            self.voxel_grid_res, self.vxbb_min, self.vxbb_max, self.sigmaA, self.sigmaS = read_voxelgrid_metadata(
                metadata_file)
            data = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
            data4d = np.reshape(data, [self.voxel_grid_res[0], self.voxel_grid_res[1], self.voxel_grid_res[2], 3])
            # data4d = np.swapaxes(data4d, 0, 1)
            data = np.reshape(data4d, [np.prod(data4d.shape[:3]), 3])

        diag = self.vxbb_max - self.vxbb_min

        xx, yy, zz = np.meshgrid(range(self.voxel_grid_res[0]), range(
            self.voxel_grid_res[1]), range(self.voxel_grid_res[2]), indexing='ij')
        voxel_positions_x = (xx + 0.5) / self.voxel_grid_res[0] * diag[0] + self.vxbb_min[0]
        voxel_positions_y = (yy + 0.5) / self.voxel_grid_res[1] * diag[1] + self.vxbb_min[1]
        voxel_positions_z = (zz + 0.5) / self.voxel_grid_res[2] * diag[2] + self.vxbb_min[2]
        voxel_positions_x = np.ravel(voxel_positions_x)
        voxel_positions_y = np.ravel(voxel_positions_y)
        voxel_positions_z = np.ravel(voxel_positions_z)

        self.voxel_positions = np.zeros([voxel_positions_x.shape[0] * 2, 3], dtype=np.float32)
        self.voxel_positions[0::2, :] = np.stack([voxel_positions_x, voxel_positions_y, voxel_positions_z], axis=1)

        voxel_offset_x = line_length * data[:, 0] + voxel_positions_x
        voxel_offset_y = line_length * data[:, 1] + voxel_positions_y
        voxel_offset_z = line_length * data[:, 2] + voxel_positions_z

        self.voxel_positions[1::2, :] = np.stack([voxel_offset_x, voxel_offset_y, voxel_offset_z], axis=1)
        self.voxel_positions = self.voxel_positions.T

    def draw_contents(self, camera, context):
        shader = context.shader
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        shader.bind()
        positions = np.array([[0, 0, 0], [0, 2, 0]])
        draw_lines(shader, np.array(self.voxel_positions, dtype=np.float32),
                   camera.viewproj, np.array([1, 0, 0], dtype=np.float32))

    def point_to_vox_idx(self, point):
        return ((point - self.vxbb_min) / (self.vxbb_max - self.vxbb_min) * self.voxel_grid_res).astype(int)


class VectorCloud:
    """Class representing a unorderd cloud of vectors, visualized as line segments """

    def __init__(self, origins, vectors, line_length=0.5, color=np.array([0, 0, 1])):
        origins = np.atleast_2d(origins)
        vectors = np.atleast_2d(vectors)
        p0 = origins
        p1 = origins + vectors * line_length
        self.positions = np.zeros((p0.shape[0] * 2, 3), dtype=np.float32)
        self.positions[0::2] = p0
        self.positions[1::2] = p1
        self.color = color.ravel().astype(np.float32)

    def draw_contents(self, camera, context):
        shader = context.shader
        shader.bind()
        # self.positions = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32).T
        draw_lines(shader, self.positions.T,
                   camera.viewproj, self.color)


class ScatterGrid:
    """Class representing a grid of scattering values"""

    def __init__(self, scatter_matrix):
        self.setup_rendering()

        self.scatter_matrix = scatter_matrix
        self.vxbb_min = self.scatter_matrix.bb_min
        self.vxbb_max = self.scatter_matrix.bb_max
        self.voxel_grid_res = self.scatter_matrix.voxel_grid_res
        self.scatter_matrix_data = self.scatter_matrix.matrix

        self.voxel_idx = [0, 0, 0]
        # self.recreate_voxel_grid()

    def setup_rendering(self):
        self.texture = glGenTextures(1)  # TODO: This should be done once we are sure that the OpenGL context exists
        self.texture2 = glGenTextures(1)

        # Set up color map
        self.cmap_img = get_color_map()

        glBindTexture(GL_TEXTURE_3D, self.texture)
        glBindTexture(GL_TEXTURE_1D, self.texture2)

    def draw_contents(self, camera, context, diff=None):
        shader = context.shader_raymarch
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, -1]]).T.astype(float)
        indices = np.array(
            [[3, 3, 6, 3, 5, 7, 1, 4, 3, 4, 2, 5],
             [1, 2, 2, 7, 6, 4, 5, 0, 0, 7, 6, 1],
             [0, 1, 3, 6, 7, 5, 4, 1, 4, 3, 5, 2]],
            dtype=np.int32)

        positions = np.array(
            [[self.vxbb_min[0], self.vxbb_min[0], self.vxbb_max[0], self.vxbb_max[0],
              self.vxbb_min[0], self.vxbb_min[0], self.vxbb_max[0], self.vxbb_max[0]],
             [self.vxbb_max[1], self.vxbb_max[1], self.vxbb_max[1], self.vxbb_max[1],
              self.vxbb_min[1], self.vxbb_min[1], self.vxbb_min[1], self.vxbb_min[1]],
             [self.vxbb_max[2], self.vxbb_min[2], self.vxbb_min[2], self.vxbb_max[2],
              self.vxbb_max[2], self.vxbb_min[2], self.vxbb_min[2], self.vxbb_max[2]]],
            dtype=np.float32)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        shader.bind()

        glUniform1i(shader.uniform("tex"), 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.texture)

        if diff is not None:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, self.voxel_data.shape[0], self.voxel_data.shape[0],
                         self.voxel_data.shape[0], 0, GL_RED, GL_FLOAT, (self.voxel_data - diff.voxel_data) ** 2)
        else:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, self.voxel_data.shape[0], self.voxel_data.shape[0],
                         self.voxel_data.shape[0], 0, GL_RED, GL_FLOAT, self.voxel_data)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glUniform1i(shader.uniform("cmap"), 1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_1D, self.texture2)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, self.cmap_img.shape[0], 0,
                     GL_RGB, GL_FLOAT, self.cmap_img)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT)

        shader.setUniform("modelViewProj", camera.viewproj)
        shader.setUniform("cameraPos", camera.pos.astype(np.float32))
        shader.setUniform("bbMin", self.vxbb_min)
        shader.setUniform("bbMax", self.vxbb_max)
        res = np.array(self.voxel_data.shape, np.float32)[:3]
        shader.setUniform("voxelRes", res)

        shader.uploadAttrib("position", positions)
        shader.uploadIndices(indices)
        shader.drawIndexed(gl.TRIANGLES, 0, 12)

    def point_to_vox_idx(self, point):
        idx = ((point - self.vxbb_min) / (self.vxbb_max - self.vxbb_min) * self.voxel_grid_res).astype(int)
        return idx

    def recreate_voxel_grid(self, area_term=None):
        linear_idx = self.scatter_matrix.idx_to_linear_idx(self.voxel_idx)
        print('Linear idx {}'.format(linear_idx))
        grid = self.scatter_matrix_data[linear_idx, :]

        grid = np.array(grid.todense())
        grid = np.reshape(grid, self.voxel_grid_res)

        self.voxel_data = grid[:, :, :, np.newaxis]
        voxres = self.voxel_data.shape[0:3]
        flat_area = 1 / voxres[0] * 1 / voxres[1]
        if area_term is not None:
            self.surf_voxel_data = self.voxel_data / np.maximum(flat_area / 100, area_term[:, :, :, np.newaxis])
        else:
            self.surf_voxel_data = self.voxel_data

        # self.voxel_data = np.log(grid[:, :, :, np.newaxis]) # Log transform
        max_val = np.max(self.voxel_data)
        if max_val > 0:
            self.voxel_data = self.voxel_data / max_val
        max_val = np.max(self.surf_voxel_data)
        if max_val > 0:
            self.surf_voxel_data = self.surf_voxel_data / max_val


class BinvoxGrid:
    """Class representing a simple voxelgrid"""

    def __init__(self, filename):
        self.texture = glGenTextures(1)
        self.texture2 = glGenTextures(1)

        # Set up color map
        self.cmap_img = get_color_map()

        glBindTexture(GL_TEXTURE_3D, self.texture)
        glBindTexture(GL_TEXTURE_1D, self.texture2)

        with open(filename, 'rb') as data_file:
            self.voxel_data = np.array(binvox.binvox_rw.read_as_3d_array(data_file).data).astype(np.float32)
            # self.voxel_data = np.flip(self.voxel_data, axis=0)
            self.voxel_data = np.swapaxes(self.voxel_data, 0, 2)

            self.voxel_grid_res = self.voxel_data.shape
            n_voxels = np.prod(self.voxel_grid_res)

            self.vxbb_min = np.array([-1, -1, -1]).astype(np.float32)
            self.vxbb_max = np.array([1, 1, 1]).astype(np.float32)

    def draw_contents(self, camera, shader):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, -1]]).T.astype(float)
        indices = np.array(
            [[3, 3, 6, 3, 5, 7, 1, 4, 3, 4, 2, 5],
             [1, 2, 2, 7, 6, 4, 5, 0, 0, 7, 6, 1],
             [0, 1, 3, 6, 7, 5, 4, 1, 4, 3, 5, 2]],
            dtype=np.int32)

        positions = np.array(
            [[self.vxbb_min[0], self.vxbb_min[0], self.vxbb_max[0], self.vxbb_max[0],
              self.vxbb_min[0], self.vxbb_min[0], self.vxbb_max[0], self.vxbb_max[0]],
             [self.vxbb_max[1], self.vxbb_max[1], self.vxbb_max[1], self.vxbb_max[1],
              self.vxbb_min[1], self.vxbb_min[1], self.vxbb_min[1], self.vxbb_min[1]],
             [self.vxbb_max[2], self.vxbb_min[2], self.vxbb_min[2], self.vxbb_max[2],
              self.vxbb_max[2], self.vxbb_min[2], self.vxbb_min[2], self.vxbb_max[2]]],
            dtype=np.float32)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        shader.bind()

        glUniform1i(shader.uniform("tex"), 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.texture)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, self.voxel_data.shape[0], self.voxel_data.shape[0],
                     self.voxel_data.shape[0], 0, GL_RED, GL_FLOAT, self.voxel_data)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glUniform1i(shader.uniform("cmap"), 1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_1D, self.texture2)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, self.cmap_img.shape[0], 0,
                     GL_RGB, GL_FLOAT, self.cmap_img)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT)

        shader.setUniform("modelViewProj", camera.viewproj)
        shader.setUniform("cameraPos", camera.pos.astype(np.float32))
        shader.setUniform("bbMin", self.vxbb_min)
        shader.setUniform("bbMax", self.vxbb_max)
        res = np.array(self.voxel_data.shape, np.float32)[:3]
        shader.setUniform("voxelRes", res)

        shader.uploadAttrib("position", positions)
        shader.uploadIndices(indices)
        shader.drawIndexed(gl.TRIANGLES, 0, 12)
