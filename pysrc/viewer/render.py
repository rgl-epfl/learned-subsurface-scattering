import matplotlib.pyplot as plt
import numpy as np
import os

from OpenGL.GL import *
from OpenGL.GLU import *

from nanogui import gl


def load_shader(shader, basename, description):
    """Loads a shader from a .frag and .vert file"""
    with open(basename + '.vert', 'r') as shader_file:
        vshader = shader_file.read()
    with open(basename + '.frag', 'r') as shader_file:
        fshader = shader_file.read()
    geomname = basename + '.geom'
    gshader = ""
    if os.path.isfile(geomname):
        with open(geomname, 'r') as shader_file:
            gshader = shader_file.read()
    shader.init(description, vshader, fshader, gshader)
    shader.bind()


def get_color_map():
    """Returns 1D image of matplotlib default color map."""
    vals = np.arange(0, 1, 1 / 128)
    img_artist = plt.imshow(vals[:, np.newaxis])
    cmapped = img_artist.cmap(img_artist.norm(vals))
    plt.close()
    return cmapped[:, :3].astype(np.float32)


def draw_grid(shader, vp_mat):
    """Draws a simple grid plane in the viewport."""
    grid_extent = 5
    grid_step = 0.5
    r = np.arange(-grid_extent, grid_extent + 1e-2, grid_step, dtype=np.float32)
    pos0 = np.zeros([3, r.shape[0] * 2], dtype=np.float32)
    pos0[:, 0::2] = np.stack([r, np.zeros_like(r), -grid_extent * np.ones_like(r)])
    pos0[:, 1::2] = np.stack([r, np.zeros_like(r), grid_extent * np.ones_like(r)])
    pos1 = np.zeros([3, r.shape[0] * 2], dtype=np.float32)
    pos1[:, 0::2] = np.stack([-grid_extent * np.ones_like(r), np.zeros_like(r), r])
    pos1[:, 1::2] = np.stack([grid_extent * np.ones_like(r), np.zeros_like(r), r])
    colors = 0.5 * np.ones(pos0.shape, dtype=np.float32)
    colors[:, colors.shape[1] // 2] *= 0
    colors[:, colors.shape[1] // 2 - 1] *= 0

    shader.setUniform("modelViewProj", vp_mat)
    shader.uploadAttrib("position", pos0)
    shader.uploadAttrib("color", colors)
    shader.drawArray(gl.LINES, 0, pos0.shape[1])
    shader.setUniform("modelViewProj", vp_mat)
    shader.uploadAttrib("position", pos1)
    shader.uploadAttrib("color", colors)
    shader.drawArray(gl.LINES, 0, pos1.shape[1])

    # Draw axis on top
    gl.Disable(gl.DEPTH_TEST)
    pos0 = np.zeros([3, 6], dtype=np.float32)
    pos0[0, 1] = grid_extent
    pos0[1, 3] = grid_extent
    pos0[2, 5] = grid_extent
    colors = np.ones(pos0.shape, dtype=np.float32)
    colors[1:, 0:2] *= 0
    colors[2, 2:4] *= 0
    colors[0, 2:4] *= 0
    colors[0:2, 4:6] *= 0
    shader.setUniform("modelViewProj", vp_mat)
    shader.uploadAttrib("position", pos0)
    shader.uploadAttrib("color", colors)
    shader.drawArray(gl.LINES, 0, pos0.shape[1])
    gl.Enable(gl.DEPTH_TEST)

def draw_full_screen_quad(shader, vp_mat, depth_map):
    """Draws a simple grid plane in the viewport."""

    tex = depth_map
    glBindTexture(GL_TEXTURE_2D, tex)
    glUniform1i(shader.uniform("renderedTexture"), 0)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)

    gl.Disable(gl.DEPTH_TEST)
    pos0 = np.array([[-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0]], dtype=np.float32).T
    uv = pos0[:2, :] * 0.5 + 0.5
    pos0 *= 0.5
    shader.uploadAttrib("position", pos0)
    shader.uploadAttrib("uv", uv)
    shader.drawArray(gl.TRIANGLES, 0, pos0.shape[1])
    gl.Enable(gl.DEPTH_TEST)



def draw_lines(shader, positions, vp_mat, color, style='-'):
    """Draw given line segment vertices"""

    if positions is None:
        return

    if style == '--':
        shader.bind()
        shader.setUniform("modelViewProj", vp_mat)
        shader.uploadAttrib("position", positions)
        uv_values = np.zeros([2, positions.shape[1]], dtype=np.float32)
        lens = np.sqrt(np.sum((positions[:, 0::2] - positions[:, 1::2]) ** 2, axis=0))
        uv_values[0, 1::2] = lens
        uv_values[1, :] = np.repeat(lens[:, np.newaxis].T, 2, axis=1)
        shader.uploadAttrib("uv", uv_values)
        shader.uploadAttrib("color", np.tile(np.array([[color[0]], [color[1]], [color[2]]],
                                                      dtype=np.float32), [1, positions.shape[1]]))
        shader.drawArray(gl.LINES, 0, positions.shape[1])
    else:
        shader.bind()
        shader.setUniform("modelViewProj", vp_mat)
        shader.uploadAttrib("position", positions)
        shader.uploadAttrib("color", np.tile(np.array([[color[0]], [color[1]], [color[2]]],
                                                      dtype=np.float32), [1, positions.shape[1]]))
        shader.drawArray(gl.LINES, 0, positions.shape[1])


def draw_box(shader, p0, p1, vp_mat, color):
    shader.bind()
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
    draw_lines(shader, np.array(positions, dtype=np.float32), vp_mat, color)


def get_view_ray_dir(position, camera):
    """Given screen coordinate, computes the viewing ray world space direction"""
    ndc = 2 * position / camera.resolution - 1
    clip_coords = np.array([[ndc[0]], [-ndc[1]], [1], [1]])
    world_coords = np.linalg.inv(camera.viewproj) * clip_coords
    world_coords = world_coords[0:3] / world_coords[3]
    world_coords = np.array(world_coords)
    ray_dir = -camera.pos + np.ravel(world_coords)
    ray_dir = ray_dir / np.sqrt(np.sum(ray_dir ** 2))
    return ray_dir
