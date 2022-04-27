import os
import pathlib
import random

import moderngl as mgl
import moderngl_window as mglw
import numpy as np
import OpenGL.GL as gl

render_vert = '''
#version 460

in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
'''
render_frag = '''
#version 460

uniform sampler2D render_texture;
uniform vec2 screen_size;
out vec4 frag_color;

void main()
{
    frag_color = texture2D(render_texture, gl_FragCoord.xy / screen_size);
}
'''

class FileWatch():
    def __init__(self, path, on_changed):
        self.stamp = 0
        self.path = path
        self.on_changed = on_changed

    def check(self):
        stamp = os.stat(self.path).st_mtime
        if stamp != self.stamp:
            self.stamp = stamp
            self.on_changed()

class ParticleAgents(mglw.WindowConfig):
    gl_version = (4,6)
    title = "particle agents"
    window_size = (1400, 1000)
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame = 0
        self.work_group_size = 32
        self.data_count = 2000000

        self.agent_compute_program = None
        self.field_compute_program = None
        self.reload_agent_shader()
        self.reload_field_shader()
        self.file_watches = []
        self.file_watches.append(FileWatch('agent.comp', self.reload_agent_shader))
        self.file_watches.append(FileWatch('field.comp', self.reload_field_shader))
        self.render_program = self.ctx.program(vertex_shader=render_vert, fragment_shader=render_frag)

        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('dir', np.float32), ('id', np.int32)])
        data_size = dtype.itemsize * self.data_count
        tau = np.pi * 2.0

        agents = np.zeros(self.data_count, dtype=dtype)
        with np.nditer(agents, op_flags=['writeonly']) as it:
            i = 0
            for agent in it:
                agent['x'] = random.uniform(0, self.window_size[0])
                agent['y'] = random.uniform(0, self.window_size[1])
                agent['dir'] = random.uniform(0, tau)
                agent['id'] = i % 3
                i += 1
        self.agent_buffer = self.ctx.buffer(agents)
        del agents

        def make_tex():
            tex = self.ctx.texture(size=self.window_size, components=4, dtype='f4')
            tex.filter = (mgl.NEAREST, mgl.NEAREST)
            tex.repeat_x = True
            tex.repeat_y = True
            return tex

        self.in_texture = make_tex()
        self.out_texture = make_tex()

        self.render_buffer = self.ctx.buffer(np.array([[-1, -1], [3, -1], [-1, 3]], dtype=np.float32))
        self.render_vao = self.ctx.vertex_array(self.render_program, self.render_buffer, 'position')

    def reload_comp_shader(self, prog, path):
        try:
            print(path)
            new_prog = self.ctx.compute_shader(pathlib.Path(path).read_text())
            if prog:
                prog.release()
            return new_prog
        except Exception as e:
            print(e)
            return prog

    def reload_agent_shader(self):
        self.agent_compute_program = self.reload_comp_shader(self.agent_compute_program, 'agent.comp')

    def reload_field_shader(self):
        self.field_compute_program = self.reload_comp_shader(self.field_compute_program, 'field.comp')

    def render(self, time, frametime):
        for watch in self.file_watches:
            watch.check()

        self.ctx.viewport = (0, 0, *self.window_size)

        gl.glCopyImageSubData(self.out_texture.glo, gl.GL_TEXTURE_2D, 0, 0, 0, 0, self.in_texture.glo, gl.GL_TEXTURE_2D, 0, 0, 0, 0, self.window_size[0], self.window_size[1], 1)

        # Run agent computation
        self.agent_buffer.bind_to_storage_buffer()
        self.in_texture.bind_to_image(1, read=True, write=False)
        self.out_texture.bind_to_image(2, read=False, write=True)
        self.agent_compute_program['frame'] = self.frame
        self.agent_compute_program.run(int(self.data_count / self.work_group_size))
        gl.glMemoryBarrier(gl.GL_ALL_BARRIER_BITS)

        (self.in_texture, self.out_texture) = (self.out_texture, self.in_texture)

        # Process texture
        self.in_texture.bind_to_image(1, read=True, write=False)
        self.out_texture.bind_to_image(2, read=False, write=True)
        self.field_compute_program.run(
            int(self.window_size[0] / self.work_group_size + self.work_group_size),
            int(self.window_size[1] / self.work_group_size + self.work_group_size))
        gl.glMemoryBarrier(gl.GL_ALL_BARRIER_BITS)

        # Render result
        self.render_program['render_texture'] = 0
        self.render_program['screen_size'] = self.window_size
        self.out_texture.use()
        self.render_vao.render()

        self.frame += 1

if __name__ == '__main__':
    mglw.run_window_config(ParticleAgents)
