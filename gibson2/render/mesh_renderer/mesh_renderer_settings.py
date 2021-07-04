import os
import gibson2
import platform


class MeshRendererSettings(object):
    def __init__(
        self,
        use_fisheye=False,
        msaa=False,
        enable_shadow=False,
        enable_pbr=True,
        env_texture_filename=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background',
                                          'photo_studio_01_2k.hdr'),
        env_texture_filename2=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background',
                                           'photo_studio_01_2k.hdr'),
        env_texture_filename3=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background',
                                           'photo_studio_01_2k.hdr'),
        light_modulation_map_filename='',
        optimized=False,
        skybox_size=20.,
        light_dimming_factor=1.0,
        fullscreen=False,
        glfw_gl_version=None,
        texture_scale=1.0,
        hide_robot=True,
        is_robosuite=False
    ):
        """
        :param use_fisheye: whether to use fisheye camera
        :param msaa: whether to msaa
        :param enable_shadow: whether to enable shadow
        :param enable_pbr: whether to enable pbr
        :param env_texture_filename: the first light probe
        :param env_texture_filename2: the second light probe
        :param env_texture_filename3: the third light probe
        :param light_modulation_map_filename: light modulation map filename
        :param optimized: whether to use optimized renderer (quality can be slightly compromised)
        :param skybox_size: size of the outdoor skybox
        :param light_dimming_factor: light dimming factor
        :param fullscreen: whether to use full screen
        :param glfw_gl_version: glfw gl version
        :param texture_scale: texture scale
        :param hide_robot: whether to hide robot when rendering
        :param is_robosuite: whether the environment is of robosuite.
        """
        self.use_fisheye = use_fisheye
        self.msaa = msaa
        self.enable_shadow = enable_shadow
        self.env_texture_filename = env_texture_filename
        self.env_texture_filename2 = env_texture_filename2
        self.env_texture_filename3 = env_texture_filename3
        self.optimized = optimized
        self.skybox_size = skybox_size
        self.light_modulation_map_filename = light_modulation_map_filename
        self.light_dimming_factor = light_dimming_factor
        self.enable_pbr = enable_pbr
        self.fullscreen = fullscreen
        self.texture_scale = texture_scale
        self.hide_robot = hide_robot
        self.is_robosuite = is_robosuite

        if glfw_gl_version is not None:
            self.glfw_gl_version = glfw_gl_version
        else:
            if platform.system() == 'Darwin':
                self.glfw_gl_version = [4, 1]
            else:
                self.glfw_gl_version = [4, 5]

    def get_fastest(self):
        self.msaa = False
        self.enable_shadow = False
        return self

    def get_best(self):
        self.msaa = True
        self.enable_shadow = True
        return self
