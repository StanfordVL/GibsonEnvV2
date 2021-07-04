import logging
import platform
from gibson2.render.mesh_renderer import tinyobjloader
import gibson2
import gibson2.render.mesh_renderer as mesh_renderer
from gibson2.render.mesh_renderer.get_available_devices import get_available_devices
from gibson2.utils.mesh_util import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, \
    safemat2quat, xyzw2wxyz, ortho, transform_vertex
from gibson2.utils.constants import AVAILABLE_MODALITIES, ShadowPass
import numpy as np
import os
import sys
from gibson2.render.mesh_renderer.materials import Material, RandomizedMaterial
from gibson2.render.mesh_renderer.instances import Instance, InstanceGroup, Robot
from gibson2.render.mesh_renderer.visual_object import VisualObject
from PIL import Image
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
Image.MAX_IMAGE_PIXELS = None


class MeshRenderer(object):
    """
    MeshRenderer is a lightweight OpenGL renderer.
    It manages a set of visual objects, and instances of those objects.
    It also manage a device to create OpenGL context on, and create buffers to store rendering results.
    """

    def __init__(self, width=512, height=512, vertical_fov=90, device_idx=0, rendering_settings=MeshRendererSettings()):
        """
        :param width: width of the renderer output
        :param height: width of the renderer output
        :param vertical_fov: vertical field of view for the renderer
        :param device_idx: which GPU to run the renderer on
        :param render_settings: rendering settings
        """
        self.rendering_settings = rendering_settings
        self.shaderProgram = None
        self.windowShaderProgram = None
        self.fbo = None
        self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d = None, None, None, None
        self.color_tex_scene_flow, self.color_tex_optical_flow = None, None
        self.depth_tex = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []
        self.visual_objects = []
        self.vertex_data = []
        self.shapes = []
        self.width = width
        self.height = height
        self.faces = []
        self.instances = []
        self.fisheye = rendering_settings.use_fisheye
        self.optimized = rendering_settings.optimized
        self.texture_files = {}
        self.enable_shadow = rendering_settings.enable_shadow
        self.platform = platform.system()
        self.optimization_process_executed = False
        self.pose_trans_array = None
        self.pose_rot_array = None
        self.last_trans_array = None
        self.last_rot_array = None

        device = None
        """
        device_idx is the major id
        device is the minor id
        you can get it from nvidia-smi -a

        The minor number for the device is such that the Nvidia device node file for each GPU will have the form
        /dev/nvidia[minor number]. Available only on Linux platform.

        TODO: add device management for windows platform.
        """

        if os.environ.get('GIBSON_DEVICE_ID', None):
            device = int(os.environ.get('GIBSON_DEVICE_ID'))
            logging.info("GIBSON_DEVICE_ID environment variable has been manually set. "
                         "Using device {} for rendering".format(device))
        else:
            if self.platform != 'Windows':
                available_devices = get_available_devices()
                if device_idx < len(available_devices):
                    device = available_devices[device_idx]
                    logging.info(
                        "Using device {} for rendering".format(device))
                else:
                    logging.info(
                        "Device index is larger than number of devices, falling back to use 0")
                    logging.info(
                        "If you have trouble using EGL, please visit our trouble shooting guide",
                        "at http://svl.stanford.edu/igibson/docs/issues.html")

                    device = 0

        self.device_idx = device_idx
        self.device_minor = device
        self.msaa = rendering_settings.msaa
        if self.platform == 'Darwin' and self.optimized:
            logging.error('Optimized renderer is not supported on Mac')
            exit()
        if self.platform == 'Darwin':
            from gibson2.render.mesh_renderer import GLFWRendererContext
            self.r = GLFWRendererContext.GLFWRendererContext(
                width, height,
                int(self.rendering_settings.glfw_gl_version[0]),
                int(self.rendering_settings.glfw_gl_version[1]),
                False,
                rendering_settings.fullscreen
            )
        elif self.platform == 'Windows':
            from gibson2.render.mesh_renderer import VRRendererContext
            self.r = VRRendererContext.VRRendererContext(
                width, height,
                int(self.rendering_settings.glfw_gl_version[0]),
                int(self.rendering_settings.glfw_gl_version[1]),
                True,
                rendering_settings.fullscreen
            )
        else:
            from gibson2.render.mesh_renderer import EGLRendererContext
            self.r = EGLRendererContext.EGLRendererContext(
                width, height, device)

        self.r.init()

        self.glstring = self.r.getstring_meshrenderer()

        logging.debug('Rendering device and GL version')
        logging.debug(self.glstring)

        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.lightcolor = [1, 1, 1]

        logging.debug('Is using fisheye camera: {}'.format(self.fisheye))

        if self.fisheye:
            logging.error('Fisheye is currently not supported.')
            exit(1)
        else:
            if self.platform == 'Darwin':
                self.shaderProgram = self.r.compile_shader_meshrenderer(
                    "".join(open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders', '410', 'vert.shader')).readlines()),
                    "".join(open(
                        os.path.join(os.path.dirname(mesh_renderer.__file__),
                                     'shaders', '410', 'frag.shader')).readlines()))
            else:
                if self.optimized:
                    self.shaderProgram = self.r.compile_shader_meshrenderer(
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                         'shaders', '450', 'optimized_vert.shader')).readlines()),
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                         'shaders', '450', 'optimized_frag.shader')).readlines()))
                else:
                    self.shaderProgram = self.r.compile_shader_meshrenderer(
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                         'shaders', '450', 'vert.shader')).readlines()),
                        "".join(open(
                            os.path.join(os.path.dirname(mesh_renderer.__file__),
                                         'shaders', '450', 'frag.shader')).readlines()))

            self.skyboxShaderProgram = self.r.compile_shader_meshrenderer(
                "".join(open(
                    os.path.join(os.path.dirname(mesh_renderer.__file__),
                                 'shaders', '410', 'skybox_vs.glsl')).readlines()),
                "".join(open(
                    os.path.join(os.path.dirname(mesh_renderer.__file__),
                                 'shaders', '410', 'skybox_fs.glsl')).readlines()))

        # default light looking down and tilted
        self.set_light_position_direction([0, 0, 2], [0, 0.5, 0])

        self.setup_framebuffer()
        self.vertical_fov = vertical_fov
        self.horizontal_fov = 2 * np.arctan(np.tan(self.vertical_fov / 180.0 * np.pi / 2.0) * self.width /
                                            self.height) / np.pi * 180.0

        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        self.znear = 0.1
        self.zfar = 100
        P = perspective(self.vertical_fov, float(
            self.width) / float(self.height), self.znear, self.zfar)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.last_V = np.copy(self.V)
        self.cache = np.copy(self.V)

        self.P = np.ascontiguousarray(P, np.float32)
        self.materials_mapping = {}
        self.mesh_materials = []
        # Number of unique shapes comprising the optimized renderer buffer
        self.or_buffer_shape_num = 0
        # Store trans and rot data for OR as a single variable that we update every frame - avoids copying variable each frame
        self.trans_data = None
        self.rot_data = None

        self.skybox_size = rendering_settings.skybox_size
        if not self.platform == 'Darwin' and rendering_settings.enable_pbr:
            self.setup_pbr()

        self.setup_lidar_param()

    def setup_pbr(self):
        """
        Set up physics-based rendering
        """
        if os.path.exists(self.rendering_settings.env_texture_filename) or \
                os.path.exists(self.rendering_settings.env_texture_filename2) or \
                os.path.exists(self.rendering_settings.env_texture_filename3):
            self.r.setup_pbr(os.path.join(os.path.dirname(mesh_renderer.__file__), 'shaders', '450'),
                             self.rendering_settings.env_texture_filename,
                             self.rendering_settings.env_texture_filename2,
                             self.rendering_settings.env_texture_filename3,
                             self.rendering_settings.light_modulation_map_filename,
                             self.rendering_settings.light_dimming_factor
                             )
        else:
            logging.warning(
                "Environment texture not available, cannot use PBR.")
        if self.rendering_settings.enable_pbr:
            self.r.loadSkyBox(self.skyboxShaderProgram, self.skybox_size)

    def set_light_position_direction(self, position, target):
        """
        Set light position and orientation

        :param position: light position
        :param target: light target
        """
        self.lightpos = position
        self.lightV = lookat(self.lightpos, target, [0, 1, 0])
        # If lightP is already set by an outside entity it will not override it.
        if not hasattr(self, 'lightP'):
            self.lightP = ortho(-5, 5, -5, 5, -10, 20.0)

    def setup_framebuffer(self):
        """
        Set up framebuffers for the renderer
        """
        [self.fbo, self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d,
         self.color_tex_scene_flow, self.color_tex_optical_flow,
         self.depth_tex] = self.r.setup_framebuffer_meshrenderer(self.width, self.height)
        if self.msaa:
            [self.fbo_ms, self.color_tex_rgb_ms, self.color_tex_normal_ms, self.color_tex_semantics_ms,
             self.color_tex_3d_ms, self.color_tex_scene_flow_ms, self.color_tex_optical_flow_ms,
             self.depth_tex_ms] = self.r.setup_framebuffer_meshrenderer_ms(self.width, self.height)

        self.depth_tex_shadow = self.r.allocateTexture(self.width, self.height)

    def load_texture_file(self, tex_filename):
        """
        Load the texture file into the renderer

        :param tex_filename: texture file filename
        :return texture_id: texture id of this texture in the renderer
        """
        # if texture is None or does not exist, return None
        if tex_filename is None or (not os.path.isfile(tex_filename)):
            return None

        # if texture already exists, return texture id
        if tex_filename in self.texture_files:
            return self.texture_files[tex_filename]

        if self.optimized:
            # assume optimized renderer will have texture id starting from 0
            texture_id = len(self.texture_files)
        else:
            texture_id = self.r.loadTexture(
                tex_filename, self.rendering_settings.texture_scale)
            self.textures.append(texture_id)

        self.texture_files[tex_filename] = texture_id
        return texture_id

    def load_randomized_material(self, material):
        """
        Load all the texture files in the RandomizedMaterial.
        Populate material_ids with the texture id assigned by the renderer.

        :param material: an instance of RandomizedMaterial
        """
        # if the material has already been initialized
        if material.material_ids is not None:
            return
        material.material_ids = {}
        for material_class in material.material_files:
            if material_class not in material.material_ids:
                material.material_ids[material_class] = []
            for material_instance in material.material_files[material_class]:
                material_id_instance = {}
                for key in material_instance:
                    material_id_instance[key] = \
                        self.load_texture_file(material_instance[key])
                material.material_ids[material_class].append(
                    material_id_instance)
        material.randomize()

    def load_object(self,
                    obj_path,
                    scale=np.array([1, 1, 1]),
                    transform_orn=None,
                    transform_pos=None,
                    input_kd=None,
                    texture_scale=1.0,
                    load_texture=True,
                    overwrite_material=None):
        """
        Load a wavefront obj file into the renderer and create a VisualObject to manage it.

        :param obj_path: path of obj file
        :param scale: scale, default 1
        :param transform_orn: rotation quaternion, convention xyzw
        :param transform_pos: translation for loading, it is a list of length 3
        :param input_kd: if loading material fails, use this default material. input_kd should be a list of length 3
        :param texture_scale: texture scale for the object, downsample to save memory.
        :param load_texture: load texture or not
        :param overwrite_material: whether to overwrite the default Material (usually with a RandomizedMaterial for material randomization)
        :return: VAO_ids
        """
        if self.optimization_process_executed and self.optimized:
            logging.error("Using optimized renderer and optimization process is already excuted, cannot add new "
                          "objects")
            return

        reader = tinyobjloader.ObjReader()
        logging.info("Loading {}".format(obj_path))
        ret = reader.ParseFromFile(obj_path)
        vertex_data_indices = []
        face_indices = []
        if not ret:
            logging.error("Warning: {}".format(reader.Warning()))
            logging.error("Error: {}".format(reader.Error()))
            logging.error("Failed to load: {}".format(obj_path))
            sys.exit(-1)

        if reader.Warning():
            logging.warning("Warning: {}".format(reader.Warning()))

        attrib = reader.GetAttrib()
        logging.debug("Num vertices = {}".format(len(attrib.vertices)))
        logging.debug("Num normals = {}".format(len(attrib.normals)))
        logging.debug("Num texcoords = {}".format(len(attrib.texcoords)))

        materials = reader.GetMaterials()
        logging.debug("Num materials: {}".format(len(materials)))

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for m in materials:
                logging.debug("Material name: {}".format(m.name))
                logging.debug("Material diffuse: {}".format(m.diffuse))

        shapes = reader.GetShapes()
        logging.debug("Num shapes: {}".format(len(shapes)))

        material_count = len(self.materials_mapping)
        if overwrite_material is not None and len(materials) > 1:
            logging.warning(
                "passed in one material ends up overwriting multiple materials")

        # set the default values of variable before being modified later.
        num_existing_mats = len(self.materials_mapping)    # Number of current Material elements 
        texuniform = False
        num_added_materials = 0        

        # Deparse the materials in the obj file by loading textures into the renderer's memory and creating a Material element for them
        for i, item in enumerate(materials):
            if overwrite_material is not None:
                self.load_randomized_material(overwrite_material)
                material = overwrite_material
            elif item.diffuse_texname != '' and load_texture:
                obj_dir = os.path.dirname(obj_path)
                texture = self.load_texture_file(
                    os.path.join(obj_dir, item.diffuse_texname))
                texture_metallic = self.load_texture_file(
                    os.path.join(obj_dir, item.metallic_texname))
                texture_roughness = self.load_texture_file(
                    os.path.join(obj_dir, item.roughness_texname))
                texture_normal = self.load_texture_file(
                    os.path.join(obj_dir, item.bump_texname))
                material = Material('texture',
                                    texture_id=texture,
                                    metallic_texture_id=texture_metallic,
                                    roughness_texture_id=texture_roughness,
                                    normal_texture_id=texture_normal)
            else:
                if input_kd is not None and len(input_kd) == 4 and input_kd[3] != 1:
                    # Pink color for translucent objects.
                    material = Material('color', kd=[1,0,1,1])
                else:
                    material = Material('color', kd=item.diffuse)
            self.materials_mapping[i + material_count] = material
            num_added_materials = len(materials)

        # Case when mesh obj is without mtl file but overwrite material is specified.
        if len(materials) == 0 and overwrite_material is not None:
            texuniform = overwrite_material.texuniform
            self.materials_mapping[num_existing_mats] = overwrite_material
            num_added_materials = 1
        
        if input_kd is not None:  # append the default material in the end, in case material loading fails
            self.materials_mapping[num_existing_mats + num_added_materials] = Material('color', kd=input_kd)
        else:
            self.materials_mapping[num_existing_mats + num_added_materials] = Material('color', kd=[0.5, 0.5, 0.5])

        VAO_ids = []

        vertex_position = np.array(attrib.vertices).reshape(
            (len(attrib.vertices) // 3, 3))
        vertex_normal = np.array(attrib.normals).reshape(
            (len(attrib.normals) // 3, 3))
        vertex_texcoord = np.array(attrib.texcoords).reshape(
            (len(attrib.texcoords) // 2, 2))

        if texuniform and overwrite_material.texture_type == "2d":
            # if any material is used multiple times transform param gets overridden.
            # Therefore original value of transform param is stored during initialization
            # from robosuite ans is stored in `_orig_transform_param`
            repeat_x = overwrite_material._orig_transform_param[0] * (np.max(vertex_texcoord[:,0]) - np.min(vertex_texcoord[:,0])) * scale[0]
            repeat_y = overwrite_material._orig_transform_param[1] * (np.max(vertex_texcoord[:,1]) - np.min(vertex_texcoord[:,1])) * scale[1]
            overwrite_material.transform_param = [repeat_x, repeat_y, 0.] 

        for shape in shapes:
            logging.debug("Shape name: {}".format(shape.name))
            if overwrite_material is None:
                # assume one shape only has one material
                material_id = shape.mesh.material_ids[0]
            else:
                # This will makes us use the input_material at self.materials_mapping[num_existing_mats + material_id=0]
                material_id = 0             
            
            logging.debug("material_id = {}".format(material_id))
            logging.debug("num_indices = {}".format(len(shape.mesh.indices)))
            n_indices = len(shape.mesh.indices)
            np_indices = shape.mesh.numpy_indices().reshape((n_indices, 3))

            shape_vertex_index = np_indices[:, 0]
            shape_normal_index = np_indices[:, 1]
            shape_texcoord_index = np_indices[:, 2]
            shape_vertex = vertex_position[shape_vertex_index]

            if len(vertex_normal) == 0:
                # dummy normal if normal is not available
                shape_normal = np.zeros((shape_vertex.shape[0], 3))
            else:
                shape_normal = vertex_normal[shape_normal_index]

            # Scale the shape before transforming
            # Need to flip normals in axes where we have negative scaling
            for i in range(3):
                shape_vertex[:, i] *= scale[i]
                if scale[i] < 0:
                    shape_normal[:, i] *= -1

            if len(vertex_texcoord) == 0:
                # dummy texcoord if texcoord is not available
                shape_texcoord = np.zeros((shape_vertex.shape[0], 2))
            else:
                shape_texcoord = vertex_texcoord[shape_texcoord_index]

            if transform_orn is not None:
                # Rotate the shape after they are scaled
                orn = quat2rotmat(xyzw2wxyz(transform_orn))
                shape_vertex = shape_vertex.dot(orn[:3, :3].T)
                # Also rotate the surface normal, note that tangent space does not need to be rotated since they
                # are derived from shape_vertex
                shape_normal = shape_normal.dot(orn[:3, :3].T)
            if transform_pos is not None:
                # Translate the shape after they are scaled
                shape_vertex += np.array(transform_pos)

            v0 = shape_vertex[0::3, :]
            v1 = shape_vertex[1::3, :]
            v2 = shape_vertex[2::3, :]
            uv0 = shape_texcoord[0::3, :]
            uv1 = shape_texcoord[1::3, :]
            uv2 = shape_texcoord[2::3, :]

            delta_pos1 = v1 - v0
            delta_pos2 = v2 - v0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0
            r = 1.0 / (delta_uv1[:, 0] * delta_uv2[:, 1] -
                       delta_uv1[:, 1] * delta_uv2[:, 0])
            tangent = (delta_pos1 * delta_uv2[:, 1][:, None] -
                       delta_pos2 * delta_uv1[:, 1][:, None]) * r[:, None]
            bitangent = (delta_pos2 * delta_uv1[:, 0][:, None] -
                         delta_pos1 * delta_uv2[:, 0][:, None]) * r[:, None]
            bitangent = bitangent.repeat(3, axis=0)
            tangent = tangent.repeat(3, axis=0)
            vertices = np.concatenate(
                [shape_vertex, shape_normal, shape_texcoord, tangent, bitangent], axis=-1)
            faces = np.array(range(len(vertices))).reshape(
                (len(vertices) // 3, 3))
            vertexData = vertices.astype(np.float32)
            [VAO, VBO] = self.r.load_object_meshrenderer(
                self.shaderProgram, vertexData)
            self.VAOs.append(VAO)
            self.VBOs.append(VBO)
            face_indices.append(len(self.faces))
            self.faces.append(faces)
            self.objects.append(obj_path)
            vertex_data_indices.append(len(self.vertex_data))
            self.vertex_data.append(vertexData)
            self.shapes.append(shape)
            # if material loading fails, use the default material
            if material_id == -1:
                self.mesh_materials.append(len(materials) + material_count)
            else:
                self.mesh_materials.append(material_id + material_count)

            logging.debug('mesh_materials: {}'.format(self.mesh_materials))
            VAO_ids.append(self.get_num_objects() - 1)

        new_obj = VisualObject(
            obj_path, VAO_ids=VAO_ids, vertex_data_indices=vertex_data_indices, face_indices=face_indices,
            id=len(self.visual_objects), renderer=self)
        self.visual_objects.append(new_obj)
        return VAO_ids

    def add_instance(self,
                     object_id,
                     pybullet_uuid=None,
                     class_id=0,
                     pose_trans=np.eye(4),
                     pose_rot=np.eye(4),
                     dynamic=False,
                     softbody=False,
                     use_pbr=True,
                     use_pbr_mapping=True,
                     shadow_caster=True,
                     parent_body=None):
        """
        Create instance for a visual object and link it to pybullet

        :param object_id: id of visual object
        :param pybullet_uuid: body id in pybullet
        :param class_id: class_id to render semantics
        :param pose_trans: initial translations for the visual object
        :param pose_rot: initial rotation matrix for the visual object
        :param dynamic: whether the instance is dynamic
        :param softbody: whether the instance is soft body
        :param use_pbr: whether to use PBR
        :param use_pbr_mapping: whether to use PBR mapping
        :param shadow_caster: whether to cast shadow
        :param parent_body: parent body name of current xml element (MuJoCo XML)
        """
        if self.optimization_process_executed and self.optimized:
            logging.error("Using optimized renderer and optimization process is already excuted, cannot add new "
                          "objects")
            return

        use_pbr = use_pbr and self.rendering_settings.enable_pbr
        use_pbr_mapping = use_pbr_mapping and self.rendering_settings.enable_pbr

        instance = Instance(self.visual_objects[object_id],
                            id=len(self.instances),
                            pybullet_uuid=pybullet_uuid,
                            class_id=class_id,
                            pose_trans=pose_trans,
                            pose_rot=pose_rot,
                            dynamic=dynamic,
                            softbody=softbody,
                            use_pbr=use_pbr,
                            use_pbr_mapping=use_pbr_mapping,
                            shadow_caster=shadow_caster,
                            parent_body=parent_body)
        self.instances.append(instance)

    def add_instance_group(self,
                           object_ids,
                           link_ids,
                           poses_trans,
                           poses_rot,
                           pybullet_uuid=None,
                           class_id=0,
                           dynamic=False,
                           robot=None,
                           use_pbr=True,
                           use_pbr_mapping=True,
                           shadow_caster=True):
        """
        Create an instance group for a list of visual objects and link it to pybullet

        :param object_ids: object ids of the visual objects
        :param link_ids: link_ids in pybullet
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param pybullet_uuid: body id in pybullet
        :param class_id: class_id to render semantics
        :param dynamic: whether the instance group is dynamic
        :param robot: The robot associated with this InstanceGroup
        :param use_pbr: whether to use PBR
        :param use_pbr_mapping: whether to use PBR mapping
        :param shadow_caster: whether to cast shadow
        """

        if self.optimization_process_executed and self.optimized:
            logging.error("Using optimized renderer and optimization process is already excuted, cannot add new "
                          "objects")
            return

        use_pbr = use_pbr and self.rendering_settings.enable_pbr
        use_pbr_mapping = use_pbr_mapping and self.rendering_settings.enable_pbr

        instance_group = InstanceGroup([self.visual_objects[object_id] for object_id in object_ids],
                                       id=len(self.instances),
                                       link_ids=link_ids,
                                       pybullet_uuid=pybullet_uuid,
                                       class_id=class_id,
                                       poses_trans=poses_trans,
                                       poses_rot=poses_rot,
                                       dynamic=dynamic,
                                       robot=robot,
                                       use_pbr=use_pbr,
                                       use_pbr_mapping=use_pbr_mapping,
                                       shadow_caster=shadow_caster)
        self.instances.append(instance_group)

    def add_robot(self,
                  object_ids,
                  link_ids,
                  poses_trans,
                  poses_rot,
                  pybullet_uuid=None,
                  class_id=0,
                  dynamic=False,
                  robot=None):
        """
        Create an instance group (a robot) for a list of visual objects and link it to pybullet

        :param object_ids: object ids of the visual objects
        :param link_ids: link_ids in pybullet
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param pybullet_uuid: body id in pybullet
        :param class_id: class_id to render semantics
        :param dynamic: whether the instance group is dynamic
        :param robot: The robot associated with this InstanceGroup
        """

        if self.optimization_process_executed and self.optimized:
            logging.error("Using optimized renderer and optimization process is already excuted, cannot add new "
                          "objects")
            return

        robot = Robot([self.visual_objects[object_id] for object_id in object_ids],
                      id=len(self.instances),
                      link_ids=link_ids,
                      pybullet_uuid=pybullet_uuid,
                      class_id=class_id,
                      poses_trans=poses_trans,
                      poses_rot=poses_rot,
                      dynamic=dynamic,
                      robot=robot,
                      use_pbr=False,
                      use_pbr_mapping=False)
        self.instances.append(robot)

    def set_camera(self, camera, target, up, cache=False):
        """
        Set camera pose

        :param camera: camera position
        :param target: camera target
        :param up: up direction
        :param cache: whether to cache pose
        """
        self.camera = camera
        self.target = target
        self.up = up
        if cache:
            self.last_V = np.copy(self.cache)

        V = lookat(self.camera, self.target, up=self.up)
        self.V = np.ascontiguousarray(V, np.float32)
        # change shadow mapping camera to be above the real camera
        self.set_light_position_direction([self.camera[0], self.camera[1], 10],
                                          [self.camera[0], self.camera[1], 0])
        if cache:
            self.cache = self.V

    def set_z_near_z_far(self, znear, zfar):
        """
        Set z limit for camera

        :param znear: lower limit for z
        :param zfar: upper limit for z
        """
        self.znear = znear
        self.zfar = zfar

    def set_fov(self, fov):
        """
        Set the field of view. Given the vertical fov, set it.
        Also, compute the horizontal fov based on the aspect ratio, and set it.

        :param fov: vertical fov
        """
        self.vertical_fov = fov
        self.horizontal_fov = 2 * np.arctan(np.tan(self.vertical_fov / 180.0 * np.pi / 2.0) * self.width /
                                            self.height) / np.pi * 180.0
        P = perspective(self.vertical_fov, float(
            self.width) / float(self.height), self.znear, self.zfar)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_light_color(self, color):
        """
        Set light color

        :param color: light color
        """
        self.lightcolor = color

    def get_intrinsics(self):
        """
        Get camera intrinsics

        :return: camera instrincs
        """
        P = self.P
        w, h = self.width, self.height
        znear = self.znear
        a = (2.0 * znear) / P[0, 0]
        b = P[2, 0] * a
        right = (a + b) / 2.0
        left = b - right
        c = -(2.0 * znear) / P[1, 1]
        d = P[2, 1] * c
        top = (c + d) / 2.0
        bottom = d - top
        fu = w * znear / (right - left)
        fv = -h * znear / (top - bottom)

        u0 = w - right * fu / znear
        v0 = h - bottom * fv / znear
        return np.array([[fu, 0, u0], [0, fv, v0], [0, 0, 1]])

    def set_projection_matrix(self, fu, fv, u0, v0, znear, zfar):
        """
        Set projection matrix, given camera intrincs parameters
        """
        w = self.width
        h = self.height
        self.znear = znear
        self.zfar = zfar
        L = -(u0) * znear / fu
        R = +(w - u0) * znear / fu
        T = -(v0) * znear / fv
        B = +(h - v0) * znear / fv
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2 * znear / (R - L)
        P[1, 1] = -2 * znear / (T - B)
        P[2, 0] = (R + L) / (R - L)
        P[2, 1] = (T + B) / (T - B)
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -1.0
        P[3, 2] = (2 * zfar * znear) / (znear - zfar)
        self.P = P

    def readbuffer(self, modes=AVAILABLE_MODALITIES):
        """
        Read framebuffer of rendering.

        :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d', 'scene_flow', 'optical_flow').
        :return: a list of numpy arrays corresponding to `modes`
        """
        results = []

        # single mode
        if isinstance(modes, str):
            modes = [modes]

        for mode in modes:
            if mode not in AVAILABLE_MODALITIES:
                raise Exception('unknown rendering mode: {}'.format(mode))
            frame = self.r.readbuffer_meshrenderer(
                mode, self.width, self.height, self.fbo)
            frame = frame.reshape(self.height, self.width, 4)[::-1, :]
            results.append(frame)
        return results

    def render(self, modes=AVAILABLE_MODALITIES, hidden=(), return_buffer=True, render_shadow_pass=True):
        """
        A function to render all the instances in the renderer and read the output from framebuffer.

        :param modes: a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d', 'scene_flow', 'optical_flow').
        :param hidden: hidden instances to skip. When rendering from a robot's perspective, it's own body can be hidden
        :param return_buffer: whether to return the frame buffers as numpy arrays
        :param render_shadow_pass: whether to render shadow
        :return: a list of float32 numpy arrays of shape (H, W, 4) corresponding to `modes`, where last channel is alpha
        """
        # run optimization process the first time render is called
        if self.optimized and not self.optimization_process_executed:
            self.optimize_vertex_and_texture()

        if 'seg' in modes and self.rendering_settings.msaa:
            logging.warning(
                "Rendering segmentation masks with MSAA on may generate interpolation artifacts. "
                "It is recommended to turn MSAA off when rendering segmentation.")

        render_shadow_pass = render_shadow_pass and 'rgb' in modes
        need_flow_info = 'optical_flow' in modes or 'scene_flow' in modes
        self.update_dynamic_positions(need_flow_info=need_flow_info)

        if self.enable_shadow and render_shadow_pass:
            # shadow pass
            if self.msaa:
                self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
            else:
                self.r.render_meshrenderer_pre(0, 0, self.fbo)

            if self.optimized:
                # If objects are not shadow casters, we do not render them during the shadow pass. This can be achieved
                # by setting their state to hidden for rendering the depth map
                # Store which instances we hide, so we don't accidentally unhide instances that should remain hidden
                shadow_hidden_instances = [
                    i for i in self.instances if not i.shadow_caster and not i.hidden]
                for instance in shadow_hidden_instances:
                    instance.hidden = True
                self.update_hidden_state(shadow_hidden_instances)
                self.r.updateDynamicData(
                    self.shaderProgram, self.pose_trans_array, self.pose_rot_array, self.last_trans_array,
                    self.last_rot_array, self.V, self.last_V, self.P,
                    self.lightV, self.lightP, ShadowPass.HAS_SHADOW_RENDER_SHADOW, self.camera)
                self.r.renderOptimized(self.optimized_VAO)
                for instance in shadow_hidden_instances:
                    instance.hidden = False
                self.update_hidden_state(shadow_hidden_instances)
            else:
                for instance in self.instances:
                    if (instance not in hidden) and instance.shadow_caster:
                        instance.render(
                            shadow_pass=ShadowPass.HAS_SHADOW_RENDER_SHADOW)

            self.r.render_meshrenderer_post()

            if self.msaa:
                self.r.blit_buffer(self.width, self.height,
                                   self.fbo_ms, self.fbo)

            self.r.readbuffer_meshrenderer_shadow_depth(
                self.width, self.height, self.fbo, self.depth_tex_shadow)

        # main pass
        if self.msaa:
            self.r.render_meshrenderer_pre(1, self.fbo_ms, self.fbo)
        else:
            self.r.render_meshrenderer_pre(0, 0, self.fbo)

        if self.rendering_settings.enable_pbr:
            self.r.renderSkyBox(self.skyboxShaderProgram, self.V, self.P)

        if self.optimized:
            if self.enable_shadow:
                self.r.updateDynamicData(
                    self.shaderProgram, self.pose_trans_array, self.pose_rot_array, self.last_trans_array,
                    self.last_rot_array, self.V, self.last_V, self.P,
                    self.lightV, self.lightP, ShadowPass.HAS_SHADOW_RENDER_SCENE, self.camera)
            else:
                self.r.updateDynamicData(
                    self.shaderProgram, self.pose_trans_array, self.pose_rot_array, self.last_trans_array,
                    self.last_rot_array, self.V, self.last_V, self.P,
                    self.lightV, self.lightP, ShadowPass.NO_SHADOW, self.camera)
            self.r.renderOptimized(self.optimized_VAO)
        else:
            for instance in self.instances:
                if instance not in hidden:
                    if self.enable_shadow:
                        instance.render(
                            shadow_pass=ShadowPass.HAS_SHADOW_RENDER_SCENE)
                    else:
                        instance.render(
                            shadow_pass=ShadowPass.NO_SHADOW)

        self.r.render_meshrenderer_post()

        if self.msaa:
            self.r.blit_buffer(self.width, self.height, self.fbo_ms, self.fbo)

        if return_buffer:
            return self.readbuffer(modes)

    def render_companion_window(self):
        """
        Render companion window.
        The viewer is responsible for calling this to update the window,
        if cv2 is not being used for window display
        """
        self.r.render_companion_window_from_buffer(self.fbo)

    def get_visual_objects(self):
        """
        Return visual objects
        """
        return self.visual_objects

    def get_instances(self):
        """
        Return instances
        """
        return self.instances

    def dump(self):
        """
        Dump instance vertex and face information
        """
        instances_vertices = []
        instances_faces = []
        len_v = 0
        for instance in self.instances:
            vertex_info, face_info = instance.dump()
            for v, f in zip(vertex_info, face_info):
                instances_vertices.append(v)
                instances_faces.append(f + len_v)
                len_v += len(v)
        instances_vertices = np.concatenate(instances_vertices, axis=0)
        instances_faces = np.concatenate(instances_faces, axis=0)

        return instances_vertices, instances_faces

    def set_light_pos(self, light):
        """
        Set light position

        :param light: light position
        """
        self.lightpos = light

    def get_num_objects(self):
        """
        Return the number of objects
        """
        return len(self.objects)

    def set_pose(self, pose, idx):
        """
        Set pose for a specific instance

        :param pose: instance pose
        :param idx: instance id
        """
        self.instances[idx].last_rot = np.copy(self.instances[idx].pose_rot)
        self.instances[idx].last_trans = np.copy(
            self.instances[idx].pose_trans)
        self.instances[idx].pose_rot = np.ascontiguousarray(
            quat2rotmat(pose[3:]))
        self.instances[idx].pose_trans = np.ascontiguousarray(
            xyz2mat(pose[:3]))

    def release(self):
        """
        Clean everything, and release the openGL context.
        """
        logging.debug('Releasing. {}'.format(self.glstring))
        self.clean()
        self.r.release()

    def clean(self):
        """
        Clean all the framebuffers, objects and instances
        """
        clean_list = [
            self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d,
            self.depth_tex, self.color_tex_scene_flow, self.color_tex_optical_flow
        ]
        fbo_list = [self.fbo]
        if self.msaa:
            clean_list += [
                self.color_tex_rgb_ms, self.color_tex_normal_ms, self.color_tex_semantics_ms, self.color_tex_3d_ms,
                self.depth_tex_ms, self.color_tex_scene_flow_ms, self.color_tex_optical_flow_ms
            ]
            fbo_list += [self.fbo_ms]

        if self.optimized:
            self.r.clean_meshrenderer_optimized(clean_list, [self.tex_id_1, self.tex_id_2], fbo_list,
                                                [self.optimized_VAO], [self.optimized_VBO], [self.optimized_EBO])
        else:
            self.r.clean_meshrenderer(
                clean_list, self.textures, fbo_list, self.VAOs, self.VBOs)
        self.color_tex_rgb = None
        self.color_tex_normal = None
        self.color_tex_semantics = None
        self.color_tex_3d = None
        self.color_tex_scene_flow = None
        self.color_tex_optical_flow = None
        self.depth_tex = None
        self.fbo = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
        self.visual_objects = []
        self.instances = []
        self.vertex_data = []
        self.shapes = []

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)

        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)

        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(np.concatenate([vec, np.array([1])]))
            return v[:3] / v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v / v[-1]
        else:
            return None

    def transform_pose(self, pose):
        """
        Transform pose from world frame to camera frame

        :param pose: pose in world frame
        :return: pose in camera frame
        """
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def render_robosuite_cameras(self, modes=('rgb')):
        """
        Render robot camera images. This is applicable for robosuite integration with iGibson.

        :return: a list of frames (number of modalities x number of robots)
        """
        frames = []
        hide_robot = self.rendering_settings.hide_robot
        for instance in self.instances:
            if isinstance(instance, Robot):
                for camera in instance.robot.cameras:
                    if camera.is_active():
                        camera_pose = camera.get_pose()
                        camera_pos = camera_pose[:3]
                        camera_ori = camera_pose[3:]
                        camera_ori_mat = quat2rotmat([camera_ori[-1], camera_ori[0], camera_ori[1], camera_ori[2]])[:3, :3]
                        camera_view_dir = camera_ori_mat.dot(np.array([0, 0, -1])) #Mujoco camera points in -z
                        self.set_camera(camera_pos, camera_pos + camera_view_dir, [0, 0, 1])
                        for item in self.render(modes=modes, hidden=[[],[instance]][hide_robot]):
                            frames.append(item)
        return frames

    def render_robot_cameras(self, modes=('rgb')):        
        """
        Render robot camera images

        :return: a list of frames (number of modalities x number of robots)
        """
        frames = []
        for instance in self.instances:
            if isinstance(instance, Robot):
                camera_pos = instance.robot.eyes.get_position()
                orn = instance.robot.eyes.get_orientation()
                mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
                view_direction = mat.dot(np.array([1, 0, 0]))
                self.set_camera(camera_pos, camera_pos +
                                view_direction, [0, 0, 1], cache=True)
                hidden_instances = []
                if self.rendering_settings.hide_robot:
                    hidden_instances.append(instance)
                for item in self.render(modes=modes, hidden=hidden_instances):
                    frames.append(item)
        return frames

    def _get_names_active_cameras(self):
        """
        Query the list of active cameras. 
        Applicable for integration with iGibson.

        :return: a list of camera names
        """
        names = []
        for instance in self.instances:
            if isinstance(instance, Robot):
                for camera in instance.robot.cameras:
                    if camera.is_active():
                        names.append(camera.camera_name)
        return names

    def _switch_camera(self, idx):
        """
        Switches the camera to particular index. 
        Applicable for integration with iGibson.
        """
        for instance in self.instances:
            if isinstance(instance, Robot):
                instance.robot.cameras[idx].switch()

    def _is_camera_active(self, idx):
        """
        Checks if camera at given index is active. 
        Applicable for integration with iGibson.
        """        
        for instance in self.instances:
            if isinstance(instance, Robot):
                return instance.robot.cameras[idx].is_active()

    def _get_camera_name(self, idx):
        """
        Checks if camera at given index is active. 
        Applicable for integration with iGibson.
        """   
        for instance in self.instances:
            if isinstance(instance, Robot):
                return instance.robot.cameras[idx].camera_name    

    def optimize_vertex_and_texture(self):
        """
        Optimize vertex and texture for optimized renderer
        """
        for tex_file in self.texture_files:
            print("Texture: ", tex_file)
        # Set cutoff about 4096, otherwise we end up filling VRAM very quickly
        cutoff = 5000 * 5000
        shouldShrinkSmallTextures = True
        smallTexSize = 512
        texture_files = sorted(self.texture_files.items(), key=lambda x: x[1])
        texture_files = [item[0] for item in texture_files]

        self.tex_id_1, self.tex_id_2, self.tex_id_layer_mapping = \
            self.r.generateArrayTextures(texture_files,
                                         cutoff,
                                         shouldShrinkSmallTextures,
                                         smallTexSize)
        print(self.tex_id_layer_mapping)
        print(len(self.texture_files), self.texture_files)
        self.textures.append(self.tex_id_1)
        self.textures.append(self.tex_id_2)

        offset_faces = []

        curr_index_offset = 0
        for i in range(len(self.vertex_data)):
            face_idxs = self.faces[i]
            offset_face_idxs = face_idxs + curr_index_offset
            offset_faces.append(offset_face_idxs)
            curr_index_offset += len(self.vertex_data[i])

        # List of all primitives to render - these are the shapes that each have a vao_id
        # Some of these may share visual data, but have unique transforms
        duplicate_vao_ids = []
        class_id_array = []
        # Stores use_pbr, use_pbr_mapping and shadow caster, with 1.0 for padding of fourth element
        pbr_data_array = []
        # Stores whether object is hidden or not - we store as a vec4, since this is the smallest
        # alignment unit in the std140 layout that our shaders use for their uniform buffers
        # Note: we can store other variables in the other 3 components in future
        hidden_array = []

        for instance in self.instances:
            if isinstance(instance, Instance):
                ids = instance.object.VAO_ids
                or_buffer_idx_start = len(duplicate_vao_ids)
                duplicate_vao_ids.extend(ids)
                or_buffer_idx_end = len(duplicate_vao_ids)
                # Store indices in the duplicate vao ids array, and hence the optimized rendering buffers, that this Instance will use
                instance.or_buffer_indices = list(
                    np.arange(or_buffer_idx_start, or_buffer_idx_end))
                class_id_array.extend(
                    [float(instance.class_id) / 255.0] * len(ids))
                pbr_data_array.extend(
                    [[float(instance.use_pbr), 1.0, 1.0, 1.0]] * len(ids))
                hidden_array.extend(
                    [[float(instance.hidden), 1.0, 1.0, 1.0]] * len(ids))
            elif isinstance(instance, InstanceGroup) or isinstance(instance, Robot):
                id_sum = 0
                # Collect OR buffer indices over all visual objects in this group
                temp_or_buffer_indices = []
                for vo in instance.objects:
                    ids = vo.VAO_ids
                    or_buffer_idx_start = len(duplicate_vao_ids)
                    duplicate_vao_ids.extend(ids)
                    or_buffer_idx_end = len(duplicate_vao_ids)
                    # Store indices in the duplicate vao ids array, and hence the optimized rendering buffers, that this InstanceGroup will use
                    temp_or_buffer_indices.extend(
                        list(np.arange(or_buffer_idx_start, or_buffer_idx_end)))
                    id_sum += len(ids)
                instance.or_buffer_indices = list(temp_or_buffer_indices)
                class_id_array.extend(
                    [float(instance.class_id) / 255.0] * id_sum)
                pbr_data_array.extend(
                    [[float(instance.use_pbr), 1.0, 1.0, 1.0]] * id_sum)
                hidden_array.extend(
                    [[float(instance.hidden), 1.0, 1.0, 1.0]] * id_sum)

        # Number of shapes in the OR buffer is equal to the number of duplicate vao_ids
        self.or_buffer_shape_num = len(duplicate_vao_ids)
        # Construct trans and rot data to be the right shape
        self.trans_data = np.zeros((self.or_buffer_shape_num, 4, 4))
        self.rot_data = np.zeros((self.or_buffer_shape_num, 4, 4))

        # Variables needed for multi draw elements call
        index_ptr_offsets = []
        index_counts = []
        indices = []
        diffuse_color_array = []
        tex_num_array = []
        tex_layer_array = []
        roughness_tex_num_array = []
        roughness_tex_layer_array = []
        metallic_tex_num_array = []
        metallic_tex_layer_array = []
        normal_tex_num_array = []
        normal_tex_layer_array = []
        transform_param_array = []

        index_offset = 0
        for id in duplicate_vao_ids:
            index_ptr_offsets.append(index_offset)
            id_idxs = list(offset_faces[id].flatten())
            indices.extend(id_idxs)
            index_count = len(id_idxs)
            index_counts.append(index_count)
            index_offset += index_count

            # Generate other rendering data, including diffuse color and texture layer
            id_material = self.materials_mapping[self.mesh_materials[id]]
            texture_id = id_material.texture_id
            if texture_id == -1 or texture_id is None:
                tex_num_array.append(-1)
                tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[texture_id]
                tex_num_array.append(tex_num)
                tex_layer_array.append(tex_layer)

            roughness_texture_id = id_material.roughness_texture_id
            if roughness_texture_id == -1 or roughness_texture_id is None:
                roughness_tex_num_array.append(-1)
                roughness_tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[roughness_texture_id]
                roughness_tex_num_array.append(tex_num)
                roughness_tex_layer_array.append(tex_layer)

            metallic_texture_id = id_material.metallic_texture_id
            if metallic_texture_id == -1 or metallic_texture_id is None:
                metallic_tex_num_array.append(-1)
                metallic_tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[metallic_texture_id]
                metallic_tex_num_array.append(tex_num)
                metallic_tex_layer_array.append(tex_layer)

            normal_texture_id = id_material.normal_texture_id
            if normal_texture_id == -1 or normal_texture_id is None:
                normal_tex_num_array.append(-1)
                normal_tex_layer_array.append(-1)
            else:
                tex_num, tex_layer = self.tex_id_layer_mapping[normal_texture_id]
                normal_tex_num_array.append(tex_num)
                normal_tex_layer_array.append(tex_layer)

            # List of 3 floats
            transform_param = id_material.transform_param
            transform_param_array.append(
                [transform_param[0], transform_param[1], transform_param[2], 1.0])

            kd = np.asarray(id_material.kd, dtype=np.float32)
            # Add padding so can store diffuse color as vec4
            # The 4th element is set to 1 as that is what is used by the fragment shader
            kd_vec_4 = [kd[0], kd[1], kd[2], 1.0]
            diffuse_color_array.append(
                np.ascontiguousarray(kd_vec_4, dtype=np.float32))

        # Convert data into numpy arrays for easy use in pybind
        index_ptr_offsets = np.ascontiguousarray(
            index_ptr_offsets, dtype=np.int32)
        index_counts = np.ascontiguousarray(index_counts, dtype=np.int32)
        indices = np.ascontiguousarray(indices, dtype=np.int32)

        # Convert frag shader data to list of vec4 for use in uniform buffer objects
        frag_shader_data = []
        pbr_data = []
        hidden_data = []
        uv_data = []
        frag_shader_roughness_metallic_data = []
        frag_shader_normal_data = []

        for i in range(len(duplicate_vao_ids)):
            data_list = [float(tex_num_array[i]), float(
                tex_layer_array[i]), class_id_array[i], 0.0]
            frag_shader_data.append(
                np.ascontiguousarray(data_list, dtype=np.float32))
            pbr_data.append(
                np.ascontiguousarray(pbr_data_array[i], dtype=np.float32))
            hidden_data.append(
                np.ascontiguousarray(hidden_array[i], dtype=np.float32))
            roughness_metallic_data_list = [float(roughness_tex_num_array[i]),
                                            float(
                                                roughness_tex_layer_array[i]),
                                            float(metallic_tex_num_array[i]),
                                            float(metallic_tex_layer_array[i]),
                                            ]
            frag_shader_roughness_metallic_data.append(
                np.ascontiguousarray(roughness_metallic_data_list, dtype=np.float32))
            normal_data_list = [float(normal_tex_num_array[i]),
                                float(normal_tex_layer_array[i]),
                                0.0, 0.0
                                ]
            frag_shader_normal_data.append(
                np.ascontiguousarray(normal_data_list, dtype=np.float32))
            uv_data.append(
                np.ascontiguousarray(transform_param_array[i], dtype=np.float32))

        merged_frag_shader_data = np.ascontiguousarray(
            np.concatenate(frag_shader_data, axis=0), np.float32)
        merged_frag_shader_roughness_metallic_data = np.ascontiguousarray(
            np.concatenate(frag_shader_roughness_metallic_data, axis=0), np.float32)
        merged_frag_shader_normal_data = np.ascontiguousarray(
            np.concatenate(frag_shader_normal_data, axis=0), np.float32)
        merged_diffuse_color_array = np.ascontiguousarray(
            np.concatenate(diffuse_color_array, axis=0), np.float32)
        merged_pbr_data = np.ascontiguousarray(
            np.concatenate(pbr_data, axis=0), np.float32)
        self.merged_hidden_data = np.ascontiguousarray(
            np.concatenate(hidden_data, axis=0), np.float32)
        self.merged_uv_data = np.ascontiguousarray(
            np.concatenate(uv_data, axis=0), np.float32)

        merged_vertex_data = np.concatenate(self.vertex_data, axis=0)
        print("Merged vertex data shape:")
        print(merged_vertex_data.shape)
        print("Enable pbr: {}".format(self.rendering_settings.enable_pbr))

        if self.msaa:
            buffer = self.fbo_ms
        else:
            buffer = self.fbo

        self.optimized_VAO, self.optimized_VBO, self.optimized_EBO = \
            self.r.renderSetup(self.shaderProgram, self.V,
                               self.P, self.lightpos,
                               self.lightcolor,
                               merged_vertex_data,
                               index_ptr_offsets, index_counts,
                               indices,
                               merged_frag_shader_data,
                               merged_frag_shader_roughness_metallic_data,
                               merged_frag_shader_normal_data,
                               merged_diffuse_color_array,
                               merged_pbr_data,
                               self.merged_hidden_data,
                               self.merged_uv_data,
                               self.tex_id_1, self.tex_id_2,
                               buffer,
                               float(
                                   self.rendering_settings.enable_pbr),
                               self.depth_tex_shadow)
        self.optimization_process_executed = True

    def update_hidden_state(self, instances):
        """
        Updates the hidden state of a list of instances
        This function is called by instances and not every frame, since hiding is a very infrequent operation.
        """
        for instance in instances:
            buf_idxs = instance.or_buffer_indices
            if not buf_idxs:
                print(
                    'ERROR: trying to set hidden state of an instance that has no visual objects!')
            # Need to multiply buf_idxs by four so we index into the first element of the vec4 corresponding to each buffer index
            vec4_buf_idxs = [idx * 4 for idx in buf_idxs]
            self.merged_hidden_data[vec4_buf_idxs] = float(instance.hidden)
        self.r.updateHiddenData(self.shaderProgram, np.ascontiguousarray(
            self.merged_hidden_data, dtype=np.float32))

    def update_dynamic_positions(self, need_flow_info=False):
        """
        Update all dynamic positions

        :param need_flow_info: whether flow information is required
        """
        for instance in self.instances:
            # if instance.dynamic:
            if isinstance(instance, Instance):
                buf_idxs = instance.or_buffer_indices
                # Continue if instance has no visual objects
                if not buf_idxs:
                    continue
                self.trans_data[buf_idxs] = np.array(instance.pose_trans)
                self.rot_data[buf_idxs] = np.array(instance.pose_rot)
            elif isinstance(instance, InstanceGroup) or isinstance(instance, Robot):
                buf_idxs = instance.or_buffer_indices
                # Continue if instance has no visual objects
                if not buf_idxs:
                    continue
                self.trans_data[buf_idxs] = np.array(instance.poses_trans)
                self.rot_data[buf_idxs] = np.array(instance.poses_rot)

        if need_flow_info:
            # this part could be expensive
            if self.pose_trans_array is not None:
                self.last_trans_array = np.copy(self.pose_trans_array)
            else:
                self.last_trans_array = np.ascontiguousarray(
                    np.concatenate(self.trans_data, axis=0))
            if self.pose_rot_array is not None:
                self.last_rot_array = np.copy(self.pose_rot_array)
            else:
                self.last_rot_array = np.ascontiguousarray(
                    np.concatenate(self.rot_data, axis=0))
        else:
            # dummy pose for zero flow
            self.last_rot_array = self.pose_rot_array
            self.last_trans_array = self.pose_trans_array

        self.pose_trans_array = np.ascontiguousarray(self.trans_data)
        self.pose_rot_array = np.ascontiguousarray(self.rot_data)

    def use_pbr(self, use_pbr, use_pbr_mapping):
        """
        Apply PBR setting to every instance

        :param use_pbr: whether to use pbr
        :param use_pbr_mapping: whether to use pbr mapping
        """
        for instance in self.instances:
            instance.use_pbr = use_pbr
            instance.use_pbr_mapping = use_pbr_mapping

    def setup_lidar_param(self):
        """
        Set up LiDAR params
        """
        lidar_vertical_low = -15 / 180. * np.pi
        lidar_vertical_high = 15 / 180. * np.pi
        lidar_vertical_n_beams = 16
        lidar_vertical_beams = np.arange(lidar_vertical_low, lidar_vertical_high +
                                         (lidar_vertical_high - lidar_vertical_low) /
                                         (lidar_vertical_n_beams-1),
                                         (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams-1))

        lidar_horizontal_low = -45 / 180. * np.pi
        lidar_horizontal_high = 45 / 180. * np.pi
        lidar_horizontal_n_beams = 468
        lidar_horizontal_beams = np.arange(lidar_horizontal_low, lidar_horizontal_high,
                                           (lidar_horizontal_high - lidar_horizontal_low) / (lidar_horizontal_n_beams))

        xx, yy = np.meshgrid(lidar_vertical_beams, lidar_horizontal_beams)
        xx = xx.flatten()
        yy = yy.flatten()

        x_samples = (np.tan(xx) / np.cos(yy) * self.height //
                     2 + self.height//2).astype(np.int)
        y_samples = (np.tan(yy) * self.height//2 +
                     self.height//2).astype(np.int)

        self.x_samples = x_samples.flatten()
        self.y_samples = y_samples.flatten()

    def get_lidar_from_depth(self):
        """
        Get partial LiDAR readings from depth sensors with limited FOV

        :return: partial LiDAR readings with limited FOV
        """
        lidar_readings = self.render(modes=('3d'))[0]
        lidar_readings = lidar_readings[self.x_samples, self.y_samples, :3]
        dist = np.linalg.norm(lidar_readings, axis=1)
        lidar_readings = lidar_readings[dist > 0]
        return lidar_readings

    def get_lidar_all(self, offset_with_camera=np.array([0, 0, 0])):
        """
        Get complete LiDAR readings by patching together partial ones

        :return: complete 360 degree LiDAR readings
        """
        for instance in self.instances:
            if isinstance(instance, Robot):
                camera_pos = instance.robot.eyes.get_position()
                orn = instance.robot.eyes.get_orientation()
                mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
                view_direction = mat.dot(np.array([1, 0, 0]))
                self.set_camera(camera_pos, camera_pos +
                                view_direction, [0, 0, 1])

        original_fov = self.vertical_fov
        self.set_fov(90)
        lidar_readings = []
        view_direction = np.array([1, 0, 0])
        r2 = np.array(
            [[np.cos(-np.pi / 2), -np.sin(-np.pi / 2), 0], [np.sin(-np.pi / 2), np.cos(-np.pi / 2), 0], [0, 0, 1]])
        r3 = np.array(
            [[np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)], [0, 1, 0],  [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])
        transformatiom_matrix = np.eye(3)

        for i in range(4):
            self.set_camera(np.array(self.camera) + offset_with_camera,
                            np.array(self.camera) + offset_with_camera + view_direction, [0, 0, 1])
            lidar_one_view = self.get_lidar_from_depth()
            lidar_readings.append(lidar_one_view.dot(transformatiom_matrix))
            view_direction = r2.dot(view_direction)
            transformatiom_matrix = r3.dot(transformatiom_matrix)

        lidar_readings = np.concatenate(lidar_readings, axis=0)

        self.set_fov(original_fov)
        return lidar_readings
