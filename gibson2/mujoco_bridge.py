from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, InstanceGroup, Instance, quat2rotmat, Material
from gibson2.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.render.viewer import Viewer
from gibson2.simulator import Simulator
from gibson2.utils.utils import quatFromXYZW, quatToXYZW
from gibson2.utils.mesh_util import xyz2mat, quat2rotmat, xyzw2wxyz

from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

import os
import numpy as np
import transforms3d
from transforms3d import quaternions
import logging
import gibson2
from pathlib import Path

import xml.dom.minidom
import xml.etree.ElementTree as ET

import robosuite.utils.transform_utils as T
import pprint


class iGibsonMujocoBridge:
    def __init__(self,
                 mujoco_env,
                 mode='gui',
                 device_idx=0,
                 render_to_tensor=False,
                 camera_name="front_view",
                 image_width=1280,
                 image_height=720,
                 vertical_fov=45,
                 ):
        """
        """
        # physics simulator
        self.mode = mode

        # renderer
        self.camera_name = camera_name
        self.device_idx = device_idx
        self.render_to_tensor = render_to_tensor
        self.env = mujoco_env
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_fov = vertical_fov
        self.render_collision_mesh = 0
        self.render_visual_mesh = 0
        # print("self.render_visual_mesh", self.render_visual_mesh)
        self.mrs_tensor = MeshRendererSettings(msaa=True, enable_shadow=True)
        self.mrs_no_tensor = MeshRendererSettings(msaa=True, enable_shadow=True)
        self.settings = self.mrs_tensor if render_to_tensor else self.mrs_no_tensor
        print("##"*80)
        # self.simulator = Simulator(mode=mode,
        #                            image_width=image_width,
        #                            image_height=image_width,
        #                            rendering_settings=self.settings)


    def set_timestep(self, timestep):
        """
        :param timestep: set timestep after the initialization of Simulator
        """

    def add_viewer(self, 
                 initial_pos = [0,0,1], 
                 initial_view_direction = [1,0,0], 
                 initial_up = [0,0,1]):
        """
        Attach a debugging viewer to the renderer. This will make the step much slower so should be avoided when
        training agents
        """
        self.viewer = Viewer(initial_pos = initial_pos,
            initial_view_direction=initial_view_direction,
            initial_up=initial_up
            )

        self.viewer.renderer = self.renderer

    def reload(self):
        """
        Destroy the MeshRenderer and physics simulator and start again.
        """
        self.disconnect()
        self.load()

    def load(self):

        """
        Set up MeshRenderer and physics simulation client. Initialize the list of objects.
        """
        # print("self.render_visual_mesh", self.render_visual_mesh)
        # mrs_tensor = MeshRendererSettings(msaa=True)
        # mrs_no_tensor = MeshRendererSettings(msaa=True, enable_shadow=True)
        mrs_tensor = MeshRendererSettings(msaa=True, enable_pbr=False, enable_shadow=True, optimized=False)
        mrs_no_tensor = MeshRendererSettings(msaa=True, enable_shadow=True, enable_pbr=False, optimized=False)


        if self.render_to_tensor:
            self.renderer = MeshRendererG2G(width=self.image_width,
                                            height=self.image_height,
                                            vertical_fov=self.vertical_fov,
                                            device_idx=self.device_idx,
                                            rendering_settings=mrs_tensor
                                            )
        else:
            self.renderer = MeshRenderer(width=self.image_width,
                                         height=self.image_height,
                                         vertical_fov=self.vertical_fov,
                                         device_idx=self.device_idx,
                                         rendering_settings=mrs_no_tensor
                                         )

        # This doesn't work after reset -> returns zeros
        # camera_position = self.env.sim.data.get_camera_xpos(self.camera_name)
        # camera_rot_mat = self.env.sim.data.get_camera_xmat(self.camera_name)
        # view_direction = -np.array(camera_rot_mat[0:3,2])

        camera_position = np.array([1.6,  0.,   1.45])
        view_direction = -np.array([0.9632, 0, 0.2574])


        self.renderer.set_camera(camera_position, camera_position + view_direction, [0, 0, 1])
        # posi = [1.2,-0.8,1.1]
        # vd = [-0.8,0.5,-0.1]
        if self.mode == 'gui' and not self.render_to_tensor:
            logging.info("Adding viewer")
            logging.info("Initial camera view aligned with {}".format(self.camera_name))
            logging.info(f"Initial camera position:{camera_position} , view_direction:{view_direction}")
            # self.simulator = Simulator(mode=self.mode,
            #                 image_width=self.image_width,
            #                 image_height=self.image_width,
            #                 rendering_settings=self.settings)
            # self.viewer = self.simulator.viewer           
            # self.viewer.px = camera_position[0]
            # self.viewer.py = camera_position[1]
            # self.viewer.pz = camera_position[2]
            # self.viewer.renderer = self.renderer
            self.add_viewer(initial_pos=camera_position, 
                            initial_view_direction=view_direction)#camera_position, camera_position - view_direction, [0, 0, 1])     

        self.visual_objects = {}
        self.robots = []
        self.scene = None
        self.objects = []       

        verbose = True

        if verbose: print("***********************")

        mjpy_model = self.env.mjpy_model    #Get model from Robosuite environment
        if verbose: print(mjpy_model.get_xml()) #Print XML
        with open("/home/divyansh/xml.xml", 'w') as f:
            f.write(mjpy_model.get_xml())
        xml_root = ET.fromstring(mjpy_model.get_xml())  #Create a workable XML object
        parent_map = {c:p for p in xml_root.iter() for c in p}  #Create parent map to query bodies of geoms   

        pp = pprint.PrettyPrinter() #create a pretty printer object

        #Create a dictionary of all meshes
        meshes = {}
        for mesh in xml_root.iter('mesh'):
            meshes[mesh.get('name')] = mesh.attrib
        
        # import pdb; pdb.set_trace();

        # import pdb; pdb.set_trace();
        if verbose: print("Meshes:")
        if verbose: pp.pprint(meshes)

        #Create a dictionary of all textures
        textures = {}
        texture_ids = {}
        normal_ids = {}
        roughness_ids = {}
        metallic_ids = {}
        for texture in xml_root.iter('texture'):
            # import pdb; pdb.set_trace();
            texture_type = texture.get('type')
            # print(texture.get('file'))
            # print(texture.get('type'), texture.get('name'), texture.get('file'))
            if texture.get('file') is not None:
                textures[texture.get('name')] = texture.attrib
                texture_ids[texture.get('name')] = (self.renderer.r.loadTexture(texture.get('file'), self.settings.texture_scale), texture_type)
                p = Path(texture.get('file'))
                roughness_fname = p.parent / (p.name + '-roughness.png')
                normal_fname = p.parent / p.name + '-normal.png'
                metallic_fname = p.parent / p.name + '-metallic.png'
                if roughness_fname.exists():
                    roughness_ids[texture.get('name')] = self.renderer.r.loadTexture(roughness_fname, self.settings.texture_scale)
                if normal_fname.exists():
                    normal_ids[texture.get('name')] = self.renderer.r.loadTexture(normal_fname, self.settings.texture_scale)
                if metallic_fname.exists():
                    metallic_ids[texture.get('name')] = self.renderer.r.loadTexture(metallic_fname, self.settings.texture_scale)                    

            else:
                value_str = texture.get('rgb1').split()
                value = [float(pp) for pp in value_str]
                texture_ids[texture.get('name')] = (np.array(value), texture_type)  
                roughness_ids[texture.get('name')] = None
                normal_ids[texture.get('name')] = None
                metallic_ids[texture.get('name')] = None


        if verbose: print("Textures:")
        if verbose: pp.pprint(textures)
        # import pdb; pdb.set_trace();

        #Create a dictionary of all materials and Material objects
        materials = {}
        material_objs = {}
        for material in xml_root.iter('material'):
            # import pdb; pdb.set_trace();
            materials[material.get('name')] = material.attrib
            texture_name = material.get('texture')
            if texture_name is not None:
                (texture_id, texture_type) = texture_ids[texture_name]
                roughness_id = roughness_ids[texture_name]
                normal_id = roughness_ids[texture_name]
                # import pdb; pdb.set_trace();
                # print((texture_id, texture_type))

                if type(texture_id) == int: 
                    repeat_str = material.get('texrepeat')

                    if repeat_str is not None:
                        repeat_str = repeat_str.split()
                        repeat = [int(pp) for pp in repeat_str]
                    else:
                        repeat = [1, 1]

                    texuniform = False
                    texuniform_str = material.get('texuniform')
                    if texuniform_str is not None:
                        texuniform = (texuniform_str == "true")

                    material_objs[material.get('name')] = Material('texture',
                                                                   texture_id=texture_id,
                                                                   repeat_x=repeat[0], 
                                                                   repeat_y=repeat[1], 
                                                                   metallic_texture_id=metallic_id,
                                                                   roughness_texture_id=roughness_id,
                                                                   normal_texture_id=normal_id,
                                                                   texuniform=texuniform, 
                                                                   texture_type = texture_type)
                else:                    
                    # This texture may have been a gradient. We don't have a way to do that 
                    material_objs[material.get('name')] = Material('color', kd= texture_id)
            else:
                color_str = material.get('rgba').split()
                color = [float(pp) for pp in color_str]
                rgb = np.array(color)[0:3]
                material_objs[material.get('name')] = Material('color',
                                                               kd=rgb)            
        if verbose: print("Materials:")
        if verbose: pp.pprint(materials)
        # import pdb; pdb.set_trace();

        mujoco_robot = MujocoRobot()

        print(material_objs)
        # import pdb; pdb.set_trace();    

        #Iterate over all cameras
        for camm in xml_root.iter('camera'):
            properties = {}
            properties['pos'] = [0,0,0]
            properties['quat'] = [1,0,0,0]
            parent_body = parent_map[camm]  #Find parent body

            for prop in properties.keys():
                if camm.get(prop) != None:
                    value_str = camm.get(prop).split()
                    value = [float(pp) for pp in value_str]
                    properties[prop] = value

            camera_quat = properties['quat']
            camera_quat = np.array([camera_quat[1],camera_quat[2],camera_quat[3],camera_quat[0]]) #xyzw

            camera_rot_mat = T.quat2mat(camera_quat)
            view_direction = -np.array(camera_rot_mat[0:3,2])
            #self.renderer.add_static_camera(camm.get("name"), properties['pos'], view_direction, parent_body.get('name'))

            parent_body_name = [parent_body.get('name'), 'worldbody'][parent_body.get('name') is None]
            camera = MujocoCamera(parent_body_name, 
                                  properties['pos'],
                                  camera_quat,
                                  active=False, 
                                  mujoco_env = self.env, 
                                  camera_name = camm.get("name"),
                                  )
            mujoco_robot.cameras.append(camera)     

        self.renderer.add_robot([],
                                [],
                                [],
                                [],
                                [],
                                [],
                                dynamic=False,
                                robot=mujoco_robot)     


        #Iterate over all geometries
        for geom in xml_root.iter('geom'):
            # import pdb; pdb.set_trace();
            if verbose: print('-----------------------------------------------------')
            #If the geometry is visual
            # print((geom.get('group') == '1' and self.render_visual_mesh), (geom.get('group') == '0' and self.render_collision_mesh))
            # import pdb; pdb.set_trace();

            # print((geom.get('group') == '1' and self.render_visual_mesh) or (geom.get('group') == '0' and self.render_collision_mesh), geom.get('name'))
            # if 
            if ((geom.get('group') == '1' and self.render_visual_mesh) or (geom.get('group') == '0' and self.render_collision_mesh)):

                if geom.get('name') != None and verbose:
                    print("Geom: " + geom.get('name'))


                parent_body = parent_map[geom]  #Find parent body
                parent_body_name = [parent_body.get('name'), 'worldbody'][parent_body.get('name') is None]
                if verbose: print("Parent body: " + parent_body_name)

                geom_type = geom.get('type')

                # if "wall" not in geom.get('name'):
                #     exit()
                print("orn:", geom.get('quat'), "pos: ", geom.get('pos'), "size: ", geom.get('size'), "rgba: ", geom.get('rgba'))
            
                
                # if geom.get('name') == 'latch_tip':
                #     continue                

                properties = {}
                properties['pos'] = [0,0,0]
                properties['quat'] = [1,0,0,0] # investigate later.
                properties['size'] = [1,1,1]
                properties['rgba'] = [1,1,1]

                for prop in properties.keys():
                    if geom.get(prop) != None:
                        value_str = geom.get(prop).split()
                        value = [float(pp) for pp in value_str]
                        properties[prop] = value

                geom_orn = properties['quat']
                geom_pos = properties['pos']

                # if "wall" in geom.get('name'):
                    # geom_orn = [1,0,0,0]

                geom_material_name = geom.get('material')
                load_texture = True
                geom_material = None
                if geom_material_name is not None:
                    load_texture = False
                    geom_material = material_objs[geom_material_name]
                


                # overriding input material.
                # geom_material = None # not working
                # load_texture = True # not working


                #There is a convention issue with the frame for OBJ files.
                #In this code, meshes have been generated from collada files (.dea) such that
                #importing both WITH MESHLAB, the original Collada (DAE) and the 
                #OBJ are aligned.
                #If you import them with other software (e.g. Blender), they would not align
                #To generate the OBJ I opened the Collada files in Blender and export with
                #Y forward, and Z up in the properties
                #With this convention, we do not need to apply any rotation to the meshes
                if geom_type == 'box':
                    filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')

                    geom_orn = [geom_orn[1],geom_orn[2],geom_orn[3],geom_orn[0]]

                    # if "wall" not in geom.get('name'):
                        # exit()
                    # print("geom_orn:       ",  geom_orn)
                    self.renderer.load_object(filename,
                                              transform_orn=geom_orn,
                                              transform_pos=geom_pos,
                                              input_kd=properties['rgba'][0:3],
                                              scale=2*np.array(properties['size'][0:3]),
                                              load_texture = load_texture,
                                              input_material = geom_material
                                              )
                    self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                               pybullet_uuid=0,
                                               class_id=0,
                                               dynamic=True,
                                               parent_body=parent_body_name,
                                               )

                elif geom_type == 'cylinder':
                    filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cylinder.obj')

                    geom_orn = [geom_orn[1],geom_orn[2],geom_orn[3],geom_orn[0]]
                    self.renderer.load_object(filename,
                                              transform_orn=geom_orn,
                                              transform_pos=geom_pos,
                                              input_kd=properties['rgba'][0:3],
                                              scale= [properties['size'][0], properties['size'][0], properties['size'][1]], #the cylinder.obj has radius 1 and height 2
                                              )
                    self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                               pybullet_uuid=0,
                                               class_id=0,
                                               dynamic=True,
                                               parent_body=parent_body_name)


                elif geom_type == 'sphere':
                    filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')

                    geom_orn = [geom_orn[1],geom_orn[2],geom_orn[3],geom_orn[0]]
                    self.renderer.load_object(filename,
                                              transform_orn=geom_orn,
                                              transform_pos=geom_pos,
                                              input_kd=properties['rgba'][0:3],
                                              scale= [2*properties['size'][0], 2*properties['size'][0], 2*properties['size'][0]], # the sphere8.obj has radius 0.5
                                              )
                    self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                               pybullet_uuid=0,
                                               class_id=0,
                                               dynamic=True,
                                               parent_body=parent_body_name)
                    
                elif geom_type == 'mesh':
                    filename = meshes[geom.attrib['mesh']]['file']
                    filename = os.path.splitext(filename)[0]+'.obj'

                    geom_orn = np.array([geom_orn[1],geom_orn[2],geom_orn[3],geom_orn[0]])

                    geom_rot = T.quat2mat(geom_orn)

                    # This line is commented out to "reload" the same meshes with different tranformations, for example, robot fingers
                    # We need to find a better way to do it if we are going to load the same object many times (save space)
                    # if not filename in self.visual_objects.keys(): 
                    
                    print(filename)
                    print(geom_material)
                    self.renderer.load_object(filename,
                                            transform_orn=geom_orn,
                                            transform_pos=geom_pos,
                                            input_kd=properties['rgba'][0:3],
                                            load_texture = load_texture,
                                            input_material = geom_material,                                            
                                            )
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                    self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                               pybullet_uuid=0,
                                               class_id=0,
                                               dynamic=True,
                                               parent_body=parent_body_name)
                else:
                    print("Other type: " + geom_type)
                    print("This model needs to import a different type of geom. Need more code")
                    exit(-1)

            elif geom.get('type') == 'plane': #Add the plane visuals even if it is not in group 1

                
                print("Plane Attributes:")

                
                props = {}

                # import pdb; pdb.set_trace();
                for propp in ['pos', 'quat', 'size', 'rgba']:
                    if geom.get(propp) != None:
                        prop_str = geom.get(propp).split()
                        prop = [float(pp) for pp in prop_str]
                        props[propp] = prop
                        print(propp)
                        print(prop)
                    else:
                        props[propp] = [0,0,0] # [0,0,0,0]
                        if propp == 'quat':
                            # props[propp] = [0,0,0,1]  
                            props[propp] = [1,0,0,0]  # this is wxyz, but the load obj ethod requires xyzw
                        if propp == 'size':
                            props[propp] = [1,1,1,1]
                        if propp == 'rgba':
                            props[propp] = [1, 1, 1, 1] #[1,0,0,1] (red)
                        print(propp + ' default')

                self.plane_pos = props['pos']
                self.plane_ori = props['quat']

                geom_material_name = geom.get('material')
                load_texture = True
                geom_material = None
                if geom_material_name is not None:
                    load_texture = False
                    geom_material = material_objs[geom_material_name]
                
                
                # import pdb; pdb.set_trace();
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                # import pdb; pdb.set_trace()
                self.renderer.load_object(filename,
                                          transform_orn=props['quat'][0:4],
                                          transform_pos=props['pos'][0:3],
                                          input_kd=props['rgba'][0:3],
                                          scale=[2*props['size'][0], 2*props['size'][1],0.01],
                                          load_texture = load_texture,
                                          input_material = geom_material
                                            ) #Forcing plane to be 1 cm width (this param is the tile size in Mujoco anyway)
                self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                           pybullet_uuid=0,
                                           class_id=0,
                                           dynamic=True,
                                           parent_body="world")

        # import pdb; pdb.set_trace();

    def load_without_pybullet_vis(load_func):
        def wrapped_load_func(*args, **kwargs):
            return None
        return wrapped_load_func

    @load_without_pybullet_vis
    def import_scene(self, scene, texture_scale=1.0, load_texture=True, class_id=0):
        """
        Import a scene. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: Scene object
        :param texture_scale: Option to scale down the texture for rendering
        :param load_texture: If you don't need rgb output, texture loading could be skipped to make rendering faster
        :param class_id: Class id for rendering semantic segmentation
        """


    @load_without_pybullet_vis
    def import_object(self, object, class_id=0):
        """
        :param object: Object to load
        :param class_id: Class id for rendering semantic segmentation
        """
        

    @load_without_pybullet_vis
    def import_robot(self, robot, class_id=0):
        """
        Import a robot into Simulator

        :param robot: Robot
        :param class_id: Class id for rendering semantic segmentation
        :return: id for robot in pybullet
        """

        

    @load_without_pybullet_vis
    def import_interactive_object(self, obj, class_id=0):
        """
        Import articulated objects into simulator

        :param obj:
        :param class_id: Class id for rendering semantic segmentation
        :return: pybulet id
        """
        

    def render(self):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.update_position(instance, self.env)
        if self.mode == 'gui' and not self.viewer is None:
            self.viewer.update()

    @staticmethod
    def update_position(instance, env):
        """
        Update position for an object or a robot in renderer.

        :param instance: Instance in the renderer
        """
        
        if isinstance(instance, Instance):
            #print("Updating geom")

            if instance.parent_body != 'worldbody':
                
                # import pdb; pdb.set_trace();
                pos_body_in_world = env.sim.data.get_body_xpos(instance.parent_body)
                rot_body_in_world = env.sim.data.get_body_xmat(instance.parent_body).reshape((3, 3))
                pose_body_in_world = T.make_pose(pos_body_in_world, rot_body_in_world)
                pose_geom_in_world = pose_body_in_world

                pos, orn = T.mat2pose(pose_geom_in_world) #xyzw
                
                # import pdb; pdb.
                # orn = [orn[-1], orn[0], orn[1], orn[2]] #wxyz

            else:

                pos = [0,0,0]
                orn = [0,0,0,1] #xyzw
                # pos = [0,0,0] if instance.pos is None else instance.pos
                # orn = instance.orn
                # if orn is None:
                #     orn = [0,0,0, 1]
                # else:
                #     pass
                #     # since the orn is already converted to wxyz format during load_obj function call
                #     # I am converting it back to xyzw
                #     # later in this function its converted back to wxyz.
                #     # pass
                #     orn = [orn[3], orn[0], orn[1], orn[2]]  # XYZW
                # orn = [0,0,0,1] if instance.orn is None else orn #wxyz -> [1,0,0,0] 

            # print(pos)
            # print(orn)
            # instance.set_position_for_part(xyz2mat(pos), j)
            # instance.set_rotation_for_part(
            #     quat2rotmat(xyzw2wxyz(orn)), j)            
            instance.set_position(pos)
            instance.set_rotation(quat2rotmat(xyzw2wxyz(orn)))
            # import pdb; pdb.set_trace()

    def isconnected(self):
        """
        :return: pybullet is alive
        """

    def disconnect(self):
        """
        clean up the simulator
        """
        self.renderer.release()


    def close(self):
        self.disconnect()

    # def set_camera(self, camera_id=0):
    #     self.viewer.set_camera(camera_id)


class MujocoRobot(object):
    def __init__(self):
        self.cameras = []

class MujocoCamera(object):
    """
    Camera class to define camera locations and its activation state (to render from them or not)
    """
    def __init__(self, 
                 camera_link_name, 
                 offset_pos = np.array([0,0,0]), 
                 offset_ori = np.array([0,0,0,1]), #xyzw -> Pybullet convention (to be consistent)
                 active=True, 
                 modes = None, 
                 camera_name = None,
                 mujoco_env = None
                 ):
        """
        :param link_name: string, name of the link the camera is attached to
        :param offset_pos: vector 3d, position offset to the reference frame of the link
        :param offset_ori: vector 4d, orientation offset (quaternion: x, y, z, w) to the reference frame of the link
        :param active: boolean, whether the camera is active and we render virtual images from it
        :param modes: string, modalities rendered by this camera, a subset of ('rgb', 'normal', 'seg', '3d'). If None, we use the default of the renderer
        """
        self.camera_link_name = camera_link_name
        self.offset_pos = np.array(offset_pos)
        self.offset_ori = np.array(offset_ori)
        self.active = active
        self.modes = modes
        self.camera_name = [camera_name, camera_link_name + '_cam'][camera_name is None]
        self.mujoco_env = mujoco_env
        ## added by dj
        # self.activate()

    def is_active(self):
        return self.active

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def switch(self):
        self.active = [True, False][self.active]

    def get_pose(self):
        offset_mat = np.eye(4)
        q_wxyz = np.concatenate((self.offset_ori[3:], self.offset_ori[:3]))
        offset_mat[:3, :3] = quaternions.quat2mat(q_wxyz)
        offset_mat[:3, -1] = self.offset_pos

        if self.camera_link_name != 'worldbody':

            pos_body_in_world = self.mujoco_env.sim.data.get_body_xpos(self.camera_link_name)
            rot_body_in_world = self.mujoco_env.sim.data.get_body_xmat(self.camera_link_name).reshape((3, 3))
            pose_body_in_world = T.make_pose(pos_body_in_world, rot_body_in_world) 

            total_pose = np.array(pose_body_in_world).dot(np.array(offset_mat))

            position = total_pose[:3, -1]

            rot = total_pose[:3, :3]
            wxyz = quaternions.mat2quat(rot)
            xyzw = np.concatenate((wxyz[1:], wxyz[:1]))

        else:
            position = np.array(self.offset_pos)
            xyzw = self.offset_ori

        return np.concatenate((position, xyzw))