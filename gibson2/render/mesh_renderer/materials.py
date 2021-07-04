import gibson2
import numpy as np
import os
import json
import random
import math


class Material(object):
    """
    Material class used for iG renderer
    """

    def __init__(self,
                 material_type='color',
                 kd=[0.5, 0.5, 0.5],
                 texture_id=None,
                 metallic_texture_id=None,
                 roughness_texture_id=None,
                 normal_texture_id=None,
                 transform_param=[1, 1, 0],
                 texuniform=False,
                 texture_type="2d"                 
                 ):
        """
        :param material_type: color or texture
        :param kd: color parameters
        :param texture_id: albedo texture id
        :param metallic_texture_id: metallic texture id
        :param roughness_texture_id: roughness texture id
        :param normal_texture_id: normal texture id
        :param transform_param: x scale, y scale, rotation
        :param texuniform: texture uniform or not (MuJoCo texture attribute)
        :param texture_type: type of texture (MuJoCo texture attribute)        
        """
        self.material_type = material_type
        self.kd = kd
        self.texture_id = texture_id
        self.metallic_texture_id = metallic_texture_id
        self.roughness_texture_id = roughness_texture_id
        self.normal_texture_id = normal_texture_id
        self.transform_param = transform_param
        self.texuniform = texuniform
        self.texture_type = texture_type         

    def is_texture(self):
        """
        Returns whether this material has texture (as opposed to single color)
        """
        return self.material_type == 'texture'

    def is_pbr_texture(self):
        """
        Returns whether this material can be used for PBR
        """
        return self.material_type == 'texture' and \
            self.metallic_texture_id is not None and \
            self.roughness_texture_id is not None and \
            self.normal_texture_id is not None

    def __str__(self):
        return "Material(material_type: {}, texture_id: {}, metallic_texture_id:{}, roughness_texture_id:{}, " \
               "normal_texture_id:{}, color: {})".format(
                   self.material_type,
                   self.texture_id, self.metallic_texture_id,
                   self.roughness_texture_id, self.normal_texture_id,
                   self.kd)

    def __repr__(self):
        return self.__str__()


class RandomizedMaterial(Material):
    """
    Randomized Material class for material randomization
    """

    def __init__(self,
                 material_classes,
                 material_type='texture',
                 kd=[0.5, 0.5, 0.5],
                 texture_id=None,
                 metallic_texture_id=None,
                 roughness_texture_id=None,
                 normal_texture_id=None):
        """
        :param material_classes: a list of material classes
        :param material_type: color or texture
        :param kd: color parameters
        :param texture_id: albedo texture id
        :param metallic_texture_id: metallic texture id
        :param roughness_texture_id: roughness texture id
        :param normal_texture_id: normal texture id
        :param transform_param: x scale, y scale, rotation
        """
        super(RandomizedMaterial, self).__init__(
            material_type=material_type,
            kd=kd,
            texture_id=texture_id,
            metallic_texture_id=metallic_texture_id,
            roughness_texture_id=roughness_texture_id,
            normal_texture_id=normal_texture_id,
        )
        # a list of material classes, str
        self.material_classes = \
            self.postprocess_material_classes(material_classes)
        # a dict that maps from material class to a list of material files
        # {
        #     'wood': [
        #         {
        #             'diffuse': diffuse_path,
        #             'metallic': metallic_path,
        #             'roughness': None
        #             'normal': normal_path
        #         },
        #         {
        #             ...
        #         }
        #     ],
        #     'metal': [
        #         ...
        #     ]
        # }
        self.material_files = self.get_material_files()
        # a dict that maps from material class to a list of texture ids
        # {
        #     'wood': [
        #         {
        #             'diffuse': 25,
        #             'metallic': 26,
        #             'roughness': None
        #             'normal': 27
        #         },
        #         {
        #             ...
        #         }
        #     ],
        #     'metal': [
        #         ...
        #     ]
        # }

        # WILL be populated when the texture is actually loaded
        self.material_ids = None

        self.random_class = None
        self.random_instance = None

    def postprocess_material_classes(self, material_classes):
        """
        Postprocess material classes.
        We currently do not have all the annotated materials, so we will need
        to convert the materials that we don't have to their closest neighbors
        that we do have.

        :param material_classes: original material classes
        :return material_classes: postprocessed material classes
        """
        for i in range(len(material_classes)):
            material_class = material_classes[i]
            if material_class in ['rock']:
                material_class = 'rocks'
            elif material_class in ['fence', '']:
                material_class = 'wood'
            elif material_class in ['flower', 'leaf']:
                material_class = 'moss'
            elif material_class in ['cork']:
                material_class = 'chipboard'
            elif material_class in ['mirror', 'glass', 'screen']:
                material_class = 'metal'
            elif material_class in ['painting', 'picture']:
                material_class = 'paper'
            material_classes[i] = material_class
        return material_classes

    def get_material_files(self):
        """
        Retrieve the material files from the material dataset,
        given the material classes

        :return material_files: a dict that maps material_class to material files
        """
        material_dir = os.path.join(gibson2.ig_dataset_path, 'materials')
        material_json_file = os.path.join(material_dir, 'materials.json')
        assert os.path.isfile(material_json_file), \
            'cannot find material files: {}'.format(material_json_file)
        with open(material_json_file) as f:
            all_materials = json.load(f)

        material_files = {}
        for material_class in self.material_classes:
            material_files[material_class] = []
            assert material_class in all_materials, \
                'unknown material class: {}'.format(material_class)

            # append gibson2.ig_dataset_path/materials to the beginning
            for material_instance in all_materials[material_class]:
                for key in all_materials[material_class][material_instance]:
                    value = all_materials[material_class][material_instance][key]
                    if value is not None:
                        value = os.path.join(material_dir, value)
                    all_materials[material_class][material_instance][key] = value
            material_files[material_class] = list(
                all_materials[material_class].values())
        return material_files

    def randomize(self):
        """
        Randomize the material by randomly sampling a material id that belongs
        to one of the material classes. All potential materials have already
        been loaded to memory.
        """
        if self.material_ids is None:
            return
        self.random_class = random.choice(list(self.material_ids.keys()))
        self.random_instance = random.choice(
            self.material_ids[self.random_class])
        self.texture_id = self.random_instance['diffuse']
        self.metallic_texture_id = self.random_instance['metallic']
        self.roughness_texture_id = self.random_instance['roughness']
        self.normal_texture_id = self.random_instance['normal']

        # scaling by 4 is typically good
        scale = np.random.normal(loc=4, scale=1)
        # scaling should be at least 2.
        scale = max(scale, 2)
        rotation = random.randint(0, 3) * math.pi / 2.
        self.transform_param = [scale, scale, rotation]

    def __str__(self):
        return (
            "RandomizedMaterial(material_type: {}, texture_id: {}, "
            "metallic_texture_id: {}, roughness_texture_id: {}, "
            "normal_texture_id: {}, color: {}, material_classes: {})".format(
                self.material_type, self.texture_id, self.metallic_texture_id,
                self.roughness_texture_id, self.normal_texture_id, self.kd,
                self.material_classes)
        )
