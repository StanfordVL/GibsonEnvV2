import pybullet as p
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data
from gibson2.data.datasets import get_model_path
import numpy as np
from PIL import Image


class Scene:
    def load(self):
        raise (NotImplementedError())


class StadiumScene(Scene):
    zero_at_running_strip_start_line = True    # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen = 105 * 0.25    # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25    # FOOBALL_FIELD_HALFWID

    def load(self):
        filename = os.path.join(pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        for i in self.ground_plane_mjcf:
            pos, orn = p.getBasePositionAndOrientation(i)
            p.resetBasePositionAndOrientation(i, [pos[0], pos[1], pos[2] - 0.005], orn)

        for i in self.ground_plane_mjcf:
            p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.5])

        return [item for item in self.stadium] + [item for item in self.ground_plane_mjcf]

    def get_random_point(self):
        return self.get_random_point_floor(0)

    def get_random_point_floor(self, floor, random_height=False):
        del floor
        return 0, np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(0.4, 0.8) if random_height else 0.0
        ])


class StadiumSceneInteractive(Scene):
    zero_at_running_strip_start_line = True    # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen = 105 * 0.25    # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25    # FOOBALL_FIELD_HALFWID

    def load(self):
        filename = os.path.join(pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        for i in self.ground_plane_mjcf:
            pos, orn = p.getBasePositionAndOrientation(i)
            p.resetBasePositionAndOrientation(i, [pos[0], pos[1], pos[2] - 0.005], orn)

        for i in self.ground_plane_mjcf:
            p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.5])

        return [item for item in self.stadium] + [item for item in self.ground_plane_mjcf]


class BuildingScene(Scene):
    def __init__(self, model_id):
        self.model_id = model_id

    def load(self):
        filename = os.path.join(get_model_path(self.model_id), "mesh_z_up_downsampled.obj")
        if os.path.isfile(filename):
            print('Using downsampled mesh!')
        else:
            filename = os.path.join(get_model_path(self.model_id), "mesh_z_up.obj")
        scaling = [1, 1, 1]
        collisionId = p.createCollisionShape(p.GEOM_MESH,
                                             fileName=filename,
                                             meshScale=scaling,
                                             flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        visualId = -1
        boundaryUid = p.createMultiBody(baseCollisionShapeIndex=collisionId,
                                        baseVisualShapeIndex=visualId)
        p.changeDynamics(boundaryUid, -1, lateralFriction=1)

        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)

        p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0],
                                          posObj=[0, 0, 0],
                                          ornObj=[0, 0, 0, 1])
        p.changeVisualShape(boundaryUid,
                            -1,
                            rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
                            specularColor=[0.5, 0.5, 0.5])

        floor_height_path = os.path.join(get_model_path(self.model_id), 'floors.txt')

        if os.path.exists(floor_height_path):
            self.floor_map = []
            with open(floor_height_path, 'r') as f:
                self.floors = sorted(list(map(float, f.readlines())))
                print(self.floors)
            for i in range(len(self.floors)):
                trav = np.array(
                    Image.open(
                        os.path.join(get_model_path(self.model_id), 'floor_trav_{}.png'.format(i))))
                self.max_length = trav.shape[0] / 200

                self.floor_map.append(trav)

        return [boundaryUid] + [item for item in self.ground_plane_mjcf]

    def get_random_point(self):
        floor = np.random.randint(0, high=len(self.floors))
        return self.get_random_point_floor(floor)

    def get_random_point_floor(self, floor, random_height=False):
        trav = self.floor_map[floor]
        y = np.where(trav == 255)[0] / 100.0 - self.max_length
        x = np.where(trav == 255)[1] / 100.0 - self.max_length
        idx = np.random.randint(0, high=len(x))
        return floor, np.array([x[idx], y[idx], self.floors[floor]])

    def coord_to_pos(self, x, y):
        x = x / 100.0 - self.max_length
        y = y / 100.0 - self.max_length
        return [x, y]