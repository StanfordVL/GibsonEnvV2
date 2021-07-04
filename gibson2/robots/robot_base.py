import pybullet as p
import gym
import gym.spaces
import gym.utils
import numpy as np
import os
import inspect
import pybullet_data
from transforms3d.euler import euler2quat
from transforms3d import quaternions
from gibson2.utils.utils import quatFromXYZW, quatToXYZW
import gibson2
import logging


class BaseRobot(object):
    """
    Base class for mujoco xml/ROS urdf based agents.
    Handles object loading
    """

    def __init__(self, model_file, base_name=None, scale=1, self_collision=False):
        """
        :param model_file: model filename
        :param base_name: name of the base link
        :param scale: scale, default to 1
        :param self_collision: whether to enable self collision
        """
        self.parts = None
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None
        self.robot_name = None
        self.base_name = base_name

        self.robot_ids = None
        self.robot_mass = None
        self.model_file = model_file
        self.physics_model_dir = os.path.join(gibson2.assets_path, "models")
        self.scale = scale
        self.eyes = None
        # the following variable is only used while rendering robosuite envs.
        self.cameras = []
        logging.info('Loading robot model file: {}'.format(self.model_file))
        if self.model_file[-4:] == 'urdf':
            self.model_type = 'URDF'
        else:
            self.model_type = 'MJCF'
            assert self.scale == 1, 'pybullet does not support scaling for MJCF model (p.loadMJCF)'
        self.config = None
        self.self_collision = self_collision

    def load(self):
        """
        Load the robot model into pybullet

        :return: body id in pybullet
        """
        flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        if self.self_collision:
            flags = flags | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT

        if self.model_type == "MJCF":
            self.robot_ids = p.loadMJCF(os.path.join(
                self.physics_model_dir, self.model_file), flags=flags)
        if self.model_type == "URDF":
            self.robot_ids = (p.loadURDF(os.path.join(
                self.physics_model_dir, self.model_file), globalScaling=self.scale, flags=flags),)

        self.parts, self.jdict, self.ordered_joints, self.robot_body, self.robot_mass = self.parse_robot(
            self.robot_ids)

        assert "eyes" in self.parts, 'Please add a link named "eyes" in your robot URDF file with the same pose as the onboard camera. Feel free to check out assets/models/turtlebot/turtlebot.urdf for an example.'
        self.eyes = self.parts["eyes"]

        return self.robot_ids

    def parse_robot(self, bodies):
        """
        Parse the robot to get properties including joint information and mass

        :param bodies: body ids in pybullet
        :return: parts, joints, ordered_joints, robot_body, robot_mass
        """
        assert len(bodies) == 1, 'robot body has length > 1'

        if self.parts is not None:
            parts = self.parts
        else:
            parts = {}

        if self.jdict is not None:
            joints = self.jdict
        else:
            joints = {}

        if self.ordered_joints is not None:
            ordered_joints = self.ordered_joints
        else:
            ordered_joints = []

        robot_mass = 0.0

        base_name, robot_name = p.getBodyInfo(bodies[0])
        base_name = base_name.decode("utf8")
        robot_name = robot_name.decode("utf8")
        parts[base_name] = BodyPart(base_name,
                                    bodies,
                                    0,
                                    -1)
        self.robot_name = robot_name
        # if base_name is unspecified or equal to the base_name returned by p.getBodyInfo, use this link as robot_body (base_link).
        if self.base_name is None or self.base_name == base_name:
            self.robot_body = parts[base_name]
            self.base_name = base_name

        for j in range(p.getNumJoints(bodies[0])):
            robot_mass += p.getDynamicsInfo(bodies[0], j)[0]
            p.setJointMotorControl2(bodies[0],
                                    j,
                                    p.POSITION_CONTROL,
                                    positionGain=0.1,
                                    velocityGain=0.1,
                                    force=0)
            _, joint_name, joint_type, _, _, _, _, _, _, _, _, _, part_name, _, _, _, _ = \
                p.getJointInfo(bodies[0], j)
            logging.debug('Robot joint: {}'.format(
                p.getJointInfo(bodies[0], j)))
            joint_name = joint_name.decode("utf8")
            part_name = part_name.decode("utf8")

            parts[part_name] = BodyPart(part_name,
                                        bodies,
                                        0,
                                        j)

            # otherwise, use the specified base_name link as robot_body (base_link).
            if self.robot_body is None and self.base_name == part_name:
                self.robot_body = parts[part_name]

            if joint_name[:6] == "ignore":
                Joint(joint_name,
                      bodies,
                      0,
                      j).disable_motor()
                continue

            if joint_name[:8] != "jointfix" and joint_type != p.JOINT_FIXED:
                joints[joint_name] = Joint(joint_name,
                                           bodies,
                                           0,
                                           j)
                ordered_joints.append(joints[joint_name])

        if self.robot_body is None:
            raise Exception('robot body not initialized.')

        return parts, joints, ordered_joints, self.robot_body, robot_mass

    def robot_specific_reset(self):
        """
        Reset function for each specific robot. Overwritten by subclasses
        """
        raise NotImplementedError

    def calc_state(self):
        """
        Calculate proprioceptive states for each specific robot.
        Overwritten by subclasses
        """
        raise NotImplementedError


class BodyPart:
    """
    Body part (link) of Robots
    """

    def __init__(self, body_name, bodies, body_index, body_part_index):
        self.bodies = bodies
        self.body_name = body_name
        self.body_index = body_index
        self.body_part_index = body_part_index
        self.initialPosition = self.get_position()
        self.initialOrientation = self.get_orientation()

    def get_name(self):
        """Get name of body part"""
        return self.body_name

    def _state_fields_of_pose_of(self, body_id, link_id=-1):
        """Get pose of body part"""
        if link_id == -1:
            (x, y, z), (a, b, c, d) = p.getBasePositionAndOrientation(body_id)
        else:
            _, _, _, _, (x, y, z), (a, b, c, d) = p.getLinkState(
                body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def _set_fields_of_pose_of(self, pos, orn):
        """Set pose of body part"""
        p.resetBasePositionAndOrientation(
            self.bodies[self.body_index], pos, orn)

    def get_pose(self):
        """Get pose of body part"""
        return self._state_fields_of_pose_of(self.bodies[self.body_index], self.body_part_index)

    def get_position(self):
        """Get position of body part"""
        return self.get_pose()[:3]

    def get_orientation(self):
        """Get orientation of body part
           Orientation is by default defined in [x,y,z,w]"""
        return self.get_pose()[3:]

    def get_rpy(self):
        """Get roll, pitch and yaw of body part
           [roll, pitch, yaw]"""
        return p.getEulerFromQuaternion(self.get_orientation())

    def set_position(self, position):
        """Get position of body part"""
        self._set_fields_of_pose_of(position, self.get_orientation())

    def set_orientation(self, orientation):
        """Get position of body part
           Orientation is defined in [x,y,z,w]"""
        self._set_fields_of_pose_of(self.current_position(), orientation)

    def set_pose(self, position, orientation):
        """Set pose of body part"""
        self._set_fields_of_pose_of(position, orientation)

    def current_position(self):
        """Synonym method for get_position"""
        return self.get_position()

    def current_orientation(self):
        """Synonym method for get_orientation"""
        return self.get_orientation()

    def reset_position(self, position):  # Backward compatibility
        """Synonym method for set_position"""
        self.set_position(position)

    def reset_orientation(self, orientation):  # Backward compatibility
        """Synonym method for set_orientation"""
        self.set_orientation(orientation)

    def reset_pose(self, position, orientation):  # Backward compatibility
        """Synonym method for set_pose"""
        self.set_pose(position, orientation)

    def get_linear_velocity(self):
        """
        Get linear velocity of the body part
        """
        if self.body_part_index == -1:
            (vx, vy, vz), _ = p.getBaseVelocity(self.bodies[self.body_index])
        else:
            _, _, _, _, _, _, (vx, vy, vz), _ = p.getLinkState(
                self.bodies[self.body_index], self.body_part_index, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def get_angular_velocity(self):
        """
        Get angular velocity of the body part
        """
        if self.body_part_index == -1:
            _, (vr, vp, vyaw) = p.getBaseVelocity(self.bodies[self.body_index])
        else:
            _, _, _, _, _, _, _, (vr, vp, vyaw) = p.getLinkState(
                self.bodies[self.body_index], self.body_part_index, computeLinkVelocity=1)
        return np.array([vr, vp, vyaw])

    def contact_list(self):
        """
        Get contact points of the body part
        """
        return p.getContactPoints(self.bodies[self.body_index], -1, self.body_part_index, -1)


class Joint:
    """
    Joint of Robots
    """

    def __init__(self, joint_name, bodies, body_index, joint_index):
        self.bodies = bodies
        self.body_index = body_index
        self.joint_index = joint_index
        self.joint_name = joint_name

        # read joint type and joint limit from the URDF file
        # lower_limit, upper_limit, max_velocity, max_torque = <limit lower=... upper=... velocity=... effort=.../>
        # "effort" is approximately torque (revolute) / force (prismatic), but not exactly (ref: http://wiki.ros.org/pr2_controller_manager/safety_limits).
        # if <limit /> does not exist, the following will be the default value
        # lower_limit, upper_limit, max_velocity, max_torque = 0.0, -1.0, 0.0, 0.0
        _, _, self.joint_type, _, _, _, _, _, self.lower_limit, self.upper_limit, self.max_torque, self.max_velocity, _, _, _, _, _ \
            = p.getJointInfo(self.bodies[self.body_index], self.joint_index)
        self.joint_has_limit = self.lower_limit < self.upper_limit

        # if joint torque and velocity limits cannot be found in the model file, set a default value for them
        if self.max_torque == 0.0:
            self.max_torque = 100.0
        if self.max_velocity == 0.0:
            # if max_velocity and joint limit are missing for a revolute joint,
            # it's likely to be a wheel joint and a high max_velocity is usually supported.
            if self.joint_type == p.JOINT_REVOLUTE and not self.joint_has_limit:
                self.max_velocity = 15.0
            else:
                self.max_velocity = 1.0

    def __str__(self):
        return "idx: {}, name: {}".format(self.joint_index, self.joint_name)

    def get_state(self):
        """Get state of joint"""
        x, vx, _, trq = p.getJointState(
            self.bodies[self.body_index], self.joint_index)
        return x, vx, trq

    def get_relative_state(self):
        """Get normalized state of joint"""
        pos, vel, trq = self.get_state()

        # normalize position to [-1, 1]
        if self.joint_has_limit:
            mean = (self.lower_limit + self.upper_limit) / 2.0
            magnitude = (self.upper_limit - self.lower_limit) / 2.0
            pos = (pos - mean) / magnitude

        # (trying to) normalize velocity to [-1, 1]
        vel /= self.max_velocity

        # (trying to) normalize torque / force to [-1, 1]
        trq /= self.max_torque

        return pos, vel, trq

    def set_position(self, position):
        """Set position of joint"""
        if self.joint_has_limit:
            position = np.clip(position, self.lower_limit, self.upper_limit)
        p.setJointMotorControl2(self.bodies[self.body_index],
                                self.joint_index,
                                p.POSITION_CONTROL,
                                targetPosition=position)

    def set_velocity(self, velocity):
        """Set velocity of joint"""
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        p.setJointMotorControl2(self.bodies[self.body_index],
                                self.joint_index,
                                p.VELOCITY_CONTROL,
                                targetVelocity=velocity)

    def set_torque(self, torque):
        """Set torque of joint"""
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        p.setJointMotorControl2(bodyIndex=self.bodies[self.body_index],
                                jointIndex=self.joint_index,
                                controlMode=p.TORQUE_CONTROL,
                                force=torque)

    def reset_state(self, pos, vel):
        """
        Reset pos and vel of joint
        """
        p.resetJointState(
            self.bodies[self.body_index], self.joint_index, targetValue=pos, targetVelocity=vel)
        self.disable_motor()

    def disable_motor(self):
        """
        disable the motor of joint
        """
        p.setJointMotorControl2(self.bodies[self.body_index],
                                self.joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0,
                                targetVelocity=0,
                                positionGain=0.1,
                                velocityGain=0.1,
                                force=0)

    def get_joint_relative_state(self):  # Synonym method
        """Synonym method for get_relative_state"""
        return self.get_relative_state()

    def set_motor_position(self, pos):  # Synonym method
        """Synonym method for set_position"""
        return self.set_position(pos)

    def set_motor_torque(self, torque):  # Synonym method
        """Synonym method for set_torque"""
        return self.set_torque(torque)

    def set_motor_velocity(self, vel):  # Synonym method
        """Synonym method for set_velocity"""
        return self.set_velocity(vel)

    def reset_joint_state(self, position, velocity):  # Synonym method
        """Synonym method for reset_state"""
        return self.reset_state(position, velocity)

    def current_position(self):  # Backward compatibility
        """Synonym method for get_state"""
        return self.get_state()

    def current_relative_position(self):  # Backward compatibility
        """Synonym method for get_relative_state"""
        return self.get_relative_state()

    def reset_current_position(self, position, velocity):  # Backward compatibility
        """Synonym method for reset_state"""
        self.reset_state(position, velocity)

    def reset_position(self, position, velocity):  # Backward compatibility
        """Synonym method for reset_state"""
        self.reset_state(position, velocity)
