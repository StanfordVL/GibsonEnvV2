from gibson2.core.physics.robot_bases import BaseRobot
from gibson2.utils.utils import rotate_vector_3d
import numpy as np
import pybullet as p
import os
import gym, gym.spaces
from transforms3d.euler import euler2quat, euler2mat
from transforms3d.quaternions import quat2mat, qmult
import transforms3d.quaternions as quat
import sys
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, \
    set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits, get_base_values

class LocomotorRobot(BaseRobot):
    """ Built on top of BaseRobot
    """

    def __init__(
            self,
            filename,  # robot file name
            action_dim,  # action dimension
            base_name=None,
            scale=1.0,
            control='torque',
            is_discrete=True,
            torque_coef=1.0,
            velocity_coef=1.0,
            self_collision=False
    ):
        BaseRobot.__init__(self, filename, base_name, scale, self_collision)
        self.control = control
        self.is_discrete = is_discrete

        assert type(action_dim) == int, "Action dimension must be int, got {}".format(type(action_dim))
        self.action_dim = action_dim

        if self.is_discrete:
            self.set_up_discrete_action_space()
        else:
            self.set_up_continuous_action_space()

        self.torque_coef = torque_coef
        self.velocity_coef = velocity_coef
        self.scale = scale

    def set_up_continuous_action_space(self):
        """
        Each subclass implements their own continuous action spaces
        """
        raise NotImplementedError

    def set_up_discrete_action_space(self):
        """
        Each subclass implements their own discrete action spaces
        """
        raise NotImplementedError

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_joint_state(0.0, 0.0)

    def get_position(self):
        '''Get current robot position
        '''
        return self.robot_body.get_position()

    def get_orientation(self):
        '''Return robot orientation
        :return: quaternion in xyzw
        '''
        return self.robot_body.get_orientation()

    def get_rpy(self):
        return self.robot_body.get_rpy()

    def set_position(self, pos):
        self.robot_body.set_position(pos)

    def set_orientation(self, orn):
        self.robot_body.set_orientation(orn)

    def set_position_orientation(self, pos, orn):
        self.robot_body.set_pose(pos, orn)

    def get_linear_velocity(self):
        return self.robot_body.get_linear_velocity()

    def get_angular_velocity(self):
        return self.robot_body.get_angular_velocity()

    def move_by(self, delta):
        new_pos = np.array(delta) + self.get_position()
        self.robot_body.reset_position(new_pos)

    def move_forward(self, forward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([forward, 0, 0])))

    def move_backward(self, backward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([-backward, 0, 0])))

    def turn_left(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(-delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def turn_right(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def keep_still(self):
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_velocity(0.0)

    def apply_robot_action(self, action):
        if self.control == 'torque':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(self.torque_coef * j.max_torque * float(np.clip(action[n], -1, +1)))
        elif self.control == 'velocity':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_velocity(self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
        elif self.control == 'position':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_position(action[n])
        elif self.control == 'differential_drive':
            # assume self.ordered_joints = [left_wheel, right_wheel]
            assert action.shape[0] == 2 and len(self.ordered_joints) == 2, 'differential drive requires the first two joints to be two wheels'
            lin_vel, ang_vel = action
            if not hasattr(self, 'wheel_axle_half') or not hasattr(self, 'wheel_radius'):
                raise Exception('Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified.')
            left_wheel_ang_vel = (lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
            right_wheel_ang_vel = (lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius
            self.ordered_joints[0].set_motor_velocity(left_wheel_ang_vel)
            self.ordered_joints[1].set_motor_velocity(right_wheel_ang_vel)
        elif type(self.control) is list or type(self.control) is tuple:
            # if control is a tuple, set different control type for each joint

            if 'differential_drive' in self.control:
                # assume self.ordered_joints = [left_wheel, right_wheel, joint_1, joint_2, ...]
                assert action.shape[0] >= 2 and len(self.ordered_joints) >= 2, 'differential drive requires the first two joints to be two wheels'
                assert self.control[0] == self.control[1] == 'differential_drive', 'differential drive requires the first two joints to be two wheels'
                lin_vel, ang_vel = action[:2]
                if not hasattr(self, 'wheel_axle_half') or not hasattr(self, 'wheel_radius'):
                    raise Exception('Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified.')
                left_wheel_ang_vel = (lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
                right_wheel_ang_vel = (lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius
                self.ordered_joints[0].set_motor_velocity(left_wheel_ang_vel)
                self.ordered_joints[1].set_motor_velocity(right_wheel_ang_vel)

            for n, j in enumerate(self.ordered_joints):
                if self.control[n] == 'torque':
                    j.set_motor_torque(self.torque_coef * j.max_torque * float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'velocity':
                    j.set_motor_velocity(self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'position':
                    j.set_motor_position(action[n])
        else:
            raise Exception('unknown control type: {}'.format(self.control))

    def policy_action_to_robot_action(self, action):
        if self.is_discrete:
            if isinstance(action, (list, np.ndarray)):
                assert len(action) == 1 and isinstance(action[0], (np.int64, int)), \
                    "discrete action has incorrect format"
                action = action[0]
            real_action = self.action_list[action]
        else:
            # self.action_space should always be [-1, 1] for policy training
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # de-normalize action to the appropriate, robot-specific scale
            real_action = (self.action_high - self.action_low) / 2.0 * action + \
                          (self.action_high + self.action_low) / 2.0
        return real_action

    def apply_action(self, action):
        real_action = self.policy_action_to_robot_action(action)
        self.apply_robot_action(real_action)

    def calc_state(self):
        j = np.array([j.get_joint_relative_state() for j in self.ordered_joints]).astype(np.float32).flatten()
        self.joint_position = j[0::3]
        self.joint_velocity = j[1::3]
        self.joint_torque = j[2::3]
        self.joint_at_limit = np.count_nonzero(np.abs(self.joint_position) > 0.99)

        pos = self.get_position()
        rpy = self.get_rpy()

        # rotate linear and angular velocities to local frame
        lin_vel = rotate_vector_3d(self.get_linear_velocity(), *rpy)
        ang_vel = rotate_vector_3d(self.get_angular_velocity(), *rpy)

        state = np.concatenate([pos, rpy, lin_vel, ang_vel, j])
        return state


class Ant(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 1.0)
        LocomotorRobot.__init__(
            self,
            "ant/ant.xml",
            action_dim=8,
            torque_coef=2.5,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="torque",
        )

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        assert False, "Ant does not support discrete actions"


class Humanoid(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.1)
        self.glass_id = None
        self.glass_offset = 0.3
        LocomotorRobot.__init__(
            self,
            "humanoid/humanoid.xml",
            action_dim=17,
            torque_coef=0.41,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="torque",
            self_collision=True,
        )

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.torque] * self.action_dim, [0.0] * self.action_dim]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def robot_specific_reset(self):
        humanoidId = -1
        numBodies = p.getNumBodies()
        for i in range(numBodies):
            bodyInfo = p.getBodyInfo(i)
            if bodyInfo[1].decode("ascii") == 'humanoid':
                humanoidId = i

        ## Spherical radiance/glass shield to protect the robot's camera
        super(Humanoid, self).robot_specific_reset()

        if self.glass_id is None:
            glass_path = os.path.join(self.physics_model_dir, "humanoid/glass.xml")
            glass_id = p.loadMJCF(glass_path)[0]
            self.glass_id = glass_id
            p.changeVisualShape(self.glass_id, -1, rgbaColor=[0, 0, 0, 0])
            p.createMultiBody(baseVisualShapeIndex=glass_id, baseCollisionShapeIndex=-1)
            cid = p.createConstraint(humanoidId,
                                     -1,
                                     self.glass_id,
                                     -1,
                                     p.JOINT_FIXED,
                                     jointAxis=[0, 0, 0],
                                     parentFramePosition=[0, 0, self.glass_offset],
                                     childFramePosition=[0, 0, 0])

        robot_pos = list(self.get_position())
        robot_pos[2] += self.glass_offset
        robot_orn = self.get_orientation()
        p.resetBasePositionAndOrientation(self.glass_id, robot_pos, robot_orn)

        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

    def apply_action(self, action):
        real_action = self.policy_action_to_robot_action(action)
        if self.is_discrete:
            self.apply_robot_action(real_action)
        else:
            for i, m, joint_torque_coef in zip(range(17), self.motors, self.motor_power):
                m.set_motor_torque(float(joint_torque_coef * self.torque_coef * real_action[i]))

    def setup_keys_to_action(self):
        self.keys_to_action = {(ord('w'),): 0, (): 1}


class Husky(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.03)
        LocomotorRobot.__init__(self,
                                "husky/husky.urdf",
                                action_dim=4,
                                torque_coef=2.5,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="torque")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.torque, self.torque, self.torque, self.torque],
                            [-self.torque, -self.torque, -self.torque, -self.torque],
                            [self.torque, -self.torque, self.torque, -self.torque],
                            [-self.torque, self.torque, -self.torque, self.torque], [0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def steering_cost(self, action):
        if not self.is_discrete:
            return 0
        if action == 2 or action == 3:
            return -0.1
        else:
            return 0

    def alive_bonus(self, z, pitch):
        top_xyz = self.parts["top_bumper_link"].get_position()
        bottom_xyz = self.parts["base_link"].get_position()
        alive = top_xyz[2] > bottom_xyz[2]
        return +1 if alive else -100  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
        }


class Quadrotor(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.02)
        LocomotorRobot.__init__(self,
                                "quadrotor/quadrotor.urdf",
                                action_dim=6,
                                torque_coef=2.5,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="torque")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.torque, 0, 0, 0, 0, 0], [-self.torque, 0, 0, 0, 0, 0],
                            [0, self.torque, 0, 0, 0, 0], [0, -self.torque, 0, 0, 0, 0],
                            [0, 0, self.torque, 0, 0, 0], [0, 0, -self.torque, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def apply_action(self, action):
        real_action = self.policy_action_to_robot_action(action)
        p.setGravity(0, 0, 0)
        p.resetBaseVelocity(self.robot_ids[0], real_action[:3], real_action[3:])

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## +x
            (ord('s'),): 1,  ## -x
            (ord('d'),): 2,  ## +y
            (ord('a'),): 3,  ## -y
            (ord('z'),): 4,  ## +z
            (ord('x'),): 5,  ## -z
            (): 6
        }


class Turtlebot(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        LocomotorRobot.__init__(self,
                                "turtlebot/turtlebot.urdf",
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = np.full(shape=self.action_dim, fill_value=self.velocity)
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }


class Turtlebot3(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 0.5)
        LocomotorRobot.__init__(self,
                                "turtlebot3_description/turtlebot3_waffle_pi_for_open_manipulator.urdf",
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = np.full(shape=self.action_dim, fill_value=self.velocity)
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }

class Freight(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        LocomotorRobot.__init__(self,
                                "fetch/freight.urdf",
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }


class Fetch(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        LocomotorRobot.__init__(self,
                                "fetch/fetch.urdf",
                                action_dim=self.wheel_dim + self.torso_lift_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity",
                                self_collision=True)

    def set_up_continuous_action_space(self):
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim +
                                    [self.torso_lift_velocity] * self.torso_lift_dim +
                                    [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        assert False, "Fetch does not support discrete actions"

    def robot_specific_reset(self):
        super(Fetch, self).robot_specific_reset()

        # roll the arm to its body
        robot_id = self.robot_ids[0]
        arm_joints = joints_from_names(robot_id,
                                       [
                                           'torso_lift_joint',
                                           'shoulder_pan_joint',
                                           'shoulder_lift_joint',
                                           'upperarm_roll_joint',
                                           'elbow_flex_joint',
                                           'forearm_roll_joint',
                                           'wrist_flex_joint',
                                           'wrist_roll_joint'
                                       ])

        rest_position = (0.02, np.pi / 2.0 - 0.4, np.pi / 2.0 - 0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0)
        # might be a better pose to initiate manipulation
        # rest_position = (0.30322468280792236, -1.414019864768982,
        #                  1.5178184935241699, 0.8189625336474915,
        #                  2.200358942909668, 2.9631312579803466,
        #                  -1.2862852996643066, 0.0008453550418615341)

        set_joint_positions(robot_id, arm_joints, rest_position)

    def get_end_effector_position(self):
        return self.parts['gripper_link'].get_position()

    def load(self):
        ids = super(Fetch, self).load()
        robot_id = self.robot_ids[0]

        # disable collision between torso_lift_joint and shoulder_lift_joint
        #                   between torso_lift_joint and torso_fixed_joint
        #                   between caster_wheel_joint and estop_joint
        #                   between caster_wheel_joint and laser_joint
        #                   between caster_wheel_joint and torso_fixed_joint
        #                   between caster_wheel_joint and l_wheel_joint
        #                   between caster_wheel_joint and r_wheel_joint
        p.setCollisionFilterPair(robot_id, robot_id, 3, 13, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 3, 22, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 20, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 21, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 22, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 1, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 2, 0)

        return ids


class JR2(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.velocity = config.get('velocity', 1.0)
        LocomotorRobot.__init__(self,
                                "jr2_urdf/jr2.urdf",
                                action_dim=4,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", True),
                                control='velocity')

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.velocity, self.velocity, 0, self.velocity],
                            [-self.velocity, -self.velocity, 0, -self.velocity],
                            [self.velocity, -self.velocity, -self.velocity, 0],
                            [-self.velocity, self.velocity, self.velocity, 0], [0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
        }


class JR2_Kinova(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 0.3)
        self.wheel_dim = 2
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.arm_dim = 5

        LocomotorRobot.__init__(self,
                                "jr2_urdf/jr2_kinova.urdf",
                                action_dim=self.wheel_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control='velocity',
                                self_collision=True)

    def set_up_continuous_action_space(self):
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim + [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.wheel_dim + self.arm_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        assert False, "JR2_Kinova does not support discrete actions"

    def get_end_effector_position(self):
        return self.parts['m1n6s200_end_effector'].get_position()

    # initialize JR's arm to almost the same height as the door handle to ease exploration
    def robot_specific_reset(self):
        super(JR2_Kinova, self).robot_specific_reset()
        self.ordered_joints[2].reset_joint_state(-np.pi / 2.0, 0.0)
        self.ordered_joints[3].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[4].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[5].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[6].reset_joint_state(0.0, 0.0)

    def load(self):
        ids = super(JR2_Kinova, self).load()
        robot_id = self.robot_ids[0]

        # disable collision between base_chassis_joint and pan_joint
        #                   between base_chassis_joint and tilt_joint
        #                   between base_chassis_joint and camera_joint
        #                   between jr2_fixed_body_joint and pan_joint
        #                   between jr2_fixed_body_joint and tilt_joint
        #                   between jr2_fixed_body_joint and camera_joint
        p.setCollisionFilterPair(robot_id, robot_id, 0, 17, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 18, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 19, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 1, 17, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 1, 18, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 1, 19, 0)

        return ids


class Locobot(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        # https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
        # Maximum translational velocity: 70 cm/s
        # Maximum rotational velocity: 180 deg/s (>110 deg/s gyro performance will degrade)
        self.linear_velocity = config.get('linear_velocity', 0.5)
        self.angular_velocity = config.get('angular_velocity', np.pi / 2.0)
        self.wheel_dim = 2
        self.wheel_axle_half = 0.115  # half of the distance between the wheels
        self.wheel_radius = 0.038  # radius of the wheels
        LocomotorRobot.__init__(self,
                                "locobot/locobot.urdf",
                                base_name="base_link",
                                action_dim=self.wheel_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="differential_drive")

    def set_up_continuous_action_space(self):
        self.action_high = np.zeros(self.wheel_dim)
        self.action_high[0] = self.linear_velocity
        self.action_high[1] = self.angular_velocity
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        assert False, "Locobot does not support discrete actions"

    def get_end_effector_position(self):
        return self.parts['gripper_link'].get_position()

