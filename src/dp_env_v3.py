#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd

from mujoco.mocap_v2 import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from gym.envs.mujoco import mujoco_env
from gym import utils

from config import Config
from pyquaternion import Quaternion
import py3dtf

from transformations import quaternion_from_euler

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class DPEnvConfig:
    def __init__(self):
        self.MAX_EP_LENGTH = 1000
        self.VEL_OBS_SCALE = 0.1
        self.FRC_OBS_SCALE = 0.001
        self.ADD_FOOT_CONTACT_OBS = True
        self.ADD_TORSO_OBS = True
        self.ADD_JOINT_FORCE_OBS = False
        self.ADD_ABSPOS_OBS = True
        self.ADD_PHASE_OBS = True

class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    version = "v0.9.abspos_obs_no_com_rew"
    CFG = DPEnvConfig()
    def __init__(self, motion=None, load_mocap=True):
        self.config = Config(motion=motion)
        xml_file_path = self.config.xml_path

        self.weight_pose = 0.5
        self.weight_vel = 0.05
        self.weight_root = 0.2
        self.weight_end_eff = 0.15
        self.weight_com = 0.1

        self.scale_pose = 2.0
        self.scale_vel = 0.1
        self.scale_end_eff = 40.0
        self.scale_root = 5.0
        self.scale_com = 10.0
        self.scale_err = 1.0

        self.mocap = MocapDM()
        self.interface = MujocoInterface()
        if load_mocap:
            self.load_mocap(self.config.mocap_path)
            self.reference_state_init()
            assert len(self.mocap.data) != 0
        else:
            # mujoco init calls step, which needs these items
            N = 3
            self.mocap.data_config = np.zeros((N, 35))
            self.mocap.data_vel = np.zeros((N, 34))
            self.mocap_data_len = N
            self.mocap.data_body_xpos = np.zeros((N, 14, 3))
            self.mocap.data_geom_xpos = np.zeros((N, 16, 3))
        self.idx_curr = -1
        self.idx_tmp_count = -1

        self.episode_reward = 0
        self.episode_length = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 6)
        utils.EzPickle.__init__(self)

        assert len(self.mocap.data_config[0]) == len(self.sim.data.qpos)
        assert len(self.mocap.data_vel[0]) == len(self.sim.data.qvel)
        assert len(self.mocap.data_body_xpos[0]) == len(self.sim.data.body_xpos)
        assert len(self.mocap.data_geom_xpos[0]) == len(self.sim.data.geom_xpos)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()[7:] # ignore root joint
        velocity = self.sim.data.qvel.flat.copy()[6:] # ignore root joint
        S = self.CFG.VEL_OBS_SCALE
        velocity = np.array(velocity) * S
        torso = self.get_torso_obs()
        foot_contact = self.get_foot_contact_obs()
        joint_force = self.get_joint_force_obs()
        abs_pos = self.get_abspos_obs()
        phase_obs  = self.get_phase_obs()
        return np.concatenate((position, velocity, torso, foot_contact, joint_force, abs_pos, phase_obs))

    def get_torso_obs(self):
        if not self.CFG.ADD_TORSO_OBS:
            return []
        b = self.model.body_name2id("chest")
        torso_xyz = np.array(self.data.body_xpos[b])
        torso_wxyz = np.array(self.data.body_xquat[b])
        torso_vel = np.array(self.data.cvel[b][3:])
        vr1, vr2, vr3 = np.array(self.data.cvel[b][:3])
        w, x, y, z = torso_wxyz
        r, p, y = py3dtf.Quaternion(x, y, z, w).to_rpy()
        torso_rpy = np.array([r, p, y])
        BDY = np.array([ # BDYVEC frame: frame where x points forward from chest, horizontal to ground
            [np.cos(-torso_rpy[2]), -np.sin(-torso_rpy[2]), 0],
            [np.sin(-torso_rpy[2]),  np.cos(-torso_rpy[2]), 0],
            [0, 0, 1]
        ])
        vx = BDY[0][0] * torso_vel[0] + BDY[0][1] * torso_vel[1] + BDY[0][2] * torso_vel[2];
        vy = BDY[1][0] * torso_vel[0] + BDY[1][1] * torso_vel[1] + BDY[1][2] * torso_vel[2];
        vz = BDY[2][0] * torso_vel[0] + BDY[2][1] * torso_vel[1] + BDY[2][2] * torso_vel[2];
        S = self.CFG.VEL_OBS_SCALE
        return np.array([
            torso_rpy[0] * S,
            torso_rpy[1] * S,
            vx * S,
            vy * S,
            vz * S,
            vr1 * S,
            vr2 * S,
            vr3 * S,
        ])
    
    def get_foot_contact_obs(self):
        if not self.CFG.ADD_FOOT_CONTACT_OBS:
            return []
        rfoot_name = "right_ankle"
        lfoot_name = "left_ankle"
        floor_name = "floor"
        lfoot_floor_contact = 0.
        rfoot_floor_contact = 0.
        lfoot_other_contact = 0.
        rfoot_other_contact = 0.
        for c in self.data.contact:
            name1 = self.model.geom_id2name(c.geom1)
            name2 = self.model.geom_id2name(c.geom2)
            floortouch = name1 == floor_name or name2 == floor_name
            if (name1 == rfoot_name or name2 == rfoot_name):
                if floortouch:
                    rfoot_floor_contact = 1.
                else:
                    rfoot_other_contact = 1.
            if (name1 == lfoot_name or name2 == lfoot_name):
                if floortouch:
                    lfoot_floor_contact = 1.
                else:
                    lfoot_other_contact = 1.
        return np.array([
            rfoot_floor_contact,
            lfoot_floor_contact,
        ])

    def get_joint_force_obs(self):
        if not self.CFG.ADD_JOINT_FORCE_OBS:
            return []
        # https://github.com/google-deepmind/mujoco/issues/1095#issuecomment-1761836433
        # qfrc_unc was renamed qfrc_smooth in later versions
        jf = self.data.qfrc_unc.flat.copy() + self.data.qfrc_constraint.flat.copy()
        S = self.CFG.FRC_OBS_SCALE
        jf = np.array(jf) * S
        return jf

    def get_phase_obs(self):
        if not self.CFG.ADD_PHASE_OBS:
            return []
        return [1.0 * self.idx_curr / self.mocap_data_len]

    def get_abspos_obs(self):
        if not self.CFG.ADD_ABSPOS_OBS:
            return []
        geom_xpos = np.array(self.sim.data.geom_xpos).flatten() * 1.0
        return geom_xpos


    def reference_state_init(self, idx_init=None):
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        if idx_init is not None:
            self.idx_init = idx_init
        self.idx_curr = self.idx_init
        self.idx_tmp_count = 0

    def early_termination(self):
        pass

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)

    def calc_config_errs(self, env_config, mocap_config):
        raise NotImplementedError("Deprecated")
        assert len(env_config) == len(mocap_config)
        return np.sum(np.abs(env_config - mocap_config))

    def calc_config_reward(self):
        raise NotImplementedError("Deprecated")


    def step(self, action):
        # self.step_len = int(self.mocap_dt // self.model.opt.timestep)
        self.step_len = 1
        # step_times = int(self.mocap_dt // self.model.opt.timestep)
        step_times = 1
        # pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, step_times)
        # pos_after = mass_center(self.model, self.sim)

        # Observation
        # -----------------------------------------
        observation = self._get_obs()

        # Reward
        mass = np.expand_dims(self.model.body_mass, 1)
        # ------------------------------------------
        # Joint Reward
        err_configs = 0.0
        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint
        self.curr_frame = target_config
        curr_config = self.sim.data.qpos[7:] # to exclude root joint
        assert len(curr_config) == len(target_config)
        err_configs = np.sum(np.abs(curr_config - target_config))
        # pitch error (pitch is -1.57 to 1.57)
        w,x,y,z = self.sim.data.qpos[3:7]
        _, curr_root_pitch, _ = py3dtf.Quaternion(x,y,z,w).to_rpy()
        w,x,y,z = self.mocap.data_config[self.idx_curr][3:7]
        _, target_root_pitch, _ = py3dtf.Quaternion(x,y,z,w).to_rpy()
        err_pitch = np.abs(curr_root_pitch - target_root_pitch)
        err_configs += err_pitch # adds pitch error to total error
        reward_config = math.exp(-err_configs)
        # QVel Reward
        target_qvel = self.mocap.data_vel[self.idx_curr][6:] # to exclude root joint
        current_qvel = self.sim.data.qvel[6:] # to exclude root joint
        assert len(target_qvel) == len(current_qvel)
        err_qvel = np.sum(np.abs(target_qvel - current_qvel))
        reward_qvel = math.exp(-0.1 * err_qvel)
        # End effector reward
        end_effectors = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
        err_end_eff = 0.0
        for end_effector in end_effectors:
            idx = self.sim.model.geom_name2id(end_effector)
            err_end_eff += np.linalg.norm(self.sim.data.geom_xpos[idx] - self.mocap.data_geom_xpos[self.idx_curr][idx])**2
        reward_end_eff = math.exp(-40 * err_end_eff)
        # C.O.M reward
        target_body_xpos = self.mocap.data_body_xpos[self.idx_curr]
        current_body_xpos = self.sim.data.body_xpos
        target_com = np.sum(target_body_xpos * mass, 0) / np.sum(mass)
        current_com = np.sum(current_body_xpos * mass, 0) / np.sum(mass)
        com_err = np.linalg.norm(target_com - current_com)**2
        reward_com = math.exp(-10 * com_err)
        # Sum reward
        wp = 0.75
        wv = 0.1
        we = 0.15
        wc = 0.0
        reward = wp * reward_config + wv * reward_qvel + we * reward_end_eff + wc * reward_com

        info = dict()

        # Termination
        # -------------------------------
        done = False
        # Low / high C.O.M termination
        if self.config.motion not in self.config.floor_motions:
            xpos = self.sim.data.xipos
            z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
            done = bool((z_com < 0.7) or (z_com > 2.0))
        # Max episode length
        if self.CFG.MAX_EP_LENGTH != 0:
            if self.episode_length >= self.CFG.MAX_EP_LENGTH:
                done = True
        # Acyclic mocap end
        if (self.idx_curr + 1) == self.mocap_data_len and self.config.motion in self.config.acyclical_motions:
            done = True

        # Post-step
        # ------------------------------------------
        # increment mocap frame
        self.idx_curr = (self.idx_curr + 1) % self.mocap_data_len
        # increment episode counters
        self.episode_reward += reward
        self.episode_length += 1

        return observation, reward, done, info

    def is_done(self):
        raise NotImplementedError("Deprecated")

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset(self):
        self.episode_reward = 0
        self.episode_length = 0
        return self.reset_model()

    def reset_model(self, idx_init=None):
        self.reference_state_init(idx_init=idx_init)
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.idx_tmp_count = -self.step_len
        return observation

    def reset_model_init(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        pass
        # self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.distance = self.model.stat.extent * 1.0
        # self.viewer.cam.lookat[2] = 2.0
        # self.viewer.cam.elevation = -20

    def render(self, mode=None): # needed by video rec
        output = mujoco_env.MujocoEnv.render(self, mode=mode)
        if mode == "rgb_array":
            # add reward to the screen
            string = "{:>5} {:>7.2f}".format(self.episode_length, self.episode_reward)
            # using opencv put_text:
            import cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            textmask = np.array(output)
            cv2.putText(textmask, string, (40, 40), font, 1., (255, 255, 255), 2, cv2.LINE_AA)
            output = textmask
        return output

def test_walk_hand_xpos_mocap(human=False):
    env = DPEnv(motion="walk")
    env.reset_model()
    hand_idx = env.sim.model.geom_name2id("left_wrist")
    mocap_hand_xpos = []
    env_hand_xpos = []
    for i in range(env.mocap_data_len):
        qpos = env.mocap.data_config[i]
        qvel = env.mocap.data_vel[i]
        env.set_state(qpos, qvel)
        if human:
            env.render(mode="human")
        mocap_hand_xpos.append(env.mocap.data_geom_xpos[i][hand_idx][2])
        env_hand_xpos.append(env.sim.data.geom_xpos[hand_idx][2])
    assert np.allclose(mocap_hand_xpos, env_hand_xpos)
    if human:
        from matplotlib import pyplot as plt
        plt.plot(env_hand_xpos)
        plt.plot(mocap_hand_xpos)
        plt.show()

if __name__ == "__main__":
    env = DPEnv()
    env.reset_model()

    import cv2
    from VideoSaver import VideoSaver
    width = 640
    height = 480

    # vid_save = VideoSaver(width=width, height=height)

    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    while True:
        # target_config = env.mocap.data_config[env.idx_curr][7:] # to exclude root joint
        # env.sim.data.qpos[7:] = target_config[:]
        # env.sim.forward()

        qpos = env.mocap.data_config[env.idx_curr]
        qvel = env.mocap.data_vel[env.idx_curr]
        # qpos = np.zeros_like(env.mocap.data_config[env.idx_curr])
        # qvel = np.zeros_like(env.mocap.data_vel[env.idx_curr])
        env.set_state(qpos, qvel)
        env.sim.step()
        env.idx_curr = (env.idx_curr + 1) % env.mocap_data_len
        # env.calc_config_reward()
        env.render(mode="human")
        # print(env._get_obs())
        # root roll pitch yaw
        w,x,y,z = qpos[3:7]; print(py3dtf.Quaternion(x,y,z,w).to_rpy())

    # vid_save.close()
