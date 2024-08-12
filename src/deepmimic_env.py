#!/usr/bin/env python3
import numpy as np
import math
import random
import json
import traceback
import time

from mujoco.mocap_v2 import MocapDM

from gym.envs.mujoco import mujoco_env
from gym import utils
from gym.spaces import Box

from config import MotionConfig, RobotConfig
import py3dtf

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

# Common observation functions
def get_obs(mjdata, mjmodel, idx_curr, motion_len, ENV_CFG, robot_config):
    position = mjdata.qpos.flat.copy()[7:] # ignore root joint
    velocity = mjdata.qvel.flat.copy()[6:] # ignore root joint
    S = ENV_CFG.VEL_OBS_SCALE
    velocity = np.array(velocity) * S
    torso = get_torso_obs(mjdata, mjmodel, ENV_CFG, robot_config)
    foot_contact = get_foot_contact_obs(mjdata, mjmodel, ENV_CFG, robot_config)
    joint_force = get_joint_force_obs(mjdata, mjmodel, ENV_CFG, robot_config)
    abs_pos = get_abspos_obs(mjdata, mjmodel, ENV_CFG, robot_config)
    phase_obs  = get_phase_obs(idx_curr, motion_len, ENV_CFG, robot_config)
    return np.concatenate((position, velocity, torso, foot_contact, joint_force, abs_pos, phase_obs))

def get_torso_obs(mjdata, mjmodel, ENV_CFG, robot_config):
    if not ENV_CFG.ADD_TORSO_OBS:
        return []
    b = mjmodel.body_name2id(robot_config.torso_body_name)
    torso_xyz = np.array(mjdata.body_xpos[b])
    torso_wxyz = np.array(mjdata.body_xquat[b])
    torso_vel = np.array(mjdata.cvel[b][3:])
    vr1, vr2, vr3 = np.array(mjdata.cvel[b][:3])
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
    S = ENV_CFG.VEL_OBS_SCALE
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

def get_foot_contact_obs(mjdata, mjmodel, ENV_CFG, robot_config):
    if not ENV_CFG.ADD_FOOT_CONTACT_OBS:
        return []
    rfoot_name = robot_config.rfoot_geom_name
    lfoot_name = robot_config.lfoot_geom_name
    floor_name = robot_config.floor_geom_name
    lfoot_floor_contact = 0.
    rfoot_floor_contact = 0.
    lfoot_other_contact = 0.
    rfoot_other_contact = 0.
    for c in mjdata.contact:
        name1 = mjmodel.geom_id2name(c.geom1)
        name2 = mjmodel.geom_id2name(c.geom2)
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

def get_joint_force_obs(mjdata, mjmodel, ENV_CFG, robot_config):
    if not ENV_CFG.ADD_JOINT_FORCE_OBS:
        return []
    # https://github.com/google-deepmind/mujoco/issues/1095#issuecomment-1761836433
    # qfrc_unc was renamed qfrc_smooth in later versions
    jf = mjdata.qfrc_unc.flat.copy() + mjdata.qfrc_constraint.flat.copy()
    S = ENV_CFG.FRC_OBS_SCALE
    jf = np.array(jf) * S
    return jf

def get_abspos_obs(mjdata, mjmodel, ENV_CFG, robot_config):
    if not ENV_CFG.ADD_ABSPOS_OBS:
        return []
    geom_xpos = np.array(mjdata.geom_xpos).flatten() * 1.0
    return geom_xpos

def get_phase_obs(idx_curr, motion_len, ENV_CFG, robot_config):
    if not ENV_CFG.ADD_PHASE_OBS:
        return []
    phase_01 = np.clip(1.0 * idx_curr / motion_len, 0., 1.)
    return [phase_01]

class DPEnvConfig:
    def __init__(self):
        self.MAX_EP_LENGTH = 1000
        self.VEL_OBS_SCALE = 0.1
        self.FRC_OBS_SCALE = 0.001
        self.ADD_FOOT_CONTACT_OBS = True
        self.ADD_TORSO_OBS = True
        self.ADD_JOINT_FORCE_OBS = False
        self.ADD_ABSPOS_OBS = False
        self.ADD_PHASE_OBS = True


class DPEnvConfig:
    def __init__(self):
        self.MAX_EP_LENGTH = 1000
        self.VEL_OBS_SCALE = 0.1
        self.FRC_OBS_SCALE = 0.001
        self.ADD_FOOT_CONTACT_OBS = True
        self.ADD_TORSO_OBS = True
        self.ADD_JOINT_FORCE_OBS = False
        self.ADD_ABSPOS_OBS = False
        self.ADD_PHASE_OBS = True



class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    version = "v0.9HRS.no_hands_20xact_mocapscale0.85_rplim60"
    ENV_CFG = DPEnvConfig()
    def __init__(self, motion=None, load_mocap=True, robot="humanoid3d"):
        self.motion_config = MotionConfig(motion=motion, robot=robot)
        self.robot_config = RobotConfig(robot=robot)
        xml_file_path = self.motion_config.xml_path
        self.mocap = MocapDM(robot=robot)
        if load_mocap:
            self.load_mocap(self.motion_config.mocap_path)
            self.reference_state_init()
            assert len(self.mocap.data_config) != 0
        else:
            # mujoco init calls step, which needs these items
            self.mocap.data_config = None
            self.mocap.data_vel = None
            self.mocap_data_len = 1
            self.mocap.data_body_xpos = None
            self.mocap.data_geom_xpos = None
        self.idx_curr = -1
        self.idx_tmp_count = -1

        self.episode_reward = 0
        self.episode_length = 0

        self.episode_debug_log = {}

        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 6)
        utils.EzPickle.__init__(self)

        # action space: remove last 14 dims (hand actions)
        if self.robot_config.robot == "unitree_g1":
            N = self.action_space.shape[0] - 14
            self.action_space = Box(low=self.action_space.low[:N], high=self.action_space.high[:N])

    def _get_obs(self):
        return get_obs(self.sim.data, self.model, self.idx_curr, self.mocap_data_len, self.ENV_CFG, self.robot_config)

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
        self.mocap_data_len = len(self.mocap.data_config)

    def calc_config_errs(self, env_config, mocap_config):
        raise NotImplementedError("Deprecated")
        assert len(env_config) == len(mocap_config)
        return np.sum(np.abs(env_config - mocap_config))

    def calc_config_reward(self):
        raise NotImplementedError("Deprecated")


    def step(self, action, force_state=None):
        # self.step_len = int(self.mocap_dt // self.model.opt.timestep)
        self.step_len = 1
        # step_times = int(self.mocap_dt // self.model.opt.timestep)
        step_times = 1

        # action pre-processing
        mujoco_action = action * 1.
        if self.robot_config.robot == "unitree_g1":
            mujoco_action = action * 20.
            if len(mujoco_action) == self.sim.data.ctrl.shape[0] - 14: # need to check because mujoco_env init does a step with full action space
                mujoco_action = np.concatenate((mujoco_action, np.zeros(14))) # add 14 hand actions (0)
        assert len(mujoco_action) == self.sim.data.ctrl.shape[0]

        # pos_before = mass_center(self.model, self.sim)
        if force_state is not None: # bypass kinematics and dynamics, and directly set pose (used for testing)
            qpos, qvel = force_state
            self.set_state(qpos, qvel)
        else:
            try:
                self.do_simulation(mujoco_action, step_times)
            except: # With unitree G1, sometimes the simulation diverges. Here, we log to disk and reset
                full_traceback = traceback.format_exc()
                # write debug log and traceback to /tmp/ for debugging
                path = "/tmp/deepmimic_episode_{}.json".format(time.strftime("%Y%m%d-%H%M_%S"))
                self.episode_debug_log["last_action"] = np.array(action * 1.).tolist()
                self.episode_debug_log["full_traceback"] = full_traceback
                self.episode_debug_log["motion"] = self.motion_config.motion
                self.episode_debug_log["robot"] = self.robot_config.robot
                with open(path, "w") as f:
                    f.write(json.dumps(self.episode_debug_log, indent=4))
                print("Error in step, debug log written to {}".format(path))
                done = True
                return self._get_obs() * 0., 0, done, {}

        # pos_after = mass_center(self.model, self.sim)

        # Observation
        # -----------------------------------------
        observation = self._get_obs()

        if self.mocap.data_config is None:
            return observation, 0, False, {}

        # Reward
        mass = np.expand_dims(self.model.body_mass, 1)
        # ------------------------------------------
        # Joint Reward
        err_configs = 0.0
        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint
        curr_config = self.sim.data.qpos[7:] # to exclude root joint
        target_qvel = self.mocap.data_vel[self.idx_curr][6:] # to exclude root joint
        current_qvel = self.sim.data.qvel[6:] # to exclude root joint
        if self.robot_config.robot == "unitree_g1":
            # [self.model.get_joint_qpos_addr(n) for n in self.model.joint_names]
            qpos_idx = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,   32, 33, 34, 35, 36   ] # exclude root and hand joints
            qvel_idx = [6, 7, 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,   31, 32, 33, 34, 35   ]
            target_config = self.mocap.data_config[self.idx_curr][qpos_idx] # to exclude root joint
            curr_config = self.sim.data.qpos[qpos_idx] # to exclude root joint
            target_qvel = self.mocap.data_vel[self.idx_curr][qvel_idx] # to exclude root joint
            current_qvel = self.sim.data.qvel[qvel_idx] # to exclude root joint
        assert len(curr_config) == len(target_config)
        self.curr_frame = target_config
        err_configs = np.sum(np.abs(curr_config - target_config))
        # pitch error (pitch is -1.57 to 1.57)
        w,x,y,z = self.sim.data.qpos[3:7]
        curr_root_roll, curr_root_pitch, _ = py3dtf.Quaternion(x,y,z,w).to_rpy()
        w,x,y,z = self.mocap.data_config[self.idx_curr][3:7]
        target_root_roll, target_root_pitch, _ = py3dtf.Quaternion(x,y,z,w).to_rpy()
        err_pitch = np.abs(curr_root_pitch - target_root_pitch)
        err_configs += err_pitch # adds pitch error to total error
        reward_config = math.exp(-err_configs)
        # QVel Reward
        assert len(target_qvel) == len(current_qvel)
        err_qvel = np.sum(np.abs(target_qvel - current_qvel))
        reward_qvel = math.exp(-0.1 * err_qvel)
        # End effector reward
        end_effectors = self.robot_config.endeffector_geom_names
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
        # Joint limit reward
        jnt_tol = self.model.jnt_range[1:] * 0.99
        jnt_pos = self.sim.data.qpos[7:]
        if self.robot_config.robot == "unitree_g1":
            jnt_tol = jnt_tol[(np.array(qpos_idx) - 7)]
            jnt_pos = jnt_pos[(np.array(qpos_idx) - 7)]
        qlim_err = np.sum((jnt_pos <= jnt_tol[:,0]) + (jnt_pos >= jnt_tol[:, 1])) / len(jnt_pos) # 0-1
        # Sum reward
        wp = 0.75
        wv = 0.1
        we = 0.15
        wc = 0.0
        wj = -0.1
        reward = wp * reward_config + wv * reward_qvel + we * reward_end_eff + wc * reward_com + wj * qlim_err

        info = dict()
        info["reward_config"] = reward_config
        info["reward_qvel"] = reward_qvel
        info["reward_end_eff"] = reward_end_eff
        info["reward_com"] = reward_com
        info["reward_joint_limit"] = qlim_err

        # Termination
        # -------------------------------
        done = False
        # Low / high C.O.M termination
        if self.motion_config.motion not in self.motion_config.floor_motions:
            xpos = self.sim.data.xipos
            z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
            done = bool((z_com < self.robot_config.low_z) or (z_com > 2.0))
            info["done_reason"] = "low_z" if z_com < self.robot_config.low_z else "high_z"
        # Run: large pitch or roll deviation (to prevent leaping)
        if self.motion_config.motion in ["run"] and self.robot_config.robot == "unitree_g1":
            MAX_ANGLE = np.deg2rad(60.)
            if np.abs(curr_root_roll - target_root_roll) > MAX_ANGLE:
                done = True
                info["done_reason"] = "run roll limit {} {}".format(curr_root_roll, target_root_roll)
            if np.abs(curr_root_pitch - target_root_pitch) > MAX_ANGLE:
                done = True
                info["done_reason"] = "run pitch limit {} {}".format(curr_root_pitch, target_root_pitch)
        # Max episode length
        if self.ENV_CFG.MAX_EP_LENGTH != 0:
            if self.episode_length >= self.ENV_CFG.MAX_EP_LENGTH:
                done = True
                info["done_reason"] = "max_ep_len"
        # Acyclic mocap end
        if (self.idx_curr + 1) == self.mocap_data_len and self.motion_config.motion in self.motion_config.acyclical_motions:
            done = True
            info["done_reason"] = "acyclical_end"

        # Post-step
        # ------------------------------------------
        # increment mocap frame
        self.idx_curr = (self.idx_curr + 1) % self.mocap_data_len
        # increment episode counters
        self.episode_reward += reward
        self.episode_length += 1

        # debug log
        self.episode_debug_log.setdefault("action", []).append(np.array(action * 1.).tolist())
        self.episode_debug_log.setdefault("body_xpos", []).append(np.array(self.sim.data.body_xpos * 1.).tolist())
        self.episode_debug_log.setdefault("body_xvelp", []).append(np.array(self.sim.data.body_xvelp * 1.).tolist())
        self.episode_debug_log.setdefault("qpos", []).append(np.array(self.sim.data.qpos * 1.).tolist())
        self.episode_debug_log.setdefault("qvel", []).append(np.array(self.sim.data.qvel * 1.).tolist())
        self.episode_debug_log.setdefault("reward", []).append(reward)

        if np.max(observation) > 100.0 or np.min(observation) < -100.0:
            full_traceback = "Observation out of bounds (deepmimic_env step)"
            # write debug log and traceback to /tmp/ for debugging
            path = "/tmp/deepmimic_episode_{}.json".format(time.strftime("%Y%m%d-%H%M_%S"))
            self.episode_debug_log["full_traceback"] = full_traceback
            self.episode_debug_log["motion"] = self.motion_config.motion
            self.episode_debug_log["robot"] = self.robot_config.robot
            with open(path, "w") as f:
                f.write(json.dumps(self.episode_debug_log, indent=4))
            print("Observation out of bounds in step, debug log written to {}".format(path))
            done = True
            return self._get_obs() * 0., 0, done, {}

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
        self.episode_debug_log = {}
        return self.reset_model()

    def reset_model(self, idx_init=None):
        self.reference_state_init(idx_init=idx_init)
        if self.mocap.data_config is None:
            return self._get_obs()
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

def loop_motion(motion, robot):
    env = DPEnv(motion=motion, robot=robot)
    env.reset_model(idx_init=0)
    while True:
        qpos = env.mocap.data_config[env.idx_curr]
        qvel = env.mocap.data_vel[env.idx_curr]
        obs, rew, done, info = env.step(np.zeros(env.action_space.shape[0]), force_state=(qpos, qvel))
        env.render(mode="human")

def check_rewards_and_joint_limits(motion, robot):
    env = DPEnv(motion=motion, robot=robot)
    env.reset_model(idx_init=0)
    # Play episode
    # -----------
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    rews = []
    log = []
    while True:
        qpos = env.mocap.data_config[env.idx_curr]
        qvel = env.mocap.data_vel[env.idx_curr]
        obs, rew, done, info = env.step(ac, force_state=(qpos, qvel))
        if env.idx_curr >= env.mocap_data_len - 1:
            done = True
        # obs, rew, _, info = env.step(ac) # checks how a small deviation affects the reward
        rews.append((rew, info))
        if done:
            break
        env.render(mode="human")
        # root rpy
        root_vel = env.sim.data.qvel[:3] * 1.0
        qw,qx,qy,qz = env.sim.data.qpos[3:7] * 1.0
        root_rpy = py3dtf.Quaternion(qx,qy,qz,qw).to_rpy()
        # log
        log.append({
            "qpos": env.sim.data.qpos[7:] * 1., "jnt_min": env.model.jnt_range[1:, 0], "jnt_max": env.model.jnt_range[1:, 1], "jnt_name": env.model.joint_names[1:],
            "end_effector_pos": {x: env.sim.data.geom_xpos[env.model.geom_name2id(x)] * 1.0 for x in env.robot_config.endeffector_geom_names},
            "root_vel": root_vel, "root_rpy": root_rpy,
            })
    # plot rewards
    # -----------
    env.close()
    import matplotlib.pyplot as plt
    qpos = np.array([x["qpos"] for x in log])
    jnt_names = log[0]["jnt_name"]
    jnt_min = np.array([x["jnt_min"] for x in log])
    jnt_max = np.array([x["jnt_max"] for x in log])
    fig, axs = plt.subplots(int(np.ceil(len(qpos[0]) / 4)), 4)
    axs = axs.flatten()
    for i in range(len(qpos[0])):
        axs[i].plot(qpos[:, i], label="qpos")
        axs[i].plot(jnt_min[:, i], label="min")
        axs[i].plot(jnt_max[:, i], label="max")
        axs[i].set_ylabel(jnt_names[i])
        if np.any(qpos[:, i] < jnt_min[:, i]) or np.any(qpos[:, i] > jnt_max[:, i]):
            print("{}: QMIN {} QMAX {} QRANGE {} {}".format(jnt_names[i], np.min(qpos[:, i]), np.max(qpos[:, i]), jnt_min[0, i], jnt_max[0, i]))
    plt.suptitle("Joint limit check")
    plt.legend()
    # reward
    plt.figure()
    tot_rew = [x[0] for x in rews]
    rew_components = {k: [x[k] for _, x in rews] for k in rews[0][1].keys()}
    plt.plot(tot_rew, label="total")
    for k, v in rew_components.items():
        plt.plot(v, label=k)
    plt.legend()
    # end effectors
    fig, axs = plt.subplots(3, 1)
    # EE_POS, EE x N x 3
    for ee_name in log[0]["end_effector_pos"].keys():
        ee_pos = np.array([x["end_effector_pos"][ee_name] for x in log])
        for dim in range(3):
            axs[dim].plot(ee_pos[:, dim], label=ee_name)
    for dim in range(3):
        axs[dim].set_title("XYZ"[dim])
    plt.legend()
    # roll pitch yaw
    plt.figure()
    rpy = np.array([x["root_rpy"] for x in log])
    plt.plot(rpy[:, 0], label="roll")
    plt.plot(rpy[:, 1], label="pitch")
    plt.plot(rpy[:, 2], label="yaw")
    plt.legend()
    plt.title("Root RPY")
    # root xyz
    plt.figure()
    xyz = np.array([x["root_vel"] for x in log])
    plt.plot(xyz[:, 0], label="x")
    plt.plot(xyz[:, 1], label="y")
    plt.plot(xyz[:, 2], label="z")
    plt.legend()
    plt.title("Root XYZ Vel")
    plt.show()

if __name__ == "__main__":
    # loop_motion("getup_facedown", "humanoid3d")
    check_rewards_and_joint_limits(motion="walk", robot="unitree_g1")
