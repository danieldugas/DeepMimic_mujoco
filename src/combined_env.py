import numpy as np
import math
import random
import json
import traceback
import time

from gym.envs.mujoco import mujoco_env
from gym import utils
from gym.spaces import Box

import py3dtf

from config import Config
from mujoco.mocap_v2 import MocapDM
from deepmimic_env import DPEnv, get_obs


class DPCombinedEnvConfig:
    def __init__(self):
        self.MAX_EP_LENGTH = 1000
        self.VEL_OBS_SCALE = 0.1
        self.FRC_OBS_SCALE = 0.001
        self.ADD_FOOT_CONTACT_OBS = True
        self.ADD_TORSO_OBS = True
        self.ADD_JOINT_FORCE_OBS = False
        self.ADD_ABSPOS_OBS = False
        self.ADD_PHASE_OBS = True

class PlayerAction:
    def __init__(self, name, vx, vy):
        self.name = name
        self.vx = vx
        self.vy = vy

class PAWalk(PlayerAction):
    def __init__(self):
        super().__init__("walk", 1.0, 0.0)

class PARun(PlayerAction):
    def __init__(self):
        super().__init__("run", 3.0, 0.0)


class DPCombinedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """ 

    motions:
    ----

    walk
    run
    action
    getup

    motion states:
    ----

    walk     run  <---timer-end|trigger--->    action
      ^      ^    
    upright detected (high c.o.m + pitch detected / close to first walk frame)
      |   /
    fall detected (low c.o.m + pitch detected / head or hand + floor collision)
      v   
    getup      

    control inputs:
    ----

    onehot
    [
    walk,
    run,
    action,
    vx, # target x vel in body frame
    vy, # target y vel in body frame
    ]

    rewards:
    ----
    motion reward + task reward

    walk: vel matching
    run: vel matching

    termination:
    ----
    too long spent in getup

    schedule:
    ----
    (optional) start with getup until first getup success
    initialize in random motion-state, random mocap frame

    """
    def __init__(self):
        self.robot = "unitree_g1"
        self.ENV_CFG = DPCombinedEnvConfig()

        # motions
        self.walk_motion_config = Config(motion="walk", robot=self.robot)
        self.walk_mocap = MocapDM(robot=self.robot)
        self.walk_mocap.load_mocap(self.walk_motion_config.mocap_path)
        self.run_motion_config = Config(motion="run", robot=self.robot)
        self.run_mocap = MocapDM(robot=self.robot)
        self.run_mocap.load_mocap(self.run_motion_config.mocap_path)
        self.action_motion_config = None
        self.action_mocap = None
        self.getup_motion_config = Config(motion="getup_facedown_slow_FSI", robot=self.robot)
        self.getup_mocap = MocapDM(robot=self.robot)
        self.getup_mocap.load_mocap(self.getup_motion_config.mocap_path)

        # episode variables
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_debug_log = {}
        # motion-state
        self.current_motion_phase_idx = None
        self.current_motion_mocap = None
        self.current_motion_config = None
        self.current_player_action = ("walk", 1.0, 0.0)
        # motion-frame

        mujoco_env.MujocoEnv.__init__(self, self.walk_motion_config.xml_path, 6)
        utils.EzPickle.__init__(self)

        # action space: remove last 14 dims (hand actions)
        if self.robot == "unitree_g1":
            N = self.action_space.shape[0] - 14
            self.action_space = Box(low=self.action_space.low[:N], high=self.action_space.high[:N])
        
    def get_current_motion_state(self):
        qpos = self.current_motion_mocap.data_config[self.idx_curr] * 1.0
        qvel = self.current_motion_mocap.data_vel[self.idx_curr] * 1.0
        return qpos, qvel

    def reset(self):
        # first state : getup
        self.current_motion_phase_idx = 0
        self.current_motion_config = self.getup_motion_config
        self.current_motion_mocap = self.getup_mocap
        # reset variables
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_debug_log = {}
        idx_init = 0
        self.idx_curr = idx_init
        qpos, qvel = self.get_current_motion_state()
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def step(self, action, force_state=None):
        step_times = 1

        # action pre-processing
        mujoco_action = action * 1.
        if self.robot == "unitree_g1":
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
                self.episode_debug_log["robot"] = self.robot
                with open(path, "w") as f:
                    f.write(json.dumps(self.episode_debug_log, indent=4))
                print("Error in step, debug log written to {}".format(path))
                done = True
                return self._get_obs() * 0., 0, done, {}

        # Observation
        # -----------------------------------------
        observation = self._get_obs() * 1.

        info = {}
        reward = 0

        done = False
        
        return observation, reward, done, info
    
    def _get_obs(self):
        current_motion_mocap_len = self.current_motion_mocap.get_length() if self.current_motion_mocap is not None else 0
        return get_obs(self.sim.data, self.model, self.current_motion_phase_idx, current_motion_mocap_len, self.ENV_CFG, self.current_motion_config)




if __name__ == "__main__":
    env = DPCombinedEnv()
    obs = env.reset()
    ep_rew = 0
    for i in range(1000):
        env.render(mode="human")
        a = env.action_space.sample()
        qpos, qvel = env.get_current_motion_state()
        obs, reward, done, info = env.step(a, force_state=(qpos, qvel))
        ep_rew += reward
        if done:
            break
    print("Episode reward: ", ep_rew)