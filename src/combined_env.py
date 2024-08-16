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

from config import MotionConfig, RobotConfig
from mujoco.mocap_v2 import MocapDM
from deepmimic_env import DPEnv, get_obs, calc_imitation_reward

DEBUG = False
PROFILE = False

class DPCombinedEnvConfig:
    def __init__(self):
        self.MAX_EP_LENGTH = 2000
        self.VEL_OBS_SCALE = 0.1
        self.FRC_OBS_SCALE = 0.001
        self.ADD_FOOT_CONTACT_OBS = True
        self.ADD_TORSO_OBS = True
        self.ADD_JOINT_FORCE_OBS = False
        self.ADD_ABSPOS_OBS = False
        self.ADD_PHASE_OBS = True
        self.ADD_PLAYER_ACTION_OBS = True
        self.MAX_PLAYER_ACTIONS = 3

# PlayerAction stores the information about the current player control
class PlayerAction:
    IDXS = {
        "walk": 0,
        "run": 1,
        "action": 2,
    }
    def __init__(self, name, vx, vy):
        self.name = name
        self.vx = vx
        self.vy = vy
        assert name in PlayerAction.IDXS
    def onehot(self, n_max_actions):
        vec = np.zeros(n_max_actions)
        vec[PlayerAction.IDXS[self.name]] = 1.0
        return vec
    def heading_in_world(self):
        target_vel = np.linalg.norm([self.vx, self.vy])
        target_heading_in_world = np.array([self.vx, self.vy, 0]) / target_vel if target_vel != 0 else np.array([0, 0, 0])
        return target_heading_in_world

class PAWalk(PlayerAction):
    def __init__(self):
        super().__init__("walk", 1.0, 0.0)

class PARun(PlayerAction):
    def __init__(self):
        super().__init__("run", 3.0, 0.0)

# MotionTransitions are like mocap data, but the target motion is just a constant fixed pose (the target pose)
class MotionTransition:
    def __init__(self):
        self.length = None
        self.target_mocap = None
        raise NotImplementedError
    def get_qpos(self, index):
        return self.target_mocap.get_qpos(1)
    def get_qvel(self, index):
        return self.target_mocap.get_qvel(1)
    def get_body_xpos(self, index):
        return self.target_mocap.get_body_xpos(1)
    def get_geom_xpos(self, index):
        return self.target_mocap.get_geom_xpos(1)
    def get_length(self):
        return self.length

class MTToWalk(MotionTransition):
    def __init__(self, walk_mocap):
        self.target_mocap = walk_mocap
        self.motion_name = "to_walk"
        self.length = 120

class MTToRun(MotionTransition):
    def __init__(self, run_mocap):
        self.target_mocap = run_mocap
        self.motion_name = "to_run"
        self.length = 120

class MTToGetup(MotionTransition):
    def __init__(self, getup_mocap):
        self.target_mocap = getup_mocap
        self.motion_name = "to_getup"
        self.length = 120


class DPCombinedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    version = "v0.1.falling_amnesty_tw_nooot"
    # tw: towalk mocap in getup
    # nooot: no-out-of-time - auto transition from getup to to_walk. only fall terminates
    ENV_CFG = DPCombinedEnvConfig()
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
    def __init__(self, verbose=0, _profile=False):
        # uid: three random number/letters/caps
        self.PROFILE = _profile
        self.uid = "".join([random.choice("!@#$%^*()_+-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(3)])
        self.verbose = verbose
        self.robot = "unitree_g1"
        self.robot_config = RobotConfig(robot=self.robot)

        # motions
        walk_motion_config = MotionConfig(motion="walk", robot=self.robot)
        getup_motion_config = MotionConfig(motion="getup_facedown_towalk", robot=self.robot)
        run_motion_config = MotionConfig(motion="run", robot=self.robot)
        self.walk_mocap = MocapDM(robot=self.robot)
        self.walk_mocap.load_mocap(walk_motion_config.mocap_path)
        self.run_mocap = MocapDM(robot=self.robot)
        self.run_mocap.load_mocap(run_motion_config.mocap_path)
        self.action_mocap = None
        self.getup_mocap = MocapDM(robot=self.robot)
        self.getup_mocap.load_mocap(getup_motion_config.mocap_path)
        self.to_walk_mocap = MTToWalk(self.walk_mocap)
        self.to_run_mocap = MTToRun(self.run_mocap)
        self.to_getup_mocap = MTToGetup(self.getup_mocap)

        # episode variables
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_debug_log = {}
        self.debug_n_bad_angles = 0
        # motion-state
        self.current_motion_n_steps = None # how many steps have we been trying to perform this motion
        self.current_motion_mocap = None # current target motion
        self.current_player_action = None
        # motion-frame

        mujoco_env.MujocoEnv.__init__(self, self.robot_config.xml_path, 6)
        utils.EzPickle.__init__(self)

        # action space: remove last 14 dims (hand actions)
        if self.robot == "unitree_g1":
            N = self.action_space.shape[0] - 14
            self.action_space = Box(low=self.action_space.low[:N], high=self.action_space.high[:N])
        
    def get_current_motion_state(self):
        current_mocap_idx = self.current_motion_n_steps % self.current_motion_mocap.get_length()
        qpos = self.current_motion_mocap.get_qpos(current_mocap_idx) * 1.0
        qvel = self.current_motion_mocap.get_qvel(current_mocap_idx) * 1.0
        return qpos, qvel

    def reset(self, rsi=True):
        # first state : getup
#         if rmi: # start with any motion
#             self.current_motion_mocap = [self.getup_mocap, self.walk_mocap, self.run_mocap][random.randint(0, 2)]
#             self.current_player_action = PARun() if self.current_motion_mocap == self.run_mocap else PAWalk()
#             self.current_motion_n_steps = random.randint(0, self.current_motion_mocap.get_length() - 1)
        if rsi:
            self.current_motion_mocap = self.getup_mocap
            self.current_player_action = PAWalk()
            self.current_motion_n_steps = random.randint(0, int(self.current_motion_mocap.get_length() * 0.8))
        else:
            self.current_motion_mocap = self.getup_mocap
            self.current_motion_n_steps = 0
            self.current_player_action = PAWalk()
        # reset variables
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_debug_log = {}
        idx_init = 0
        qpos, qvel = self.get_current_motion_state()
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def step(self, action, force_state=None):
        if self.PROFILE:
            start_timer = time.time()
        step_times = 1
        info = {}

        # action pre-processing
        if self.PROFILE:
            act_timer = time.time()
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
                if self.PROFILE:
                    sim_start = time.time()
                self.do_simulation(mujoco_action, step_times)
                if self.PROFILE:
                    sim_end = time.time()
                    print(f"{self.uid} Sim step (ms) {sim_start * 1000} -> {sim_end * 1000} = {(sim_end - sim_start) * 1000}") # PROFILE
            except: # With unitree G1, sometimes the simulation diverges. Here, we log to disk and reset
                if DEBUG:
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
        if self.PROFILE:
            act_end = time.time()
            print(f"{self.uid} Act (ms) {act_timer * 1000} -> {act_end * 1000} = {(act_end - act_timer) * 1000}")

        # Maybe sample a different player action
        # -----------------------------------------
        # We should use the current PA for this reward, but the next PA for this observation
        if self.PROFILE:
            pa_start = time.time()
        next_player_action = self.current_player_action
        is_player_action_change = False
        AVG_STEPS_TO_PA_CHANGE = 500
        # funny enough, the discrete probability function mean, which is:
        # \sum _{k=0}^{\infty \:}\left(k\left(1-q\right)^k\left(q\right)\right)
        # \sum _{k=0}^{\infty \:}\left(k\left(1-0.01\right)^k\left(0.01\right)\right) <- wolfram alpha solves this: =99
        # is N_STEPS when q ~= 1 / N_STEPS
        if np.random.rand() < (1. / AVG_STEPS_TO_PA_CHANGE): # player input on average every 500 steps
            next_player_action = PARun() if isinstance(self.current_player_action, PAWalk) else PAWalk()
            is_player_action_change = True
        if self.PROFILE:
            pa_end = time.time()
            print(f"{self.uid} PA sample (ms) {pa_start * 1000} -> {pa_end * 1000} = {(pa_end - pa_start) * 1000}")

        # Observation
        # -----------------------------------------
        if self.PROFILE:
            obs_start = time.time()
        observation = self._get_obs() * 1.
        if self.PROFILE:
            obs_end = time.time()
            print(f"{self.uid} Obs get (ms) {obs_start * 1000} -> {obs_end * 1000} = {(obs_end - obs_start) * 1000}")


        # Reward
        # -----------------------------------------
        if self.PROFILE:
            rew_start = time.time()
        # imitation reward
        wp = 0.75
        wv = 0.1
        we = 0.15
        wc = 0.0
        wj = -0.1
        current_motion_mocap_len = self.current_motion_mocap.get_length() if self.current_motion_mocap is not None else 1
        mocap_step_idx = (self.current_motion_n_steps % current_motion_mocap_len) if self.current_motion_n_steps is not None else 0
        imitation_reward, intermediate_values = calc_imitation_reward(
            wp, wv, we, wc, wj,
            self.sim.data, self.model, self.current_motion_mocap, mocap_step_idx, self.ENV_CFG, self.robot_config, info
        )
        # task reward
        task_reward = 0.0
        if self.current_motion_mocap == self.walk_mocap or self.current_motion_mocap == self.run_mocap:
            # heading + velocity error (only when walk or run)
            # here we compare the mocap freejoint qvel to the sim freejoint qvel (assumes mocap heading is same as PlayerAction heading)
            assert np.allclose(self.current_player_action.heading_in_world()[0], 1.0)
            target_vel = self.current_motion_mocap.get_qvel(mocap_step_idx)[:2]
            freejoint_vel = self.sim.data.qvel[:2]
            heading_and_vel_error = np.linalg.norm(target_vel - freejoint_vel)
            task_reward = np.exp(-heading_and_vel_error * 10.) # error 0.5m/s, reward = 0.006 # error 0.1m/s, reward = 0.367
        wi = 0.7
        wt = 0.3
        reward = imitation_reward * wi + task_reward * wt
        if self.PROFILE:
            rew_end = time.time()
            print(f"{self.uid} Reward calc (ms) {rew_start * 1000} -> {rew_end * 1000} = {(rew_end - rew_start) * 1000}")
        
        info["imitation_reward"] = imitation_reward
        info["task_reward"] = task_reward

        # tgt root velocity - 

        # Termination
        # -------------------------------
        if self.PROFILE:
            term_start = time.time()
        done = False
        # Ran out of time
        #   getup   -> done
        #   togetup -> getup
        #   towalk  -> walk
        #   torun   -> run
        #   walk    x
        #   run     x
        # Random playeraction change
        #   getup   x
        #   togetup x
        #   towalk  -> towalk / torun (dep. on PlayerAction)
        #   torun   -> towalk / torun (dep. on PlayerAction)
        #   walk    ->   walk / torun (dep. on PlayerAction)
        #   run     -> towalk / run   (dep. on PlayerAction)
        # Successful
        #   getup   -> towalk / torun (dep. on PlayerAction)
        #   togetup -> getup
        #   towalk  -> walk
        #   torun   -> run
        #   walk    x
        #   run     x
        # Fallen
        #   getup   x
        #   togetup x
        #   towalk  -> to_getup
        #   torun   -> to_getup
        #   walk    -> to_getup
        #   run     -> to_getup
        if self.current_motion_mocap is not None:
            is_out_of_time = self.current_motion_n_steps >= self.current_motion_mocap.get_length()
            if is_out_of_time:
                if self.current_motion_mocap == self.getup_mocap:
#                     done = True
#                     info["done_reason"] = "oot_in_getup"
                    self.change_to_motion(self.to_getup_mocap)
                if self.current_motion_mocap == self.to_getup_mocap:
                    self.change_to_motion(self.getup_mocap)
                if self.current_motion_mocap == self.to_walk_mocap:
                    self.change_to_motion(self.walk_mocap)
                if self.current_motion_mocap == self.to_run_mocap:  
                    self.change_to_motion(self.run_mocap)
            if is_player_action_change:
                if self.current_motion_mocap == self.to_walk_mocap:
                    if isinstance(next_player_action, PARun):
                        self.change_to_motion(self.to_run_mocap)
                if self.current_motion_mocap == self.to_run_mocap:
                    if isinstance(next_player_action, PAWalk):
                        self.change_to_motion(self.to_walk_mocap)
                if self.current_motion_mocap == self.walk_mocap:
                    if isinstance(next_player_action, PARun):
                        self.change_to_motion(self.to_run_mocap)
                if self.current_motion_mocap == self.run_mocap:
                    if isinstance(next_player_action, PAWalk):
                        self.change_to_motion(self.to_walk_mocap)
            ALIM = np.deg2rad(15)
            mass, curr_root_roll, target_root_roll, curr_root_pitch, target_root_pitch, config_angle_diffs = intermediate_values
            is_pitch_close = np.abs(curr_root_pitch - target_root_pitch) < ALIM
            is_roll_close = np.abs(curr_root_roll - target_root_roll) < ALIM
            is_angle_diff_close = np.all(np.abs(config_angle_diffs) < ALIM)
            is_successful = is_pitch_close and is_roll_close and is_angle_diff_close
            self.debug_n_bad_angles = np.sum(np.abs(config_angle_diffs) > ALIM) + (np.abs(curr_root_pitch - target_root_pitch) > ALIM) + (np.abs(curr_root_roll - target_root_roll) > ALIM)
            if is_successful:
                if self.current_motion_mocap == self.getup_mocap:
                    # only check for the last 20 frames:
                    if self.current_motion_n_steps > self.current_motion_mocap.get_length() - 20:
                        self.change_to_motion(self.to_walk_mocap)
                if self.current_motion_mocap == self.to_getup_mocap:
                    self.change_to_motion(self.getup_mocap)
                if self.current_motion_mocap == self.to_walk_mocap:
                    self.change_to_motion(self.walk_mocap)
                if self.current_motion_mocap == self.to_run_mocap:
                    self.change_to_motion(self.run_mocap)
            is_fallen = False
            if self.current_motion_mocap in [self.to_walk_mocap, self.to_run_mocap, self.walk_mocap, self.run_mocap]:
                xpos = self.sim.data.xipos
                z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
                is_fallen = bool((z_com < self.robot_config.low_z) or (z_com > 2.0))
                # Run: large pitch or roll deviation (to prevent leaping)
                MAX_ANGLE = np.deg2rad(60.)
                if np.abs(curr_root_roll - target_root_roll) > MAX_ANGLE:
                    is_fallen = True
                if np.abs(curr_root_pitch - target_root_pitch) > MAX_ANGLE:
                    is_fallen = True
                if is_fallen:
                    has_earned_amnesty = self.current_motion_mocap in [self.walk_mocap, self.run_mocap] and self.current_motion_n_steps > 150
                    if not has_earned_amnesty: # we allow falling only after some successfuly walking/running
                        done = True
                        info["done_reason"] = "fallen without amnesty"
                    if self.current_motion_mocap == self.to_walk_mocap:
                        self.change_to_motion(self.to_getup_mocap)
                    if self.current_motion_mocap == self.to_run_mocap:
                        self.change_to_motion(self.to_getup_mocap)
                    if self.current_motion_mocap == self.walk_mocap:
                        self.change_to_motion(self.to_getup_mocap)
                    if self.current_motion_mocap == self.run_mocap:
                        self.change_to_motion(self.to_getup_mocap)
            # Max episode length
            if self.ENV_CFG.MAX_EP_LENGTH != 0:
                if self.episode_length >= self.ENV_CFG.MAX_EP_LENGTH:
                    done = True
                    info["done_reason"] = "max_ep_len"
        if self.PROFILE:
            term_end = time.time()
            print(f"{self.uid} Term check (ms) {term_start * 1000} -> {term_end * 1000} = {(term_end - term_start) * 1000}")
            
        
        # Post-step
        # ------------------------------------------
        if self.PROFILE:
            post_start = time.time()
        # set player action
        self.current_player_action = next_player_action
        # increment mocap frame
        if self.current_motion_n_steps is not None:
            self.current_motion_n_steps += 1
        # increment episode counters
        self.episode_reward += reward
        self.episode_length += 1


        # debug log
        if DEBUG:
            self.episode_debug_log.setdefault("action", []).append(np.array(action * 1.).tolist())
            self.episode_debug_log.setdefault("body_xpos", []).append(np.array(self.sim.data.body_xpos * 1.).tolist())
            self.episode_debug_log.setdefault("body_xvelp", []).append(np.array(self.sim.data.body_xvelp * 1.).tolist())
            self.episode_debug_log.setdefault("qpos", []).append(np.array(self.sim.data.qpos * 1.).tolist())
            self.episode_debug_log.setdefault("qvel", []).append(np.array(self.sim.data.qvel * 1.).tolist())
            self.episode_debug_log.setdefault("reward", []).append(reward)

        if np.max(observation) > 100.0 or np.min(observation) < -100.0:
            if DEBUG:
                full_traceback = "Observation out of bounds (deepmimic_env step)"
                # write debug log and traceback to /tmp/ for debugging
                path = "/tmp/deepmimic_episode_{}.json".format(time.strftime("%Y%m%d-%H%M_%S"))
                self.episode_debug_log["full_traceback"] = full_traceback
                self.episode_debug_log["robot"] = self.robot_config.robot
                with open(path, "w") as f:
                    f.write(json.dumps(self.episode_debug_log, indent=4))
                print("Observation out of bounds in step, debug log written to {}".format(path))
            done = True
            return self._get_obs() * 0., 0, done, {}
        if self.PROFILE:
            post_end = time.time()
            print(f"{self.uid} Post-step (ms) {post_start * 1000} -> {post_end * 1000} = {(post_end - post_start) * 1000}")

        if self.PROFILE:
            end = time.time()
            print(f"{self.uid} Env step (ms) {start_timer * 1000} -> {end * 1000} = {(end - start_timer) * 1000}") # PROFILE
        return observation, reward, done, info
    
    def _get_obs(self):
        # for the phase obs, we loop the phase index
        current_motion_mocap_len = self.current_motion_mocap.get_length() if self.current_motion_mocap is not None else 1
        current_motion_phase_idx = (self.current_motion_n_steps % current_motion_mocap_len) if self.current_motion_n_steps is not None else 0
        return get_obs(self.sim.data, self.model, current_motion_phase_idx, current_motion_mocap_len, self.current_player_action, self.ENV_CFG, self.robot_config)

    def viewer_setup(self):
        """ overrides MujocoEnv.viewer_setup """
        self.viewer.cam.trackbodyid = 1 # doesn't work
        self.viewer.cam.distance = self.model.stat.extent * 0.3
        self.viewer.cam.elevation = -20
        self.viewer.cam.lookat[2] = 0.6


    def render(self, mode=None): # needed by video rec
        output = mujoco_env.MujocoEnv.render(self, mode=mode)
        self.viewer.cam.lookat[0] = self.sim.data.qpos[0] # works for camera tracking
        if mode == "rgb_array":
            # add reward to the screen
            string = "{} {} {:>5} {:>7.2f}".format(self.current_motion_mocap.motion_name[-8:], self.debug_n_bad_angles, self.episode_length, self.episode_reward)
            # using opencv put_text:
            import cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            textmask = np.array(output)
            cv2.putText(textmask, string, (40, 40), font, 1., (255, 255, 255), 2, cv2.LINE_AA)
            output = textmask
        return output

    def change_to_motion(self, motion):
        if self.verbose:
            print("Changing to motion: {}".format(motion.motion_name))
        self.current_motion_mocap = motion
        self.current_motion_n_steps = 0


if __name__ == "__main__":
    env = DPCombinedEnv(verbose=1)
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
