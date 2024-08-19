from extracted_policy import ExtractedPolicy
from deepmimic_env import DPEnv
import numpy as np
import os
from stable_baselines3 import PPO
import torch as th

from play_extracted import log_actobs

if __name__ == "__main__":
    """ POLAR BREEZE 65 """
    env = DPEnv(motion="run", robot="unitree_g1")
    """ 
        self.MAX_EP_LENGTH = 1000
        self.VEL_OBS_SCALE = 0.1
        self.FRC_OBS_SCALE = 0.001
        self.ADD_FOOT_CONTACT_OBS = True
        self.ADD_TORSO_OBS = True
        self.ADD_JOINT_FORCE_OBS = False
        self.ADD_ABSPOS_OBS = False
        self.ADD_PHASE_OBS = True
        version = "v0.9HRS.no_hands_20xact_mocapscale0.85_rplim60"
    """
    name = "run_polar_breeze"
    chkpt_path = os.path.expanduser("~/deep_mimic/run_polar-breeze-65_videos/run_polar-breeze-65_best")
    # chkpt_path = os.path.expanduser("~/deep_mimic/walk_woven-glade-55_videos/walk_woven-glade-55_best")
    assert name in chkpt_path.replace("-", "_")
    model = PPO.load(chkpt_path)
    obs = env.reset()
    idx_init = 20
    obs = env.reset_model(idx_init=idx_init)
    print("let init_qpos = [", ", ".join(["{:.5f}".format(n) for n in env.sim.data.qpos]), "];")
    print("let init_qvel = [", ", ".join(["{:.5f}".format(n) for n in env.sim.data.qvel]), "];")
    print("this.mocap_start_frame = ", idx_init, ";")
    print("this.mocap_len = ", env.mocap.get_length(), ";")
    ep_rew = 0
    qlog = []
    for i in range(1000):
        env.render(mode="human")
        obs_th = th.tensor(obs[None, :], dtype=th.float32)
        a_th, _, _ = model.policy.forward(obs_th, deterministic=True)
        a = a_th.detach().numpy()[0]
        a = log_actobs(qlog, obs, a, ep_rew, i, env.get_time(), env_name=f"deepmimic_{name}_f{idx_init}", max_frames=100, enabled=False, ZEROACT=False, ONEACT=False)
        obs, reward, done, info = env.step(a)
        ep_rew += reward
        if done:
            break
    print("Episode reward: ", ep_rew)
    assert ep_rew > 90, "Run Polar Breeze is broken"