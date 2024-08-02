from extracted_policy import ExtractedPolicy
from deepmimic_env import DPEnv
import numpy as np
import os
from stable_baselines3 import PPO
import torch as th

if __name__ == "__main__":
    """ WOVEN GLADE """
    env = DPEnv(motion="walk", robot="unitree_g1")
    """ 
        MAX_EP_LENGTH = 1000
        VEL_OBS_SCALE = 0.1
        FRC_OBS_SCALE = 0.001
        ADD_FOOT_CONTACT_OBS = True
        ADD_TORSO_OBS = True
        ADD_JOINT_FORCE_OBS = False
        ADD_ABSPOS_OBS = False
        ADD_PHASE_OBS = True
        version = "v0.9HRS.no_abpos_10xactscale"
    """
    chkpt_path = os.path.expanduser("~/deep_mimic/woven-glade-55")
    # chkpt_path = os.path.expanduser("~/deep_mimic/walk_woven-glade-55_videos/walk_woven-glade-55_best")
    model = PPO.load(chkpt_path)
    obs = env.reset()
    obs = env.reset_model(idx_init=10)
    ep_rew = 0
    for i in range(1000):
        env.render(mode="human")
        obs_th = th.tensor(obs[None, :], dtype=th.float32)
        a_th, _, _ = model.policy.forward(obs_th, deterministic=True)
        a = a_th.detach().numpy()[0]
        obs, reward, done, info = env.step(a)
        ep_rew += reward
        if done:
            break
    print("Episode reward: ", ep_rew)