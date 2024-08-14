from extracted_policy import ExtractedPolicy
from deepmimic_env import DPEnv
import numpy as np
import os
from stable_baselines3 import PPO
import torch as th

if __name__ == "__main__":
    """ PRETTY TOTEM 70 """
    env = DPEnv(motion="getup_facedown_slow_FSI", robot="unitree_g1")
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
    chkpt_path = os.path.expanduser("~/deep_mimic/getup_facedown_slow_FSI_pretty-totem-70_videos/getup_facedown_slow_FSI_pretty-totem-70_best")
    # chkpt_path = os.path.expanduser("~/deep_mimic/walk_woven-glade-55_videos/walk_woven-glade-55_best")
    model = PPO.load(chkpt_path)
    idx_init = 0
    obs = env.reset()
    obs = env.reset_model(idx_init=idx_init)
    print("let init_qpos = [", ", ".join(["{:.5f}".format(n) for n in env.sim.data.qpos]), "];")
    print("let init_qvel = [", ", ".join(["{:.5f}".format(n) for n in env.sim.data.qvel]), "];")
    print("this.mocap_start_frame = ", idx_init, ";")
    print("this.mocap_len = ", env.mocap.get_length(), ";")
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