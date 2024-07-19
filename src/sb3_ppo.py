"""
pip install mujoco-py stable-baselines3 "cython<3" gym==0.15.4 pyquaternion shimmy tensorboard
"""
import os
import time

from dp_env_v3 import DPEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import MlpPolicy

from stable_baselines3.common.callbacks import BaseCallback

def eval_dashboard_rollout(model, eval_env, n, run_name):
    """ Collect an episode and plot it """
    from matplotlib import pyplot as plt
    import numpy as np
    import torch as th
    buffer = []
    obs = eval_env.reset()
    ep_rew = 0
    fig_paths = []
    fig_dir_name = os.path.expanduser("/tmp/sb3_eval_" + str(n) + "_" + run_name)
    while True:
        with th.no_grad():
            action, _states = model.predict(obs, deterministic=True)
            obs_tensor = th.as_tensor(obs.reshape((1, -1)), dtype=th.float32, device=model.device)
            actions, values, log_probs = model.policy(obs_tensor, deterministic=True)
            val = values.detach().cpu().numpy().flatten()[0]
        frame = eval_env.render(mode='rgb_array')
        obs, rewards, dones, info = eval_env.step(action)
        ep_rew += rewards
        buffer.append((obs, action, rewards, dones, info, ep_rew * 1., val, frame))
        if dones:
            break
    # make frames
    for i, (obs, action, rewards, dones, info, ep_rew, val, frame) in enumerate(buffer):
        rew_curve = [x[2] for x in buffer[:i+1]]
        ep_rew_curve = [x[5] for x in buffer[:i+1]]
        val_curve = [x[6] for x in buffer[:i+1]]
        # plot actions top left, frame top right, rewards bottom left, obs, bottom right
        fig, ax = plt.subplots(2, 2)
        action_range = np.arange(len(action))
        ax[0, 0].axhline(0, color='black', lw=1)
        for di in range(-5, 0):
            if i + di >= 0:
                ax[0, 0].step(action_range, buffer[i+di][1], where='mid', alpha=0.5, color='grey')
        ax[0, 0].step(action_range, action, where='mid')
        ax[0, 1].imshow(frame)
        ax[1, 0].axhline(0, color='black', lw=1)
        ax[1, 0].plot(ep_rew_curve)
        ax[1, 0].plot(rew_curve)
        ax[1, 0].plot(val_curve)
        ax[1, 1].axhline(0, color='black', lw=1)
        obs_range = np.arange(len(obs))
        for di in range(-5, 0):
            if i + di >= 0:
                ax[1, 1].step(obs_range, buffer[i+di][0], where='mid', alpha=0.5, color='grey')
        ax[1, 1].step(obs_range, obs, where='mid')
        # save the figure as a png
        fig_name = os.path.join(fig_dir_name, f"step_{i}.png")
        os.makedirs(fig_dir_name, exist_ok=True)
        fig.savefig(fig_name)
        fig_paths.append(fig_name)
    # assemble frames into a video
    import cv2
    import glob
    img_array = []
    for filename in fig_paths:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    video_dir = os.path.expanduser("~/deep_mimic/" + run_name + "_videos")
    video_path =  video_dir + '/global_step_{}.mp4'.format(n)
    os.makedirs(video_dir, exist_ok=True)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), 1, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Saved video to", video_path)

class EvalDashboardCallback(BaseCallback):
    def __init__(self, eval_env, run_name, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.run_name = run_name

    def _on_step(self) -> bool:
        n = self.num_timesteps
        model = self.model
        eval_env = self.eval_env
        if n % 10000 == 0 or n == 2:
            eval_dashboard_rollout(model, eval_env, n, self.run_name)
        return True

if __name__ == "__main__":

    M = 1000000
    # train a policy
    # hyperparams
    TOT = 100*M
    N_AG = 64
    HRZ = 128
    MINIB = 4
    EPOCHS = 12
    LR = 0.00025
    LOG_FREQ = 1*M // N_AG # log every 1M global steps
    class Run:
        name = "test" + time.strftime("%Y%m%d-%H%M_%S")
    run = Run()
    policy_kwargs = dict(net_arch=[256, 128])
    eval_env = DPEnv()
    envs = SubprocVecEnv([lambda: DPEnv() for i in range(N_AG)])
#     envs = VecVideoRecorder( envs, os.path.expanduser(f"~/wasm_flagrun/{run.name}_videos"), record_video_trigger=lambda x: x % (1*M // N_AG) == 0, video_length=1000,)
    model = PPO(MlpPolicy, envs, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=os.path.expanduser("~/wasm_flagrun/"),
                    n_steps=HRZ, learning_rate=LR)
    print("Begin Learn")
    print("-----------")
    model.learn(total_timesteps=100*M, tb_log_name=run.name, callback=EvalDashboardCallback(eval_env, run.name))
    model.save("~/wasm_flagrun/" + run.name)

    del model # remove to demonstrate saving and loading
