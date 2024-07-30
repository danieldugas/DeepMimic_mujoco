"""
pip install mujoco-py stable-baselines3 "cython<3" gym==0.15.4 pyquaternion shimmy tensorboard py3dtf
"""
import os
import time
import wandb
# this stops a strange bug where after hours of training tkinter causes a crash in the workers
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

from deepmimic_env import DPEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import MlpPolicy

from stable_baselines3.common.callbacks import BaseCallback


def eval_dashboard_rollout(model, eval_env, n, run_name, log_wandb=True):
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
        try:
            with th.no_grad():
                action, _states = model.predict(obs, deterministic=True)
                obs_tensor = th.as_tensor(obs.reshape((1, -1)), dtype=th.float32, device=model.device)
                actions, values, log_probs = model.policy(obs_tensor, deterministic=True)
                val = values.detach().cpu().numpy().flatten()[0]
        except:
            val = 0
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
        fig, ax = plt.subplots(2, 2, num="eval")
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
        plt.close()
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
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24 if len(img_array) > 10 else 1, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Saved video to", video_path)
    # append length and reward to log
    ep_len = len(buffer)
    log_path = video_dir + '/log.csv'
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("global_step,ep_len,ep_rew\n")
    with open(log_path, 'a') as f:
        f.write(f"{n},{ep_len},{ep_rew}\n")
    print("Logged to", log_path)
    # load logfile, plot, and save plot
    # use numpy to load the csv
    plot_path = video_dir + '/rew_plot.png'
    log = np.loadtxt(log_path, delimiter=',', skiprows=1)
    if len(log.shape) == 1: # only one item, nothing to plot
        log = log.reshape((1, -1))
    fig, ax = plt.subplots(1, 1)
    ax.plot(log[:, 0], log[:, 2])
    ax.set_xlabel("Global Step")
    fig.savefig(plot_path)
    plt.close()
    fig, ax = plt.subplots(1, 1)
    len_plot_path = video_dir + '/len_plot.png'
    ax.plot(log[:, 0], log[:, 1])
    ax.set_xlabel("Global Step")
    fig.savefig(len_plot_path)
    plt.close()
    # log to wandb
    if log_wandb:
        wandb.log({
            "eval_episode_length": ep_len,
            "eval_episode_reward": ep_rew,
            "eval_global_step": n,
            "eval_best_episode_reward": np.max(log[:, 2]),
            "eval_best_episode_global_step": int(log[np.argmax(log[:, 2]), 0]),
        })
    # save model if best
    if np.max(log[:, 2]) == log[-1, 2]:
        model.save(video_dir + "/" + run_name + "_best")
    # print ep len and reward
    print("Eval: LEN {}, EP_REW {}".format(ep_len, ep_rew))


class EvalDashboardCallback(BaseCallback):
    def __init__(self, eval_env, run_name, log_wandb=True, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.run_name = run_name
        self.log_wandb = log_wandb

    def _on_step(self) -> bool:
        n = self.num_timesteps
        model = self.model
        eval_env = self.eval_env
        n_agents = self.num_timesteps // self.n_calls
        if self.n_calls % (500000 // n_agents) == 0 or self.n_calls == 1:
            eval_dashboard_rollout(model, eval_env, n, self.run_name, log_wandb=self.log_wandb)
        return True

def parse_reason(required=True):
    import sys
    reason = ""
    if len(sys.argv) > 1:
        reason = sys.argv[1]
    if len(sys.argv) > 2:
        raise ValueError("Too many arguments")
    print("Reason: " + reason)
    if reason == "" and required:
        raise ValueError("Please provide a reason for this run")
    return reason

if __name__ == "__main__":
    DBG_NO_WANDB = False
    reason = parse_reason(required=not DBG_NO_WANDB)
    motion = "getup_facedown"
    task = ""
    M = 1000000
    # train a policy
    # hyperparams
    TOT = 100*M
    N_AG = 16
    HRZ = 256
    MINIB = 512
    EPOCHS = 20
    LR = 0.0004
    LOG_FREQ = 1*M // N_AG # log every 1M global steps
    policy_kwargs = dict(net_arch=[256, 128])
    # run info
    class Run:
        name = "test" + time.strftime("%Y%m%d-%H%M_%S")
    run = Run()
    batch_size = HRZ * N_AG
    minibatch_size = 256
    config = {
        "run_reason": reason,
        "policy_type": "MlpPolicy",
        "total_timesteps": TOT,
        "env_name": "deep_mimic_mujoco",
        "motion": motion,
        "task": task,
        "version": DPEnv.version,
        "env_cfg": DPEnv.CFG.__dict__.copy(),
        "arch": policy_kwargs["net_arch"],
        "n_agents": N_AG,
        "horizon": HRZ,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "learning_rate": LR, 
        "epochs": EPOCHS,
        "machine_name": os.environ.get("MACHINE_NAME", "unknown"),
    }
    if not DBG_NO_WANDB:
        wandb.login()
        run = wandb.init(
            project="deep_mimic",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
    eval_env = DPEnv(motion=motion)
    envs = SubprocVecEnv([lambda: DPEnv(motion=motion) for i in range(N_AG)])
    model = PPO(MlpPolicy, envs, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=os.path.expanduser("~/tensorboard/"),
                    n_steps=HRZ, learning_rate=LR, n_epochs=EPOCHS, batch_size=minibatch_size)
    print("Begin Learn")
    print("-----------")
    model.learn(total_timesteps=100*M, tb_log_name=run.name, callback=EvalDashboardCallback(
        eval_env, motion + task + "_" + run.name, log_wandb=not DBG_NO_WANDB))
    model.save(os.path.expanduser("~/deep_mimic/" + run.name))

    del model # remove to demonstrate saving and loading

requirements = """
absl-py==2.1.0
cachetools==5.4.0
certifi==2024.7.4
cffi==1.15.1
charset-normalizer==3.3.2
cloudpickle==1.2.2
cycler==0.11.0
Cython==0.29.37
Farama-Notifications==0.0.4
fasteners==0.19
fonttools==4.38.0
future==1.0.0
glfw==2.7.0
google-auth==2.32.0
google-auth-oauthlib==0.4.6
grpcio==1.62.2
gym==0.15.4
gymnasium==0.28.1
idna==3.7
imageio==2.31.2
importlib-metadata==6.7.0
jax-jumpy==1.0.0
kiwisolver==1.4.5
Markdown==3.4.4
MarkupSafe==2.1.5
matplotlib==3.5.3
mujoco-py==2.1.2.14
numpy==1.21.6
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
oauthlib==3.2.2
opencv-python==4.10.0.84
packaging==24.0
pandas==1.3.5
Pillow==9.5.0
protobuf==3.20.3
py3dtf==0.2
pyasn1==0.5.1
pyasn1-modules==0.3.0
pycparser==2.21
pyglet==1.3.2
pyparsing==3.1.2
pyquaternion==0.9.9
python-dateutil==2.9.0.post0
pytz==2024.1
requests==2.31.0
requests-oauthlib==2.0.0
rsa==4.9
scipy==1.7.3
Shimmy==1.1.0
six==1.16.0
stable-baselines3==2.0.0
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
torch==1.13.1
typing_extensions==4.7.1
urllib3==2.0.7
Werkzeug==2.2.3
zipp==3.15.0
"""
