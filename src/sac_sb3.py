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

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from sb3_ppo import EvalDashboardCallback, parse_reason

if __name__ == "__main__":
    DBG_NO_WANDB = False
    reason = parse_reason(required=not DBG_NO_WANDB)
    motion = "walk"
    task = ""
    robot = "unitree_g1"
    M = 1000000
    # train a policy
    # hyperparams
    TOT = 100*M
    N_AG = 32
    HRZ = 4096
    MINIB = 512
    EPOCHS = 20
    LR = 0.0004
    LOG_FREQ = 1*M // N_AG # log every 1M global steps
    BUFFER_SIZE = 1*M
    policy_kwargs = dict(net_arch=[1024, 512])
    # run info
    class Run:
        name = "test" + time.strftime("%Y%m%d-%H%M_%S")
    run = Run()
    batch_size = HRZ * N_AG
    minibatch_size = batch_size // MINIB
    config = {
        "run_reason": reason,
        "policy_type": "MlpPolicy",
        "total_timesteps": TOT,
        "env_name": "deep_mimic_mujoco",
        "algorithm": "SAC",
        "motion": motion,
        "task": task,
        "robot": robot,
        "version": DPEnv.version,
        "env_cfg": DPEnv.CFG.__dict__.copy(),
        "arch": policy_kwargs["net_arch"],
        "n_agents": N_AG,
        "horizon": HRZ,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "learning_rate": LR, 
        "epochs": EPOCHS,
        "buffer_size": BUFFER_SIZE,
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
    eval_env = DPEnv(motion=motion, robot=robot)
    # env = DPEnv(motion=motion)
    envs = SubprocVecEnv([lambda: DPEnv(motion=motion, robot=robot) for i in range(N_AG)])
    model = SAC(MlpPolicy, envs, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=os.path.expanduser("~/tensorboard/"),
                    buffer_size=BUFFER_SIZE,
                    # n_steps=HRZ, learning_rate=LR, n_epochs=EPOCHS, batch_size=minibatch_size,
                )
    print("Begin Learn")
    print("-----------")
    model.learn(total_timesteps=100*M, tb_log_name=run.name, callback=EvalDashboardCallback(eval_env, motion + task + "_" + run.name))
    model.save(os.path.expanduser("~/deep_mimic/" + run.name))

    del model # remove to demonstrate saving and loading
