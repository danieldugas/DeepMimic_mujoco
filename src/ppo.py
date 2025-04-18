"""
pip install mujoco-py pyquaternion joblib stable-baselines==2.10.2 gym==0.15.4 torch tensorflow==1.13.1
pip install "cython<3"
pip install protobuf==3.19.6
"""

import os
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines import PPO2

from deepmimic_env import DPEnv

if __name__ == "__main__":

    M = 1000000
    # train a policy
    # hyperparams
    TOT = 100*M
    N_AG = 32
    HRZ = 128
    MINIB = 4
    EPOCHS = 12
    LR = 0.00025
    LOG_FREQ = 1*M // N_AG # log every 1M global steps
    class Run:
        name = "test" + time.strftime("%Y%m%d-%H%M_%S")
    run = Run()
    policy_kwargs = dict(net_arch=[256, 128])
    envs = SubprocVecEnv([lambda: DPEnv() for i in range(N_AG)])
    envs = VecVideoRecorder( envs, os.path.expanduser(f"~/wasm_flagrun/{run.name}_videos"), record_video_trigger=lambda x: x % (1*M // N_AG) == 0, video_length=1000,)
    model = PPO2(MlpPolicy, envs, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=os.path.expanduser("~/wasm_flagrun/"),
                    n_steps=HRZ, nminibatches=MINIB, noptepochs=EPOCHS, learning_rate=LR)
    print("Begin Learn")
    print("-----------")
    model.learn(total_timesteps=100*M, tb_log_name=run.name)
    model.save("~/wasm_flagrun/" + run.name)

    del model # remove to demonstrate saving and loading
