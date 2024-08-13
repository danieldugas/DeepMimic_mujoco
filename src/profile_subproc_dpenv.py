from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from deepmimic_env import DPEnv
import numpy as np
import time

if __name__ == "__main__":
    motion = "run"
    robot = "unitree_g1"
    N = 8
    env = SubprocVecEnv([lambda: DPEnv(robot=robot, motion=motion, _profile=True) for _ in range(N)])
    env.reset()
    sum_step_time_ms = 0
    max_step_time_ms = 0
    for i in range(1000):
        ac = np.random.randn(N, env.action_space.shape[0])
        print("Step: ", i)
        vec_start = time.time()
        obs, reward, done, info = env.step(ac)
        vec_end = time.time()
        print(f"000 Vec step (ms) {vec_start * 1000} -> {vec_end * 1000} = {(vec_end - vec_start) * 1000}")
        sum_step_time_ms += (vec_end - vec_start) * 1000
        avg_step_time_ms = sum_step_time_ms / (i + 1)
        max_step_time_ms = max(max_step_time_ms, (vec_end - vec_start) * 1000)
        print("----")
        print(f"Average step time (ms): {avg_step_time_ms}ms | max step time (ms): {max_step_time_ms}ms")