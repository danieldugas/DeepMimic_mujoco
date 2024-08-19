from extracted_policy import ExtractedPolicy
from deepmimic_env import DPEnv
import numpy as np
import os

def log_actobs(qlog, obs, a, r, frame, time, env_name, max_frames, enabled, ZEROACT, ONEACT, folder=None):
    if enabled:
        if folder is None:
            folder = os.path.expanduser("~/deep_mimic/qlogs")
        postfix = ""
        if ZEROACT:
            a = np.zeros_like(a)
            postfix = "zero_"
        if ONEACT:
            a = np.ones_like(a)
            postfix = "one_"
        qlog.append([time] + [len(obs)] + list(obs) + [len(a)] + list(a) + [1] + [r])
        if len(qlog) >= max_frames:
            qlog = np.array(qlog)
            path = os.path.join(folder, "{}_{}qlog.csv".format(env_name, postfix))
            np.savetxt(path, qlog, delimiter=",")
            print("Log written to ", path)
            raise ValueError("Logged")
            return
    return a

if __name__ == "__main__":
    env = DPEnv()
    pi = ExtractedPolicy()
    obs = env.reset()
    obs = env.reset_model(idx_init=14)
    ep_rew = 0
    qlog = []
    for i in range(1000):
        env.render(mode="human")
        obs = obs[:66] # remove the last values that weren't present during training
        action = pi.act(obs)
        action = np.clip(action, -0.5, 0.5)
        log_actobs(qlog, obs, action, ep_rew, i, env.get_time(), env_name="deepmimic", max_frames=100, enabled=False, ZEROACT=False, ONEACT=False)
        obs, reward, done, info = env.step(action)
        ep_rew += reward
        if done:
            break
    print("Episode reward: ", ep_rew)