from extracted_policy import ExtractedPolicy
from dp_env_v3 import DPEnv
import numpy as np

def log_actobs(qlog, obs, a, r, frame, time, env_name, max_frames, enabled, ZEROACT, ONEACT):
    if enabled:
        postfix = ""
        if ZEROACT:
            a = np.zeros_like(a)
            postfix = "zero_"
        if ONEACT:
            a = np.ones_like(a)
            postfix = "one_"
        qlog.append([len(obs)] + list(obs) + [len(a)] + list(a) + [1] + [r])
        if len(qlog) >= max_frames:
            qlog = np.array(qlog)
            path = "/home/daniel/{}_{}qlog.csv".format(env_name, postfix)
            np.savetxt(path, qlog, delimiter=",")
            print("Log written to ", path)
            raise ValueError("Logged")
            return

if __name__ == "__main__":
    env = DPEnv()
    pi = ExtractedPolicy()
    obs = env.reset()
    ep_rew = 0
    qlog = []
    for i in range(1000):
        env.render(mode="human")
        obs = obs[:66] # remove the last values that weren't present during training
        action = pi.act(obs)
        action = np.clip(action, -0.5, 0.5)
        log_actobs(qlog, obs, action, ep_rew, i, i * 0.1, env_name="deepmimic", max_frames=100, enabled=False, ZEROACT=False, ONEACT=False)
        obs, reward, done, info = env.step(action)
        ep_rew += reward
        if done:
            break
    print("Episode reward: ", ep_rew)