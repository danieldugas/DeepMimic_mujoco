from deepmimic_env import DPEnv
import json
import numpy as np
from tqdm import tqdm
import time
from matplotlib import pyplot as plt

if __name__ == "__main__":
    log_path = "/tmp/deepmimic_episode_20240731-1158_31.json"
    with open(log_path, "r") as f:
        log = f.read()
    json_dict = json.loads(log)
    motion = json_dict["motion"]
    robot = json_dict["robot"]

    print(json_dict["full_traceback"])
    qpos_log = json_dict["qpos"]
    qvel_log = json_dict["qvel"]

    env = DPEnv(motion=motion, robot=robot)
    env.reset_model(idx_init=0)
    # Play episode
    # -----------
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    for qpos, qvel in tqdm(zip(qpos_log, qvel_log), total=len(qpos_log)):
        # obs, rew, done, info = env.step(ac, force_state=(np.array(qpos), np.array(qvel)))
        env.set_state(np.array(qpos), np.array(qvel))
        env.render(mode="human")
        time.sleep(0.1)

    action_log = np.array(json_dict["action"])
    xvel_log = np.array(json_dict["body_xvelp"])
    # plot actions
    plt.figure()
    for i in range(len(action_log[0])):
        plt.plot(action_log[:, i], label=f"action_{i}")
    plt.legend()
    # vels
    plt.figure()
    for i in range(len(xvel_log[0])):
        plt.plot(xvel_log[:, i], label=f"xvel_{i}")
    plt.legend()
    plt.show()
    


