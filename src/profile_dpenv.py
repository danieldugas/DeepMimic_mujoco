from deepmimic_env import DPEnv

if __name__ == "__main__":
    env = DPEnv(motion="getup_facedown_slow_FSI", robot="unitree_g1")
    env.reset()
    for i in range(10000):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        if done:
            env.reset()
        