from combined_env import DPCombinedEnv

if __name__ == "__main__":
    env = DPCombinedEnv()
    env.reset()
    for i in range(10000):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        if done:
            env.reset()
        