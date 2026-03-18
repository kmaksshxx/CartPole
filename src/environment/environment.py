import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
env.reset()
done = False

while not done:
    s, a, done, _, _ = env.step(1)

env.close()
