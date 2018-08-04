import gym
import numpy as np
target_timestep = 40000


def run_episode(env, parameters, render=False):
    observation = env.reset()
    total_reward = 0
    for _ in range(target_timestep):
        if render:
            env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def setup():
    e = gym.make('CartPole-v0')
    e._max_episode_steps = target_timestep
    e._reward_threshold = 1000000
    return e


env = setup()
best_params = None
best_reward = 0
for _ in range(10000):
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env, parameters)
    if reward > best_reward:
        best_reward = reward
        best_params = parameters
        # considered solved if the agent lasts 200 timesteps
    print(_, reward, best_reward)
    if reward == target_timestep:
        break
run_episode(env, best_params, True)
