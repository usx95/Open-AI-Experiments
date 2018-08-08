from gym.envs.registration import register

register(
    id='Windy-Grid-v0',
    entry_point='gym_grids.env.windygrid:WindyGridWorldEnv',
    max_episode_steps=100000,
    reward_threshold=10000000000,
    kwargs={'rows': 10, 'col': 20, 'standard': 1, 'max_wind': 2}
)
