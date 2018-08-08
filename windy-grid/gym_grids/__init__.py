
from gym.envs.registration import register
register(
    id='Windy-Grid-v0',
    entry_point='gym_grids.env.windygrid:WindyGridWorldEnv',
    max_episode_steps=10000000000,
    reward_threshold=10000000000,
)