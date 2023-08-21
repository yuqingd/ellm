from .env import Env
from .text_env import BaseTextEnv
from .recorder import Recorder

try:
    import gym
    gym.register(
        id='CrafterReward-v1',
        entry_point='text_crafter.text_crafter:BaseTextEnv',
        max_episode_steps=10000,
        kwargs={'reward': True, 'env_reward': True, 'seed': 1})
    gym.register(
        id='CrafterNoReward-v1',
        entry_point='text_crafter.text_crafter:Env',
        max_episode_steps=10000,
        kwargs={'reward': False, 'env_reward': False, 'seed': 1})
    gym.register(
        id='CrafterTextEnv-v1',
        entry_point='text_crafter.text_crafter:BaseTextEnv',
        max_episode_steps=10000,
        kwargs={'reward': False, 'env_reward': False, 'seed': 1})
    gym.register(
        id='CrafterBaseTextEnv-v1',
        entry_point='text_crafter.text_crafter:BaseTextEnv',
        max_episode_steps=10000,
        kwargs={'reward': False, 'env_reward': False, 'seed': 1})
except ImportError:
    pass
