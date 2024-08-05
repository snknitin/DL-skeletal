# Register the environment
from gym.envs.registration import register

# register(
#     id='MultiSKU-v0',
#     entry_point='src.gymenv.RL_env:MultiSKUEnvironment',
#     kwargs={}
# )

from src.gymenv.dummy_env import DummyEnv

register(
    id='DummyEnv-v0',
    entry_point='src.gymenv.dummy_env:DummyEnv',
    kwargs={}
)