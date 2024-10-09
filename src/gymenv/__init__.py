# Register the environment
from gym.envs.registration import register

# Register the environment
from gym.envs.registration import register
# from src.gymenv.single_env import SingleFCEnvironment
from src.gymenv.multi_env import MultiFCEnvironment

# register(
#     id='SingleFC-v0',
#     entry_point='src.gymenv.single_env:SingleFCEnvironment',
#     kwargs={}
# )

register(
    id='MultiFC_OT-v0',
    entry_point='src.gymenv.multi_env:MultiFCEnvironment',
    kwargs={}
)
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