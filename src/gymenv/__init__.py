# Register the environment
from gym.envs.registration import register

register(
    id='MultiSKU-v0',
    entry_point='src.gymenv.RL_env:MultiSKUEnvironment',
    kwargs={}
)