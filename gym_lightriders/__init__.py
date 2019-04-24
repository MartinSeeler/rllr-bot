from gym.envs.registration import register

register(
    id='LightRiders-v0',
    entry_point='gym_lightriders.envs:LightRidersEnv',
)