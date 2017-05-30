from gym.envs.registration import register

register(
    id='robot-v0',
    entry_point='gym_robot.envs:RobotEnv',
    timestep_limit=1000,
)
