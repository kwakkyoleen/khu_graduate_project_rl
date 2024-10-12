import gymnasium as gym
from .obstacle_env import ObstacleEnv, ObstacleEnvCfg

gym.register(
    id="Obstacle-direct-v0",
    entry_point="mydofbot.envcfg:ObstacleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ObstacleEnvCfg,
    },
)
