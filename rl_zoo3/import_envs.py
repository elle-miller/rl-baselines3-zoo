from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs_gymnasium
except ImportError:
    pass

try:
    import highway_env
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import custom_envs
except ImportError:
    pass

try:
    import gym_donkeycar
except ImportError:
    pass

try:
    import panda_gym
except ImportError:
    pass

try:
    import rocket_lander_gym
except ImportError:
    pass

try:
    import minigrid
except ImportError:
    pass


# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )

# added
from internal.tactile_testing.src.utils import HandManipulateEggWrapper

def create_inhand_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = HandManipulateEggWrapper(env)
        return env
    return make_env

env_id = "HandManipulateEgg_TouchGrid-v1"
register(
    id=env_id,
    entry_point=create_inhand_env(env_id),  # type: ignore[arg-type]
)

def _merge(a, b):
        a.update(b)
        return a

register(
    id=f"HandManipulateEgg-v1",
    entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
    kwargs=_merge(
        {
            "target_position": "random",
            "target_rotation": "xyz",
        },
    ),
    max_episode_steps=100,
)