"""--robot.type=so101_mujoco: the B.L.U.S.H sim greenhouse as a lerobot robot.

Imported automatically by every lerobot command (lerobot-teleoperate, -record, -replay, -rollout,
async_inference.robot_client) because the package name starts with lerobot_robot_.
"""
from .config_so101_mujoco import SO101MujocoConfig
from .so101_mujoco import SO101Mujoco

__all__ = ["SO101Mujoco", "SO101MujocoConfig"]
