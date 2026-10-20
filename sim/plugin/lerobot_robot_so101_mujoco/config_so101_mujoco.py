from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class SO101MujocoConfig(RobotConfig):
    # First vine layout. Each new scene (key N in the MuJoCo window) takes the next seed.
    seed: int = 0
    # Open the MuJoCo window. Off for headless rollouts on a machine without a display.
    viewer: bool = True
    # Must stay false, like the real follower (CLAUDE.md rule): the sim speaks range-normalised units.
    use_degrees: bool = False
    # Same seatbelt as the real follower: caps how far one action may move a joint (normalised units).
    max_relative_target: float | None = None
    # Teleop: put the sim arm where the leader is on the first action of each scene, so it doesn't
    # swing across at full gain. Rollouts set this false: the policy starts from the sim's pose.
    snap_to_first_action: bool = True
    # Birdseye target box for the policy input, picked on the scene's first frame and held fixed,
    # like the real start-frame box (sim/target.py):
    #   "yolo"      the ripeness YOLO selects (ripeness env, sim/yolo_worker.py); the true box fills
    #               in when it misses or picks another fruit. What sim/scripts use.
    #   "true_box"  the sim's own target box, no YOLO
    #   "none"      raw camera images
    overlay: str = "none"
    # Full sim state per frame for offline re-renders (step 6). None = don't log.
    state_log_dir: str | None = "~/.cache/blush_sim/states"
