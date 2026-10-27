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
    # What gets drawn into the camera frames (harvest/yolo_overlay.py):
    #   "yolo"      the YOLO view, every frame: red boxes + class + confidence on every detection on
    #               both cameras, and the green target box on the birdseye, picked among the ripe
    #               fruits on the scene's first frame and then held (sim/target.py). The true box fills
    #               in when YOLO misses. YOLO runs in the ripeness env (sim/yolo_worker.py). What
    #               sim/scripts use.
    #   "true_box"  only the green target box, from the sim's truth, no YOLO
    #   "none"      raw camera images
    overlay: str = "none"
    # Full sim state per frame for offline re-renders (step 6). None = don't log.
    state_log_dir: str | None = "~/.cache/blush_sim/states"
    # True (teleop, rollouts): physics runs in its own thread at wall-clock speed, like a real arm.
    # False (scripted episodes, sim/generate_scripted.py): no thread; every send_action advances the
    # sim exactly 1/fps, so an episode is deterministic and never time-compressed by a slow frame.
    realtime: bool = True
    fps: int = 30                    # realtime=False only: sim time per send_action = 1/fps
    # Randomise lighting, colours (within ripeness class), camera poses and the physics guesses per
    # scene, drawn from the scene seed (sim/randomize.py). Off = the nominal scene.
    domain_rand: bool = False
    # Sim seconds to let the vines settle (arm held at the start pose) before a scene's first frame.
    settle_s: float = 0.0
