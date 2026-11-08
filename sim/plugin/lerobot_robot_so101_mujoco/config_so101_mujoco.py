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
    # What gets drawn into the camera frames (harvest/yolo_overlay.py). The harvest is always whole trusses
    # (sim/truss_scene.py, K7 2026-11-03; the single-fruit task was removed 2026-11-08):
    #   "yolo"      the truss YOLO (yolo_weights, required) picks the target: on rig "birdseye" its view is drawn
    #               every frame (red boxes + class + confidence on both cameras) plus the green target box and the
    #               cut-point cross, picked among the ready trusses on the scene's first frame and then held
    #               (sim/truss_target.py). The true box fills in when YOLO misses. YOLO runs in the ripeness env
    #               (sim/yolo_worker.py).
    #   "true_box"  only the green target box + cut-point cross, from the sim's truth, no YOLO
    #   "none"      raw camera images
    overlay: str = "none"
    # Full sim state per frame for offline re-renders (step 6). None = don't log.
    state_log_dir: str | None = "~/.cache/blush_sim/states"
    # True (teleop, rollouts): physics runs in its own thread at wall-clock speed, like a real arm.
    # False (scripted episodes, sim/generate_truss.py): no thread; every send_action advances the
    # sim exactly 1/fps, so an episode is deterministic and never time-compressed by a slow frame.
    realtime: bool = True
    fps: int = 30                    # realtime=False only: sim time per send_action = 1/fps
    # Randomise lighting, colours (within ripeness class), camera poses and the physics guesses per
    # scene, drawn from the scene seed (sim/randomize.py). Off = the nominal scene.
    domain_rand: bool = False
    # Sim seconds to let the vines settle (arm held at the start pose) before a scene's first frame.
    settle_s: float = 0.0
    # overlay="yolo": the truss YOLO's weights, e.g. runs/yolo_truss/truss_v1/weights/best.pt (sim/yolo_truss/train.sh;
    # runs/ is gitignored, so there's no default). overlay="yolo" without it is refused.
    yolo_weights: str | None = None
    # Which cameras the policy gets (sim/cameras.py RIG_CAMERAS):
    #   "birdseye"  birdseye + claw, every dataset up to blush_sim_truss_v1
    #   "base"      base + claw (K7 2026-11-06): the camera low on a bracket left of the arm's base. The target is still
    #               picked on the birdseye's first frame (the truss YOLO was trained on birdseye views); its box + cut
    #               point go onto the base camera from the sim's truth. The claw camera stays clean (SockBot: a box on
    #               the wrist camera is noise).
    rig: str = "birdseye"
    # Each scene starts this many degrees (uniform, per arm joint, drawn from the seed) off the start pose, so a policy
    # learns "see the truss, then reach" rather than one fixed first move (SockBot's ghost-sock lesson). 0 = exact.
    start_jitter_deg: float = 0.0
