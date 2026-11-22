#!/usr/bin/env python3
"""Every krish5831 dataset onto the Hub, PRIVATE, under one naming convention (K7 2026-11-16: "the older datasets
also put them on the hub and then delete locally and also keep a strict and good naming convention").

    krish5831/<module>_<world>_<source>_v<N>[_<subset>][_<YYYYMMDDhhmm>]
      module  blush | shear | arm (an SO-101 test that belongs to no module)
      world   sim | real
      source  teacher (closed-loop scripted teacher) | scripted (open-loop scripted expert)
              | teleop (a human, leader arm)
      vN      the version of that source (teacher v3, the v2 truss planner, the v1 / v2 camera rig)
      subset  only for a partial or special set: pilot | onelayout | onerow | markercup ...
      time    only for one-off real recordings: when the recording started

    python hub_datasets.py plan        what would happen, nothing changed
    python hub_datasets.py run         rename / upload, then verify every file by size + hash
    python hub_datasets.py verify      verify only (local copies that still exist)
    python hub_datasets.py run NAME..  only these (old names)
Deletes nothing: local copies go only after "VERIFIED" for that dataset, by hand.
"""
import hashlib
import pathlib
import sys

from huggingface_hub import DatasetCard, HfApi

U = "krish5831"
L = pathlib.Path.home() / ".cache/huggingface/lerobot" / U
api = HfApi()

# old name -> new name. Same name = keep. Each row: (old, new, what it is)
MAP = [
    ("blush_sim_teacher_v3", "blush_sim_teacher_v3", "B.L.U.S.H step 2: 610 teacher-v3 episodes (525 cuts, 85 holds)"),
    ("blush_sim_teacher_v3_pilot", "blush_sim_teacher_v3_pilot", "B.L.U.S.H: 5-episode pilot of teacher v3"),
    ("blush_sim_teacher_onelayout_v1", "blush_sim_teacher_v3_onelayout",
     "B.L.U.S.H step 1: 50 noisy teacher-v3 episodes on one layout (seed 9100009)"),
    ("blush_sim_truss_v2", "blush_sim_scripted_v2",
     "B.L.U.S.H: 200 episodes of the v2 scripted truss planner (SmolVLA era)"),
    ("shear_sim_teacher_v3", "shear_sim_teacher_v3", "S.H.E.A.R step 2: 600 teacher-v3 episodes (540 cuts, 60 holds)"),
    ("shear_sim_teacher_onerow_v1", "shear_sim_teacher_v3_onerow",
     "S.H.E.A.R step 1: 50 noisy teacher-v3 episodes on one row (910057)"),
    ("shear_sim_scripted_v1", "shear_sim_scripted_v1",
     "S.H.E.A.R: 200 episodes of the v1 scripted expert (SmolVLA era)"),
    ("marker_test_20260920_201308", "arm_real_teleop_v1_markercup_202609202013",
     "Real SO-101, leader-arm teleop: grab the marker, put it in the cup (camera rig v1)"),
    ("marker_test_20260920_213650", "arm_real_teleop_v1_markercup_202609202136",
     "Real SO-101, leader-arm teleop: grab the marker, put it in the cup (camera rig v1)"),
    ("marker_test_new_camera_setup_20260920_221944", "arm_real_teleop_v2_markercup_202609202219",
     "Real SO-101, leader-arm teleop: grab the marker, put it in the cup (new camera rig, v2)"),
    ("blush_sim_teacher_v3_finisher_pilot", "blush_sim_teacher_v3_finisher_pilot",
     "B.L.U.S.H: 10-episode pilot of the last-stage finisher's data. The step-3 ACT policy (100k) approaches; at "
     "20-50 mm from the cue's cut point teacher v3 takes over (TrussTeacher.rejoin, B.L.U.S.H dev 020c8fe; its normal "
     "plan is unchanged from v3) and finishes; recorded from the hand-over on, kept only if the whole scene passed "
     "generate_truss.judge (eval_truss.py --record, dev f88dc77). Seeds 9890000-9890027"),
    ("blush_sim_teacher_v3_finisher", "blush_sim_teacher_v3_finisher",
     "B.L.U.S.H: the last-stage finisher's dataset, 201 episodes (78357 frames, 30 fps, base + claw cameras). Same "
     "recipe as the pilot: the step-3 ACT policy (act_teacher_v3, 100k) approaches; at 20-50 mm from the cue's cut "
     "point teacher v3 takes over (TrussTeacher.rejoin) and finishes; recorded from the hand-over on, kept only if the "
     "whole scene passed generate_truss.judge (eval_truss.py --hybrid --rejoin --record, B.L.U.S.H dev 0fba9b1, before "
     "the add_self_check leak fix, which changes no plan). Merged from 33 parts (merge_parts.py, 2026-11-17). Seeds "
     "from 9800000, 9830000, 9860000 (parts p00-p02) and 9870000 + 500 j (p03-p32), all inside 9800000-9899999"),
    ("blush_sim_teacher_v5", "blush_sim_teacher_v5",
     "B.L.U.S.H: 800 teacher-v5 episodes in the new crate world (crate 23x28 cm), 736 cuts + 64 holds, 355 with "
     "DART noise on the executed action (labels stay the teacher's clean ones), 383877 frames, 30 fps, base + claw "
     "cameras. generate_truss.py --teacher --no-dr --no-yolo --balance --hold-share 0.10 --noise-share 0.5 "
     "--retries 6 (B.L.U.S.H dev 2efd61b: plans chosen on clean dry runs, strain filter, cue at the plan's cut_u). "
     "Teacher v5 passes 163/200 untouched gate scenes (82 %); only passing episodes are kept. Merged from 32 parts "
     "of 25 (merge_parts.py, 2026-11-19); seeds 10000000 + 1000 P + k, P = 0..31. replay_check.py: 5/5 clean "
     "episodes reproduce, 5/5 noisy ones pass on their labels"),
]


def exists(name):
    try:
        return api.dataset_info(f"{U}/{name}")
    except Exception:
        return None


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def blob(p):
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(name, local):
    info = api.dataset_info(f"{U}/{name}", files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    files = [p for p in sorted(local.rglob("*"))
             if p.is_file() and not ({"images", ".cache"} & set(p.relative_to(local).parts))]
    bad = []
    for p in files:
        rel, s = str(p.relative_to(local)), remote.get(str(p.relative_to(local)))
        if s is None:
            bad.append(f"missing on the Hub: {rel}")
        elif s.size != p.stat().st_size:
            bad.append(f"size differs: {rel}")
        elif (s.lfs.sha256 != sha256(p)) if s.lfs is not None else (s.blob_id != blob(p)):
            bad.append(f"hash differs: {rel}")
    ok = not bad and info.private is True
    print(f"   {U}/{name}: private={info.private}, {len(files)} local files, {len(remote)} on the Hub -> "
          f"{'VERIFIED' if ok else 'NOT VERIFIED'}", flush=True)
    for b in bad[:8]:
        print("      ✗", b, flush=True)
    return ok


def card(name, old, what):
    c = DatasetCard(f"---\nlicense: other\ntags:\n- LeRobot\n- vinea\n---\n# {U}/{name}\n\n{what}.\n\n"
                    + (f"Formerly `{U}/{old}`.\n\n" if old != name else "")
                    + "Naming: `<module>_<world>_<source>_v<N>[_<subset>][_<YYYYMMDDhhmm>]`. "
                    "Private, not for release.\n")
    c.push_to_hub(f"{U}/{name}", repo_type="dataset")


mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
only = set(sys.argv[2:])                                # optional: just these old names
allok = True
for old, new, what in MAP:
    if only and old not in only:
        continue
    local = L / old
    has_local = (local / "meta/info.json").exists() and (local / "data").exists()
    hub_old, hub_new = exists(old), exists(new) if new != old else None
    if new != old and hub_new is not None:
        step = "already under the new name"
    elif hub_old is not None:
        step = "keep the name" if new == old else f"rename on the Hub {old} -> {new}"
    elif has_local:
        step = f"upload the local copy as {new}"
    else:
        step = "NOT FOUND (no local copy, not on the Hub)"
    hub = hub_old or hub_new
    print(f"{old} -> {new}: {step}; local copy {'yes' if has_local else 'no'}; "
          f"{'private' if hub and hub.private else 'public' if hub else '-'}",
          flush=True)
    if mode != "run":
        continue
    if step.startswith("rename"):
        api.move_repo(f"{U}/{old}", f"{U}/{new}", repo_type="dataset")
    elif step.startswith("upload"):
        api.create_repo(f"{U}/{new}", repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=f"{U}/{new}", folder_path=str(local), repo_type="dataset",
                          ignore_patterns=["images/", ".cache/"])
    if not step.startswith("NOT FOUND"):
        api.update_repo_settings(f"{U}/{new}", private=True, repo_type="dataset")
        card(new, old, what)
        try:
            v = __import__("json").load(open(local / "meta/info.json"))["codebase_version"] if has_local else "v3.0"
            api.create_tag(f"{U}/{new}", tag=v, repo_type="dataset", exist_ok=True)
        except Exception as e:                          # a tag that exists already is fine
            print(f"   (tag: {type(e).__name__})", flush=True)
    if mode in ("run", "verify") and has_local:
        allok &= verify(new, local)
if mode == "verify":
    for old, new, _what in MAP:
        if (L / old / "data").exists():
            allok &= verify(new, L / old)
print("ALL VERIFIED" if allok and mode != "plan" else "done" if mode == "plan" else "SOMETHING NOT VERIFIED",
      flush=True)
