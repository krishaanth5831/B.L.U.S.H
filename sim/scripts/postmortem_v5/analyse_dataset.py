#!/usr/bin/env python3
"""Post-mortem 2026-11-19: what the teacher-v5 dataset itself says, no simulator stepping and no policy.

    conda activate lerobot && python analyse_dataset.py            (run from anywhere; ~1 min, CPU)

A. Teacher consistency: for pairs of clean episodes whose target cue is nearly the same (same slot, cut pixel within
   PAIR_PX), how far apart are the teacher's jaw positions at the same clock time, its closing times, and its joint
   angles at the cut? A policy that outputs ONE answer per input can at best sit between them.
B. Cue sufficiency: how well do the 6 cue numbers (box + cut pixel in the base camera) pin down the 3D point the
   teacher cut at? Leave-one-out nearest-neighbour regression; the error split along / across the camera's ray.
C. Schedule: when does the teacher close the jaw, and how much does that time vary?
D. Slots: how tightly do the cut points cluster per slot (a policy that only learned "which slot" would miss by this).

Writes analyse_dataset.json next to this file and prints a summary.
"""
import collections
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE.parents[2] / "sim"
OUT = HERE.parents[2] / "outputs" / "notes" / "2026-11-19_v5_postmortem"   # generated files (git-ignored)
sys.path.insert(0, str(SIM))
import arm_map  # noqa: E402
import expert  # noqa: E402
import truss_ik  # noqa: E402

SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
FPS = 30
CAM0 = np.array([90.0, 260.0, 100.0])      # base camera position, mm (eval_truss summary cam0_mm, 2026-11-19)
PAIR_PX = 8.0                              # "nearly the same cue": cut pixel within this many px (my choice)
W, H = 640, 480


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/data/*/*.parquet"))])
    df = df.sort_values(["episode_index", "frame_index"])
    rows = []
    for f in LOGS:
        rows += [json.loads(ln) for ln in open(f) if ln.strip()]
    kept = [r for r in rows if r.get("kept") is True]
    eps = {int(k): g for k, g in df.groupby("episode_index")}
    print(f"dataset: {len(eps)} episodes, {len(df)} frames; attempt logs: {len(kept)} kept rows")
    # the merge kept the parts' order (p00..p31) and each part's episode order: check it against the cue numbers
    bad = 0
    for i, r in enumerate(kept):
        e0 = np.asarray(eps[i]["observation.environment_state"].iloc[0], float)
        if not np.allclose(e0, np.asarray(r["env"], float), atol=2e-4):
            bad += 1
    print(f"episode <-> attempt-log match by cue numbers: {len(kept) - bad}/{len(kept)}")
    if bad:
        sys.exit("⚠️ the dataset's episode order is not the attempt logs' order; fix the mapping first")

    ik, fm = expert.ArmIK(n_samples=10), arm_map.FollowerMap()
    E = []
    for i, r in enumerate(kept):
        if r.get("hold"):
            continue
        g = eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(float)
        ac = np.stack(g["action"].to_numpy()).astype(float)
        grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
        jaw = np.empty((len(st), 3))
        for k, s in enumerate(st):
            o, R = ik.fk(fm.norm_to_rad(s)[:5])
            jaw[k] = (o + R @ grip_pt) * 1000
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        close = np.flatnonzero(opened & (ac[:, 5] < 10))
        if not len(close):
            continue
        kc = int(close[0])                                   # first frame the teacher COMMANDS the jaw shut
        kst = np.flatnonzero(opened & (st[:, 5] < 10))
        ks = int(kst[0]) if len(kst) else kc                 # first frame the jaw IS nearly shut
        E.append(dict(i=i, seed=r["seed"], slot=r["target"], noisy=r.get("noise") is not None,
                      env=np.asarray(r["env"], float), plan=r["plan"], frames=len(st), t_close=kc / FPS,
                      t_shut=ks / FPS, jaw=jaw, st=st, ac=ac, cut_xyz=jaw[ks], q_cut=st[ks, :5]))
    clean = [e for e in E if not e["noisy"]]
    print(f"cut episodes {len(E)}, clean {len(clean)}, noisy {len(E) - len(clean)}")
    out = dict(episodes=len(E), clean=len(clean))

    # ---- C. schedule
    tc = np.array([e["t_close"] for e in clean])
    out["C_close_time_s"] = dict(p5=q(tc, 5), p25=q(tc, 25), p50=q(tc, 50), p75=q(tc, 75), p95=q(tc, 95),
                                 min=float(tc.min()), max=float(tc.max()))
    fr = np.array([e["frames"] for e in clean]) / FPS
    out["C_episode_s"] = dict(p5=q(fr, 5), p50=q(fr, 50), p95=q(fr, 95))
    share = float(np.mean([e["t_close"] * FPS / e["frames"] for e in clean]))
    out["C_share_of_frames_before_close"] = share
    print(f"\nC. the teacher commands the jaw shut at {q(tc, 50):.2f} s (median; 5-95 %: {q(tc, 5):.2f}-{q(tc, 95):.2f} s, "
          f"min {tc.min():.2f}, max {tc.max():.2f}); episodes last {q(fr, 50):.1f} s (5-95 %: {q(fr, 5):.1f}-{q(fr, 95):.1f})")
    print(f"   share of an episode's frames before the close: {share:.0%}")

    # ---- D. slots
    print("\nD. cut points per slot (clean episodes): n, spread = median distance to the slot's mean cut point")
    slot_stats, d_proto = {}, []
    for s in sorted({e["slot"] for e in clean}):
        P = np.array([e["cut_xyz"] for e in clean if e["slot"] == s])
        if len(P) < 3:
            continue
        d = np.linalg.norm(P - P.mean(0), axis=1)
        d_proto += d.tolist()
        slot_stats[s] = dict(n=len(P), mean=P.mean(0).round(1).tolist(), std=P.std(0).round(1).tolist(),
                             median_to_mean=round(float(np.median(d)), 1), p90=round(q(d, 90), 1))
        print(f"   {s}: n {len(P):3d}  mean {P.mean(0).round(0)} mm  std {P.std(0).round(1)}  "
              f"median to mean {np.median(d):.1f} mm, 90 % {q(d, 90):.1f}")
    out["D_slots"] = slot_stats
    out["D_slot_only_miss_mm"] = dict(p50=q(d_proto, 50), p90=q(d_proto, 90))
    print(f"   a policy that only knew the slot would miss by {q(d_proto, 50):.1f} mm (median), {q(d_proto, 90):.1f} (90 %)")

    # ---- B. cue sufficiency (leave-one-out k nearest neighbours, locally linear)
    X6 = np.array([e["env"] for e in clean]) * [W, H, W, H, W, H]          # px
    Y = np.array([e["cut_xyz"] for e in clean])

    def loo(X, k=8):
        err = np.empty_like(Y)
        for n in range(len(X)):
            d = np.linalg.norm(X - X[n], axis=1)
            d[n] = np.inf
            nb = np.argsort(d)[:k]
            A = np.c_[X[nb] - X[n], np.ones(k)]               # local linear fit, evaluated at the query
            coef, *_ = np.linalg.lstsq(A, Y[nb], rcond=None)
            err[n] = coef[-1] - Y[n]
        return err

    def split(err):
        ray = Y - CAM0
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        along = np.abs(np.einsum("ni,ni->n", err, ray))
        across = np.linalg.norm(err - np.einsum("ni,ni->n", err, ray)[:, None] * ray, axis=1)
        return dict(total_p50=q(np.linalg.norm(err, axis=1), 50), total_p90=q(np.linalg.norm(err, axis=1), 90),
                    along_ray_p50=q(along, 50), along_ray_p90=q(along, 90),
                    across_ray_p50=q(across, 50), across_ray_p90=q(across, 90))
    out["B_cue_to_3d_mm"] = {"all_6_numbers": split(loo(X6)), "cut_pixel_only": split(loo(X6[:, 4:])),
                             "box_only": split(loo(X6[:, :4]))}
    print("\nB. how well the cue numbers pin down the 3D cut point (leave-one-out, mm):")
    for k, v in out["B_cue_to_3d_mm"].items():
        print(f"   {k:16s} total {v['total_p50']:.1f} (90 %: {v['total_p90']:.1f}) | along the camera ray "
              f"{v['along_ray_p50']:.1f} ({v['along_ray_p90']:.1f}) | across {v['across_ray_p50']:.1f} "
              f"({v['across_ray_p90']:.1f})")
    depth = np.linalg.norm(Y - CAM0, axis=1)
    out["B_depth_from_base_camera_mm"] = dict(p5=q(depth, 5), p50=q(depth, 50), p95=q(depth, 95))
    print(f"   distance base camera -> cut point: {q(depth, 50):.0f} mm (5-95 %: {q(depth, 5):.0f}-{q(depth, 95):.0f})")

    # ---- A. teacher consistency on near-identical cues
    def mode(e):
        p = e["plan"]
        return (p.get("phase"), p.get("approach_blend_deg"), p.get("tilt_deg"), p.get("neighbours"))

    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    P = collections.defaultdict(list)
    n_pairs = n_same = 0
    for a in range(len(clean)):
        ea = clean[a]
        for b in range(a + 1, len(clean)):
            eb = clean[b]
            if ea["slot"] != eb["slot"]:
                continue
            dpx = float(np.linalg.norm((ea["env"][4:] - eb["env"][4:]) * [W, H]))
            if dpx > PAIR_PX:
                continue
            n_pairs += 1
            same = mode(ea) == mode(eb)
            n_same += same
            tag = "same_mode" if same else "other_mode"
            for key in ("all", tag):
                P[key, "cue_px"].append(dpx)
                P[key, "cut_mm"].append(float(np.linalg.norm(ea["cut_xyz"] - eb["cut_xyz"])))
                P[key, "dt_close"].append(abs(ea["t_close"] - eb["t_close"]))
                P[key, "q_cut"].append(np.abs(ea["q_cut"] - eb["q_cut"]))
                for t in times:
                    k = int(t * FPS)
                    if k < min(ea["frames"], eb["frames"]):
                        P[key, f"jaw_{t:.0f}s"].append(float(np.linalg.norm(ea["jaw"][k] - eb["jaw"][k])))
                # the approach in "progress" terms, time taken out: jaw distance to its own cut point 0.5 s before
                # the close and the direction it comes from
                ka, kb = int(ea["t_close"] * FPS) - 15, int(eb["t_close"] * FPS) - 15
                if ka > 0 and kb > 0:
                    va, vb = ea["jaw"][ka] - ea["cut_xyz"], eb["jaw"][kb] - eb["cut_xyz"]
                    P[key, "approach_vec_mm"].append(float(np.linalg.norm(va - vb)))
    out["A_pairs"] = dict(n=n_pairs, same_mode=n_same, pair_px=PAIR_PX)
    print(f"\nA. pairs of clean episodes with the same slot and cut pixel within {PAIR_PX:.0f} px: {n_pairs} "
          f"({n_same} planned in the same mode = phase, approach angle, tilt, neighbour rule)")
    for key in ("all", "same_mode", "other_mode"):
        if not P[key, "cue_px"]:
            continue
        res = dict(n=len(P[key, "cue_px"]), cue_px_p50=q(P[key, "cue_px"], 50), cut_mm_p50=q(P[key, "cut_mm"], 50),
                   cut_mm_p90=q(P[key, "cut_mm"], 90), dt_close_p50=q(P[key, "dt_close"], 50),
                   dt_close_p90=q(P[key, "dt_close"], 90),
                   approach_vec_p50=q(P[key, "approach_vec_mm"], 50), approach_vec_p90=q(P[key, "approach_vec_mm"], 90))
        for t in times:
            v = P[key, f"jaw_{t:.0f}s"]
            res[f"jaw_{t:.0f}s_p50"], res[f"jaw_{t:.0f}s_p90"] = q(v, 50), q(v, 90)
        qd = np.array(P[key, "q_cut"])
        res["q_cut_norm_p50"] = np.percentile(qd, 50, axis=0).round(1).tolist()
        res["q_cut_norm_p90"] = np.percentile(qd, 90, axis=0).round(1).tolist()
        out[f"A_{key}"] = res
        print(f"   [{key}] n {res['n']}: cue apart {res['cue_px_p50']:.1f} px; cut points apart {res['cut_mm_p50']:.1f} mm "
              f"(90 %: {res['cut_mm_p90']:.1f}); closing time apart {res['dt_close_p50']:.2f} s (90 %: "
              f"{res['dt_close_p90']:.2f})")
        print("      jaw apart at the same clock time: " + ", ".join(
            f"{t:.0f} s: {res[f'jaw_{t:.0f}s_p50']:.0f} mm ({res[f'jaw_{t:.0f}s_p90']:.0f})" for t in times))
        print(f"      approach vector 0.5 s before the close apart: {res['approach_vec_p50']:.0f} mm "
              f"({res['approach_vec_p90']:.0f}); joints at the cut apart (norm units, 5 joints) median "
              f"{res['q_cut_norm_p50']}, 90 % {res['q_cut_norm_p90']}")

    # what a one-answer policy would do on such pairs: the midpoint. How far is the midpoint's jaw from either demo?
    json.dump(out, open(OUT / "analyse_dataset.json", "w"), indent=1)
    np.savez_compressed(OUT / "analyse_dataset_episodes.npz",
                        seed=[e["seed"] for e in E], slot=[e["slot"] for e in E], noisy=[e["noisy"] for e in E],
                        env=[e["env"] for e in E], cut_xyz=[e["cut_xyz"] for e in E],
                        t_close=[e["t_close"] for e in E], q_cut=[e["q_cut"] for e in E],
                        n_fruit=[e["plan"].get("n_fruit") for e in E],
                        beta=[e["plan"].get("approach_blend_deg") for e in E],
                        tilt=[e["plan"].get("tilt_deg") for e in E], phase=[e["plan"].get("phase") for e in E],
                        plan_index=[e["plan"].get("plan_index") for e in E])
    print(f"\nsaved {OUT / 'analyse_dataset.json'}")


if __name__ == "__main__":
    main()
