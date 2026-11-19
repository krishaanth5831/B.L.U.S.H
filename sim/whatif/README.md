# What-ifs and offline searches (November 2026)

Experiments that answered "what if the teacher / scene did X?" without changing dev, and the offline searches
behind some of truss_teacher.py's constants. Moved here from a scratch folder on 2026-11-19 (K7: keep work in
the repos). Patches apply to the dev commit named; their notes and results tables are in the untracked
`outputs/notes/` (2026-11-16_17_step3_night/, 2026-11-18_teacher_v5/), the write-up in the vault's
post-mortem note, section 6.11.

| File | What it tested | Result |
|---|---|---|
| `crate.patch` | crate 6 cm longer toward the plant row | about half of the crate failures fixed; became dev 01e5d9e (crate 23 x 28 cm) |
| `release_fallback.patch` | a second release spot when the first has no plan | 63 to 67 of 154 held-out scenes, none lost; became teacher v4 (0401563) |
| `crate_plus_fallback_full.patch` | both of the above together | teacher v4 + new crate: 80 of 154 |
| `leak_fix.patch` | the evaluator's memory leak (add_self_check) | became dev 160bbbd |
| `liftoff.patch` | lift the open jaw straight up before `return` | 0 of the 4 trusses left standing in the jaw fixed (4 variants); not used |
| `base_shift.patch` | move base + crate along the row (a rail) | 22 of 37 blocked scenes right of x 0.30 solvable (6-fruit 17/23, 8-fruit 0/5); not built |
| `rrt_fallback.patch` | teacher v5 drives an offline RRT route as its last resort | 1 more of the 36 failed gate scenes; not used (the teacher never drives an RRT route, post-mortem rule 3) |
| `other_ready.py` | harvest another ready truss when the most visible one has no plan | about 4 more of the 36 failed gate scenes (old code); not built |
| `via_search.py`, `via_pick.py` | offline RRT routes on 60 scenes from 9940000, clustered into single via-poses | the 3 poses of truss_teacher.VIA_NORM_V5 |
| `run_seeds.sh`, `v5_read.py` | teacher v5 clean dry runs over a seed file, and the reader | the gate: 163 of 200 scenes 9930000-9930199 |

```bash
python whatif/v5_read.py gate                       # after run_seeds.sh gate <seed file> 6
python whatif/other_ready.py 9930000 9930010        # per seed: the other ready trusses, planned and judged
```
