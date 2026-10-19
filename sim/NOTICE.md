# Third-party files

`assets/so101/` (fetched by `fetch_so101.sh`, not committed) comes from TheRobotStudio/SO-ARM100
(https://github.com/TheRobotStudio/SO-ARM100), commit a758567c3978dfeefe282ede0500085a48fe8f78,
`Simulation/SO101/`. Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
`scene.py` modifies the model at load time: it swaps the stock jaw meshes for the fin-ray parts, adds a
palm collider and a claw camera, and widens the joint and actuator ranges.

`assets/finray/*.stl` are derived works of the same files. They come from
`hardware/end_effector/finray/` (see its `stock/NOTICE.md`), re-exported in metres.
