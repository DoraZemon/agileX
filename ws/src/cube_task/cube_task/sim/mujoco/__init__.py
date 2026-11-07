"""
MuJoCo physics simulation of the competition (optional; needs the MuJoCo venv).

Modules (import them directly; nothing here imports mujoco):

- engine: MujocoSim, physics stepped in lockstep with core.clock.SimClock.
- scene: build_scene(info) -> world.MujocoWorld, the MJCF scene of a scenario.
- world: MujocoWorld, object state, placements and the physics score.
- ik, arm: MujocoIK and MujocoArm (ArmInterface on the physics).
- camera: MujocoCamera (CameraSource rendering the task camera).
- video: VideoRecorder (MP4 with perception inset and HUD).
- run: scene_config, PhysicsMonitor, failure_causes, final_truth, Viewer,
  RunExtras (what run_task --mode mujoco and sim_bench --backend mujoco add).

Entry point: bash scripts/mujoco_run.sh run_task --task L1 --mode mujoco ...
"""
