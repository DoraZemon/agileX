# cube_task module contracts (for implementers)

Package root: `ws/src/cube_task` (contains `config/`, `models/`, `cube_task/`, `test/`).
Competition: PiPER arm + Orbbec 大白 RGB-D camera, both installed and calibrated by
the organizers. L1: 3 colored blocks / 180 s into specified zones. L2: 6 fruit /
vegetable models / 300 s into category containers. L3 (optional): natural-language
instructions. Fully autonomous after the start signal. Almost no time on the real
robot, so everything must be testable in simulation and replay, and every
hardware-specific value must come from config.

## Ground rules

- Python 3.8 compatible: `from __future__ import annotations` at the top of every
  module, no `match`, no `X | Y` outside annotations, no `str.removeprefix`.
- Runtime dependencies: numpy 1.21.5, OpenCV 4.5.4 (cv2, includes cv2.aruco),
  PyYAML 5.4, stdlib. Optional: pyAgxArm + python-can (real arm), rclpy + ROS2
  Humble messages (real camera), onnxruntime (if installable without upgrading
  numpy). Import optional dependencies lazily inside functions / constructors so the
  package imports and the tests run without them. Do not pip-install into the
  system Python; the user pins numpy==1.21.5 (`~/constraints.txt`).
- Style: must pass `ament_flake8` (pycodestyle + pyflakes, max line 99) and
  `ament_pep257` (docstring summary line ends with a period, imperative mood, blank
  line after the summary when there is more text, and D213: a multi-line docstring
  starts with `"""` alone on the first line and the summary on the second). Run both on your files:
  `source /opt/ros/humble/setup.bash && ament_flake8 <files> && ament_pep257 <files>`.
- Tests: pytest files under `test/`, run from the package root:
  `cd ws/src/cube_task && python3 -m pytest -q test/test_x.py`. `test/conftest.py`
  puts the package root on sys.path. Each test file runs in < 30 s, is
  deterministic (fixed seeds), needs no hardware and no network.
- Shared files you must NOT edit: `cube_task/core/{types,interfaces,transforms,
  clock,config,zones}.py`, `config/default.yaml`, `setup.py`, `package.xml`,
  `test/conftest.py`, this file. Read config with `cfg.get('a.b', default)` so a key
  missing from default.yaml cannot crash. If you need a contract change or a new
  config key, list it in your final report under `CONTRACT_CHANGE_REQUESTS`.
- Never call `disable()` or `reset()` on the real arm in normal flow (the arm
  drops). Never add code that moves the real arm without going through
  `ArmInterface` + `SafeArm` workspace checks.
- Logging: use the stdlib `logging` module, logger name `cube_task.<module>`.

## Frames and conventions (see core/transforms.py, core/types.py)

- `base`: PiPER base frame, z up, x forward. `flange`: link6 as reported by
  `get_flange_pose()` / `fk()`. `tcp = flange @ T_flange_tcp`, where T_flange_tcp =
  `pose6_to_T(cfg robot.tcp_offset)`; the TCP is the fingertip center.
- Camera frame: color optical frame (x right, y down, z forward).
- Eye-to-hand: extrinsics file holds `T_base_camera`. Eye-in-hand: it holds
  `T_flange_camera`, and `T_base_camera = T_base_flange(at capture) @ T_flange_camera`.
  Capture only while the arm is stationary.
- Poses are 4x4 float64. SDK pose6 `[x, y, z, roll, pitch, yaw]`, R = Rz Ry Rx.
- Flange z axis = approach direction. Finger closing axis = flange (and tool) y.
  Top-down grasp orientation: `top_down_R(closing_angle)` (tool z = base -Z, tool y
  at closing_angle in base XY). Equivalent closing angles differ by pi (by pi/2 for
  square blocks); pick one with `choose_closing_angle()` to keep joint6 near 0.
- Depth: float32 meters aligned to the color image; 0 / NaN / inf = invalid.
  Intrinsics are of the color image; `dist` all zeros = rectified.
- Time: everything uses the injected `Clock` (`RealClock` or `SimClock`). Never
  call `time.time()` / `time.sleep()` directly in task, robot or sim logic, so the
  simulation runs faster than real time and tests are deterministic.

## Detection geometry (filled by perception/pipeline.py)

For each object instance found above the table inside the pick polygon:
`center_base = [x, y, top_z]` with x, y = center of the minimum-area rectangle of the
instance's above-table points projected on base XY, top_z = `top_percentile` of
their z. `height = top_z - table_z(x, y)`. `yaw` = rectangle major-axis angle in
[-pi/2, pi/2). `length` >= `width` = rectangle extents. `extra['table_z']` = table
z under the object. `mask` = HxW bool in the color image. Geometry is filled before
classification so classifiers may use length / width / height.

## Factories and constructor signatures

Checked against the code on 2026-11-06 (`inspect.signature` of each name). Only the public
API other modules use is listed; read the docstrings for details.

| Module | API |
|---|---|
| core/config.py | `load_config(base=None, extra=(), overrides=(), use_overlays=True, root=None) -> Config`; `Config(data, root, sources)`: `.get(key, default)`, `.set(key, value)`, `.has`, `.section`, `.path(key)`, `.resolve(p)`, `.to_yaml()`, `.copy()`, `.sources`, `.root`; `package_root()` (CUBE_TASK_ROOT, else the source tree; after a plain colcon build the installed `<prefix>/share/cube_task` read-only with a warning; RuntimeError naming the fix when neither has config/default.yaml); `deep_merge(base, over)`; `read_yaml(path)`; `write_yaml(path, data, header='')` (atomic) |
| core/zones.py | `parse_zones(cfg) -> {name: Zone}`; `parse_zone(name, d)` (center 2 or 3 values, size default [0.10, 0.10], yaw, wall_height, release_height, label) |
| core/extrinsics.py | `Extrinsics(T, mount, parent_frame=None, child_frame='camera_color_optical_frame', source='', rms=None, stamp=None)` (raises ValueError for a non-optical child_frame): `.name` ('T_<parent>_camera'), `.to_dict()`, `.describe()`; `load_extrinsics(path, default_mount=None) -> Extrinsics` (formats: ours, `matrix` / `T`, `T_base_camera` / `T_flange_camera` / `T_tcp_camera`, inverted `T_camera_*`, `translation` + `quaternion` (xyzw) / `rotation` 3x3 / `rpy` / `rpy_deg`, `xyz` + `rpy`, handeye_calibration_ros JSON `position` + `orientation`, bare 4x4 / 16 numbers / plain text / .npy; translation >= 3 m rejected; several representations must agree; parent_frame tcp / `T_tcp_camera` / `T_camera_tcp` needs a `tcp_offset` pose6 (the driver TCP offset during calibration) and loads as parent 'flange'; `rpy` with an angle beyond +-2*pi rejected (use `rpy_deg`); child_frame must end in `_optical_frame` (case-insensitive) unless `T_link_optical` (4x4 or translation + quaternion) is given, which is composed; a .json file that is not JSON is read as YAML with a warning); `extrinsics_from_dict(d, default_mount=None, source=None, what='extrinsics')`; `save_extrinsics(path, ext) -> Path` (atomic; .json path -> JSON with `_comment`, else commented YAML; ValueError for .npy and for parent_frame tcp); `extrinsics_from_cfg(cfg) -> Extrinsics or None` (camera.extrinsics_inline wins over camera.extrinsics_file; missing file -> None; mount must equal camera.mount); `camera_pose_in_base(ext, T_base_flange=None, T_flange_tcp=None) -> T_base_camera` (parent_frame base / flange / tcp); `is_optical_frame(name)`, `non_optical_frame_message(child)`, `MAX_RPY_RAD` |
| core/transforms.py | pose / rotation helpers used everywhere (`pose6_to_T`, `T_to_pose6`, `top_down_T`, `top_down_R`, `inv_T`, `make_T`, `rpy_to_rot`, `rot_to_rpy`, `quat_to_rot`, `rot_to_quat`, `rotation_angle`, `is_rotation`); `fit_rigid(src, dst) -> (T, rms)` (calib_points) |
| core/logger.py | `RunLogger(base_dir, task, clock, save_images=True, name=None)`: `.dir` (`<base_dir>/<YYYYmmdd_HHMMSS>_<task>`), `.event(kind, **fields)` (events.jsonl, `t` from the injected clock, `wall`, `kind`), `.save_image(name, bgr)`, `.save_json(name, obj)`, `.save_text(name, text)`, `.path(name)`, `.close(summary=None)` (summary.json), `.elapsed`, `.n_events`, `.n_failures`; never raises on I/O errors. Module helpers `to_jsonable`, `safe_filename`, `atomic_write_text` |
| camera/factory.py | `make_camera(cfg, clock, sim_world=None, arm=None) -> CameraSource` (backend ros / file / sim; 'sim' reads cfg sim.intrinsics / sim.T_base_camera / sim.T_flange_camera) |
| camera/utils.py | `capture_averaged(camera, n, newer_than=None, timeout=2.0) -> Frame or None` (per-pixel median depth of n consecutive frames, ignoring invalid); `valid_depth_mask(depth)` |
| camera/ros_camera.py | `RosCamera(cfg, clock)`: `start()`, `stop()`, `get_frame(timeout, newer_than)`, `intrinsics()`, `status()`, `diagnose() -> str`; pure helpers `image_msg_to_numpy(msg, yuv422_order='uyvy')`, `depth_msg_to_meters(msg, depth_scale)`, `camera_info_to_intrinsics(msg, ignore_distortion=False)` (8 coefficients labeled plumb_bob = rational); `status()` includes `header_offset_s` (color - depth header stamp, s) and `sync_mode`; config camera.ros.sync_mode (header / receipt), ignore_distortion, intrinsics_override |
| camera/registration.py | `register_depth_to_color(depth_d, intr_d, intr_c, T_color_depth, fill_holes=True) -> depth in the color image` (used when camera.ros.depth_registered is false) |
| camera/file_camera.py, camera/recorder.py | `FileCamera(dir, clock, loop=True, repeat=1)`; `save_frame(dir, idx, frame, extra=None, save_depth_npy=False) -> Path`, `load_frame(dir, idx) -> (Frame, extra)`, `next_index(dir)` (files `NNNNNN_color.png`, `_depth.png` 16-bit mm, optional `_depth.npy` float m (preferred on load), `_meta.json` with intrinsics / stamp / frame_id / source / extra) |
| sim/world.py | `SimWorld`, `SimObject`, `SimZone` (below) |
| sim/render.py | `render_rgbd(world, intrinsics, T_base_camera, gripper=None, rng=None, noise_std=0.0, dropout=0.0) -> (color, depth)` |
| sim/sim_camera.py | `SimCamera(world, clock, intrinsics, T_base_camera=None, T_flange_camera=None, arm=None, noise_std=0.0008, dropout=0.002, seed=0)`: `inject(kind, count=1)` (FAULT_KINDS none_frames / stale / no_depth), `clear_faults()`, `current_T_base_camera()` |
| sim/scenarios.py | `make_scenario(task, seed, mount='eye_to_hand', difficulty='normal') -> (SimWorld, info)`; info keys: task, seed, mount, difficulty, labels, class_to_zone, categories, `zones` (core Zone dict), pick_polygon, intrinsics, `T_base_camera` (eye_to_hand) or `T_flange_camera` + `T_base_camera_observe` (eye_in_hand), observe_joints, objects, required, time_limit_s; L3 also instructions, instruction_specs, label_aliases, zone_aliases. `scenario_config_overlay(info, extrinsics_error=(0, 0)) -> dict` to merge into cfg (zones, tasks.<T>.labels / class_to_zone (+ L3 aliases), perception.roi.pick_polygon, camera.backend=sim, camera.mount, camera.extrinsics_inline, robot.backend=sim, robot.poses.observe_joints, sim.*); `perturb_pose(T, trans_err, rot_err_deg, rng)` |
| sim/teach.py | `collect_sim_samples(seeds, mount='eye_to_hand', task='L2', match_dist=0.02) -> (X, y, feature_names)`; `train_knn_from_sim(out, mount='eye_to_hand', n_scenes=40, first_seed=100, k=3, max_distance=3.0) -> (path, leave-one-out accuracy)` |
| robot/kinematics.py | `PiperKinematics(model='piper', joint_limits=None)`: `fk(q) -> T_base_flange`, `fk_all`, `jacobian`, `ik(T_base_flange, q_seed=None, max_iter=60, tol_pos=1e-4, tol_rot=1e-3, n_random=4) -> (q or None, pos_err, rot_err)`, `refine(T_base_flange, q, ...)`, `ik_tcp(T_base_tcp, T_flange_tcp, q_seed=None, **kw)`, `solve_tcp_with_tilt(T_base_tcp, T_flange_tcp, max_tilt=30 deg, step=5 deg, q_seed=None) -> (T_base_tcp_used, q, tilt) or None`, `within_limits(q)`, `limit_violations(q)`, `clamp(q)`, `limits`; module functions `tilt_outward(T_base_tcp, tilt)`, `flange_to_tcp_from_cfg(cfg)` |
| robot/safety.py | `SafetyGuard(cfg)`: `check_tcp(T_base_tcp, table_z=None, check_tilt=True) -> (ok, reason)`, `tilt_from_down(T)`; `SafeArm(arm, guard, kin=None, T_flange_tcp=None)` wraps any ArmInterface and rejects unsafe / unreachable targets with `MotionResult(ok=False, error='rejected')`; extra methods `set_table_z(z)`, `table_z`, `check_reachable(T_base_tcp, q_seed=None) -> (ok, reason)`, `find_reachable_tilt(T_base_tcp, q_seed=None, step_deg=5.0) -> (T or None, tilt rad, reason)`, `inner` (wrapped arm), `guard`, `kin`, and the pass-through teach API `supports_teach()`, `read_teach_pose()`, `exit_teach_mode()`. Health refusal returns error 'comm' or 'arm_error' |
| task/reach.py | `reach_adjust(arm, T_base_tcp, q_seed=None) -> (T or None, tilt_deg, reason)`; `adjust_plan_poses(arm, grasp_plan, place_plan) -> (ok, reason, tilts)`. PiPER joint5 (+-70 deg) with a 0.14 m TCP reaches a vertical tool only up to TCP z ~0.10 m at r 0.2-0.3 m (0.065 m at r 0.40 m), so pre-grasp / transit / place / observe poses are leaned outward by the smallest tilt that works (max robot.workspace.max_tilt_from_down_deg) |
| robot/sim_arm.py | `SimArm(cfg, clock, world=None)` implements ArmInterface; `inject(kind, at_call=None, count=1)` (FAULT_KINDS timeout / unreachable / comm_loss / grasp_miss / drop / arm_error), `set_joints(q)` (teleport, scenario setup and simulated drag-teach only), `world` |
| robot/pyagxarm_arm.py | `PyAgxArm(cfg, clock, wall_time=None)` implements ArmInterface with pyAgxArm; `supports_teach()`, `read_teach_pose()` (joints / flange / TCP, also in drag-teach mode), `exit_teach_mode()` (official agx_arm_ctrl sequence: unsmoothed move_js jump to zero, e-stop, reset(), enable, move_j zero; on-site tools only, behind `Ui.confirm_word`; raises ArmError and sends nothing unless every joint is within `EXIT_TEACH_MAX_JOINT_RAD` = 0.05 rad of 0); no joint / pose command (including the hold of `stop()`) is sent while ctrl_mode is TEACHING_MODE; move_tcp sends roll / yaw at +-pi with the sign of the feedback angle (`canonical_pose6`); helpers `linear_moves_enabled(cfg)` (robot.motion.linear_moves, default true), `seamless_teach_switch(version)` (S-V1.8-5+, `SEAMLESS_TEACH_FIRMWARE`), `piper_firmware_tuple(version)`, `teach_exit_hint(version)`, `zero_pose_problem(joints, tol)` |
| robot/factory.py | `make_arm(cfg, clock, sim_world=None) -> SafeArm` (robot.backend pyagxarm or sim) |
| perception/factory.py | `make_classifier(cfg, task) -> InstanceClassifier or None`; `make_detector(cfg, task) -> Detector2D or None`; `make_perception(cfg, task) -> Perception`; `classifier_backend(cfg, task) -> str` (tasks.<T>.classifier, else perception.classifier.backend) |
| perception/detector_onnx.py | `OnnxDetector(cfg, section='perception.classifier.onnx')` implements Detector2D (YOLOv8 / YOLO11 ONNX, cv2.dnn or onnxruntime); `OnnxDetector.from_paths(model_file, labels_file, **options)`, `.detect(color)` |
| perception/pipeline.py | `Perception(cfg, classifier=None, detector=None)`: `.process(frame, T_base_camera, zones=None) -> PerceptionResult` (never raises; problems in `.warnings`), `.reset_table()` |
| perception/grasp.py | `GraspPlanner(cfg, task)`: `.plan(det) -> GraspPlan`, `.finger_hits(...)`; `PlacePlanner(cfg)`: `.plan(zone, held_offset, transit_z, footprint=None, occupied=(), center_first=True) -> PlacePlan` (occupied = (x, y, footprint, height) of objects already released into the zone; release at the first free slot, and at least 1 cm above the tallest of them; place.spread_releases false = zone center), `.release_xy(zone, footprint=None, occupied=(), center_first=True)`; helpers `max_gripper_width(cfg)`, `transit_height(cfg, zones, table_z, held_offset=0.0, obstacles_top_z=None)`, `held_offset_of(plan)`, `detection_table_z(det)`, `release_slots(zone, footprint, margin=0.015)`, `expected_table_z(cfg)`, `table_floor_z(perceived_z, expected_z)` (expected_z is a lower bound), `table_z_warning(perceived_z, expected_z, tol=0.01)` |
| perception/classify_color.py | `ColorClassifier(classes, min_fraction=0.35, allowed_labels=None)`: `.classify(frame, dets)`, `.fractions`, `.pixel_classes`; `hsv_ranges_from_samples(pixels_by_label, lo_pct=2.0, hi_pct=98.0, hue_margin=4.0, sv_margin=25.0, min_saturation=40, min_chromatic_fraction=0.3)` (color_calib) |
| perception/classify_knn.py | `KnnClassifier(k=3, max_distance=3.0, std_floor=0.001, feature_floors=None)` (None = `FEATURE_STD_FLOORS`, per-feature std floors on the physical scale; {} = old behaviour), `.std_floors()`, `.fit(X, y, feature_names=None)`, `.add_samples(X, y)`, `.predict`, `.classify(frame, dets)`, `.leave_one_out() -> dict`, `.save(path, created=None)`, `KnnClassifier.load(path, k=3, max_distance=3.0, std_floor=0.001, feature_floors=None)`, `.n_samples`; model files unchanged (version 1, std recomputed at load) |
| perception/features.py | `feature_vector(frame, det, height_crop=None, points_xy=None) -> (vector, names)`; `upper_mask(det, height_crop=None)` |
| task/skills.py | `PickPlaceSkill(cfg, arm, clock, logger=None)`: `.execute(grasp, place, dry_run=False) -> PickResult`; `is_fault_motion_error(error)`; lift / release / retreat / recovery moves are move_l unless robot.motion.linear_moves is false (then P2P) |
| task/runner.py | `TaskRunner(cfg, task, arm, camera, perception, clock, logger, operator=None, dry_run=False, extrinsics=None, targets=None, instructions=None, mode=None)`: `.run() -> TaskReport`, `.fault_detail` (text after a fault / refusal), `.apply_selector(selector, dets)`. extrinsics None -> `extrinsics_from_cfg(cfg)`. targets: label list overriding tasks.<T>.order (`required` is not changed). instructions: L3 strings. mode 'real' / 'sim' (default from robot.backend) |
| task/selection.py | `select_target(candidates, strategy='nearest', order=None, arm_xy=(0, 0), already_failed=None, blacklist_radius=0.03, neighbors=None) -> Detection or None` (strategies nearest / score / isolated); `is_blacklisted(det, positions, radius)` |
| task/instruction.py | `parse_instruction(text, cfg, labels=None, zones=None) -> InstructionSpec(label, selector, zone, count, raw, source, ok, reason)` (rules; LLM when tasks.L3.llm.enabled, falling back to rules; the LLM is not asked when the rules refuse with `NOT_SUPPORTED`); `parse_rules(text, cfg, labels, zones)` (rules only; also used by tools/run_task.py for the summary); `NOT_SUPPORTED` (reason prefix 'not supported: ' for constructs a spec cannot express: ordinals, objects described by a zone / container or by an unknown place); `SELECTORS`, `COUNT_ALL = -1` |
| task/selfcheck.py | `self_check(cfg, task, arm=None, camera=None, mode='real', extrinsics=None, clock=None) -> (ok, [messages])` ('[FAIL]' / '[WARN]' / '[ OK ]' prefixes); `observe_tcp_pose(cfg)`, `camera_pose_for_check(cfg, ext, arm) -> (T or None, why)`, `pick_area_mask(poly, table_z, T_base_camera, intr)`, `check_pick_area(cfg, camera, ext, arm, newer_than=None) -> (ok, msg)` (the runner calls it at the observe pose for eye_in_hand; away from the observe pose self_check only WARNs about the pick area of an eye_in_hand camera), `at_observe_pose(cfg, arm, tol=0.035)` |
| task/operator.py | `Operator(enabled=True, estop_callback=None, stdin=None, out=None)`: `start()`, `wait_start(prompt=..., discard_earlier=False)`, `abort(reason)`, `estop()`, `feed(ch)`, `close()`, `set_estop_callback(fn)`, `abort_event`, `start_event`, `estopped` (keys below) |
| tools/common.py | `add_common_args(p, task_required=False, default_task='L1')` (--task --mode --config --set --site --seed --mount --difficulty --extrinsics-error --replay-dir --log-dir -v); `load_cfg_for(args, task=None) -> (cfg, world or None, info or None)` (sim: default without site overlays (--site adds them) + scenario overlay; replay: camera.backend=file, robot.backend=sim); `open_session(args, task=None, arm=True, camera=True, perception=True, logger=False, operator=False, connect=True, cfg_world_info=None) -> Session` (connect=True calls connect() + enable()); `Session` (context manager): `.cfg .mode .task .clock .world .info .arm .camera .perception .logger .operator`, `.close()` (never disables); `ensure_sim_knn(cfg, task, mount)` (cached in ~/.cache/cube_task, key = hash of sim/ and perception/ sources); `setup_logging(verbose)` |
| tools/ui.py | `Ui(headless=False, preview_dir=None, input_fn=None, output=None)`: `say`, `show(name, img, save_every=1) -> preview path or None`, `preview_path(name)`, `command(text, wait_ms=30)`, `key(wait_ms)`, `prompt(text, default='')`, `confirm(text, default=False)`, `confirm_word(text, word='YES')`, `set_mouse_callback(name, fn)`, `close()`, `.eof`, `.headless`; `depth_colormap(depth, zmin=0.2, zmax=1.5)`, `side_by_side(a, b)` |
| tools/onsite.py | shared helpers of the on-site tools: `add_onsite_args(p, site=True)` (--headless --preview-dir --site-dir), `make_ui(args, cfg, tool, ui=None)`, `write_site_yaml(path, data, header='') -> backup path` (backup `<stem>.bak_<YYYYmmdd_HHMMSS><suffix>`, atomic write), `output_path(cfg, mode, site_dir, default)` (sim -> `<logging.dir>/sim_site`), `resolve_speed(cfg, requested=None) -> (speed, note)` (a higher request is clamped to robot.speed_percent), `MotionGuard(ui, mode, speed).approve(what)` (prints every motion, asks once in real mode), `ensure_motion_ready(ui, arm, ...)` (offers Enter / c / x / q in drag-teach mode), `exit_teach_mode_confirmed(ui, arm)` (refused unless the arm is folded at joint zero, then typed `EXIT`), `take_can_control_confirmed(ui, arm)` (typed `CAN`, calls `arm.enable()` only, no motion), `arm_firmware_version(arm)`, `extrinsics_out_path(cfg, mode, site_dir)` (a configured suffix other than .yaml / .yml / .json, e.g. .npy, is written as the .yaml sibling), `read_pose(arm)`, `read_pose_averaged(arm, clock, n=8, dt=0.02)`, `go_observe(arm, cfg, guard)`, `connect_arm_readonly(s, ui, required=False, why=...)`, `camera_pose_for_frame(ext, frame, arm, T_flange_tcp=None)` (uses the flange pose recorded with a replayed frame), `capture_view`, `compose_view`, `write_extrinsics_site(path, ext)`, `assign_command(text, n)`, `detection_table(dets)`, `coverage_check(points, polygon=None, ...)` |
| tools/run_task.py | `main(argv=None) -> exit code` (0 complete, 1 incomplete, 2 refused before motion, 3 fault, 130 Ctrl-C; `EXIT_MEANING`); `build_parser()`; `parse_targets(text, cfg, task)`, `resolve_speed(args, cfg)` (refuses a raise without --allow-faster), `exit_code_for(rep, dry_run, estopped, truth=None)` |
| tools/check_setup.py | `main(argv=None)` (0 no FAIL, 1 FAIL, 2 bad --skip); groups modules / can / ros / config / live; `parse_ip_link(text)`, `check_can(rep, channel, bitrate=1000000, run=None)`, `config_messages(cfg, task)` |
| tools/sim_bench.py | `main(argv=None)` (0 no wrong placement / error, 1 otherwise, 2 bad arguments, 130 Ctrl-C); `make_specs(args)`, `run_one(spec) -> dict` (never raises), `ground_truth(world, info, task, targets=None) -> {correct, wrong, remaining}`, `is_success(run)`, `aggregate(runs)`, `reproduce_command(spec)` |
| tools/capture, view, teach_poses, calib_verify, calib_points, color_calib, teach_classes | each has `build_parser()` and `main(argv=None, ui=None) -> exit code` (ui injectable for tests); see the module docstrings and docs/现场流程.md. teach_poses also has `DELETE`, `pop_dotted`, `clip_to_limits`, `joint_limit_violations`, `zone_sources`, `zone_users`, `JOINT_LIMIT_TOL` (command `d [name]` deletes a zone) |
| `__main__.py` | `python3 -m cube_task <tool> [args]` runs `cube_task.tools.<tool>.main()`; setup.py registers the same 10 tools as console scripts (`ros2 run cube_task <tool>`) |

## Simulation world (sim/world.py), used by SimArm and SimCamera

```
SimObject(name, label, shape, size, color_bgr, T_base_obj)
    shape: 'box' size (sx, sy, sz) | 'sphere' (r,) | 'ellipsoid' (rx, ry, rz)
           | 'cylinder' (r, h) | 'capsule_chain' (r, [(x, y, z) local points])
           | 'sphere_cluster' (r, [local centers])
    T_base_obj: object frame, origin at the bottom center, z up.
SimZone(name, center (3,), size (2,), yaw, color_bgr, wall_height)
SimWorld(table_z=0.0, objects=[], zones=[], table_color_bgr=(...), seed=0)
    .objects, .zones (lists), .held (SimObject or None)
    .try_grasp(T_base_tcp, open_width, close_width) -> Optional[float]
        Called by SimArm when the gripper closes. Succeeds when one object's
        extent along the tool y axis at TCP height is < open_width, the TCP is
        inside the object's height range, and the TCP xy is within the object's
        footprint (tolerance 8 mm along tool x, 30 % of half extent along tool
        y). Attaches it and returns the held width (m); else None.
    .update_tcp(T_base_tcp, width)  # held object follows the TCP; gripper drawn
    .release(T_base_tcp) -> Optional[str]
        Drops the held object straight down: lands on the floor of the zone that
        contains its xy (returns zone name) or on the table (returns None).
    .placements() -> Dict[str, List[str]]   # zone name -> labels inside
    .remaining(pick_polygon=None) -> List[SimObject]
    .score(class_to_zone) -> dict(correct=int, wrong=int, remaining=int)
```

Also: `SimWorld.check_grasp()`, `zone_at()`, `last_grasp` / `last_release` (reason of the
last attempt: 'no_object' | 'collision_finger' | ...). Merge `scenario_config_overlay()`
over a config loaded with `load_config(use_overlays=False)` so site files do not leak in.

## Perception notes

- `Detection.extra` keys starting with `_` hold numpy arrays (not JSON); skip them when
  dumping. `extra['cap_center_xy']` = center of the topmost points; the grasp planner uses
  it (grasp.use_cap_center) because the footprint center of round objects seen by an
  oblique camera is biased toward the camera by ~0.15 r. The cap uses the points with
  top_z - max(2 mm, 5 % height) <= z <= top_z + max(3 mm, band, 3 x table rms)
  (`segment.cap_band`).
- Geometry points exclude depth-edge pixels. Isolated near depth spikes are dropped
  (`segment.depth_spikes`, perception.segmentation.despike). The split barrier step is
  `segment.split_jump(params, table rms)`, recorded in `Segmentation.split_jump_m`.
- KNN `max_distance` is an RMS standardized distance (divided by sqrt(n_features)).
- Reject reasons: too_small, too_large, touches_roi_border, touches_image_border,
  too_tall, touches_zone, occluded, few_points (see segment.REASON_ORDER).
- ONNX models for OpenCV 4.5.4 must be exported with tools/yolo/export_onnx.py (graph
  patch); see docs/yolo.md.

## Operator controls (task/operator.py)

`Operator(enabled=True)`: Enter starts the timed run; `p` or space aborts (stop
issuing commands, hold); `e` triggers `arm.emergency_stop()`. Disabled operator
(tests / sim) starts immediately and never aborts unless `.abort()` is called.
The runner starts the key reader before its first motion; an Enter typed before the start
prompt does not count.
