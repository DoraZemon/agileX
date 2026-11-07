# Site configuration

Files here are merged over `config/default.yaml` (see `site_overlays` there):

| File | Written by | Content |
|---|---|---|
| `tasks.yaml` | you (copy `config/examples/tasks.example.yaml`) | labels, class_to_zone |
| `poses.yaml` | `teach_poses` | observe pose, zone centers |
| `colors.yaml` | `color_calib` | HSV ranges of the blocks |
| `local.yaml` | you | speed, TCP, gripper, camera topics |
| `extrinsics.yaml` | official file or `calib_points` | camera pose (not merged; read by camera.extrinsics_file) |

Simulation tools ignore this directory unless `--site` is passed.
