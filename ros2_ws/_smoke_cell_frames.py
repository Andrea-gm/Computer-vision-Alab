from orbbec_dual_sync.cell_frames import (
    bootstrap_pose_from_base_landmark,
    default_cell_frames_path,
    format_base_retreat,
    load_cell_frames,
)

p = default_cell_frames_path()
cf = load_cell_frames(p)
print("path", p)
print("exists", p.is_file())
print("bootstrap", cf.bootstrap_from_blue_base)
pose = bootstrap_pose_from_base_landmark([100.0, 200.0, 1500.0])
print("t", pose.t_mm.tolist())
print(format_base_retreat([300.0, -100.0, 400.0]))
print("ok")
