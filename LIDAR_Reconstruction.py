
import glob
import os
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ============================================================
# 1. CONFIGURATION
# ============================================================

folder_path = (
    "C:/University Data/AI & LLMS (Aqeel Tariq)/Research/"
    "Data/GrAco_Flight3_Extracted/lidar_sampled_points"
)

ml_data_path = (
    "C:/University Data/AI & LLMS (Aqeel Tariq)/Research/ml_data.csv"
)

output_dir = (
    "C:/University Data/AI & LLMS (Aqeel Tariq)/Research/"
    "Data/GrAco_Flight3_Extracted/visualization"
)

os.makedirs(output_dir, exist_ok=True)

file_pattern = os.path.join(folder_path, "velodyne__points_*.csv")
file_list = sorted(glob.glob(file_pattern))

position_columns = ["east_m", "north_m", "up_m"]
orientation_columns = ["roll", "pitch", "yaw"]

ANGLES_IN_DEGREES = True

# Assumption: LiDAR axes align with UAV body axes.
# Use the actual sensor extrinsic calibration if available.
R_lidar_to_body = R.identity()

# Point cloud settings
VOXEL_SIZE = 0.25

# Ground removal
GROUND_DISTANCE_THRESHOLD = 0.4
RANSAC_ITERATIONS = 1000

# Poisson reconstruction
POISSON_DEPTH = 9
POISSON_DENSITY_PERCENTILE = 5

# Visualization colors (RGB, range 0–1)
TRAJECTORY_COLOR = [1.0, 0.0, 1.0]  # Bright magenta
START_COLOR = [0.0, 0.8, 0.0]       # Green
END_COLOR = [1.0, 0.0, 0.0]         # Red
MESH_COLOR = [0.8, 0.8, 0.8]        # Gray

# Increase the radius to make the path more prominent.
TRAJECTORY_RADIUS = 0.20
MARKER_RADIUS = 0.50
POINT_SIZE = 3.0

# Limit the number of tube segments for rendering performance.
MAX_TRAJECTORY_SEGMENTS = 700

# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

if not os.path.isfile(ml_data_path):
    raise FileNotFoundError(
        f"Cannot find trajectory data:\n{ml_data_path}"
    )

if not file_list:
    raise FileNotFoundError(
        f"No LiDAR CSV files found:\n{folder_path}"
    )

ml_data = pd.read_csv(ml_data_path)

required_columns = position_columns + orientation_columns

missing_columns = [
    column for column in required_columns
    if column not in ml_data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in ml_data.csv: {missing_columns}"
    )

print("=" * 60)
print("GRACO AERIAL-03 RECONSTRUCTION")
print("=" * 60)
print(f"LiDAR frames found: {len(file_list):,}")
print(f"Trajectory rows: {len(ml_data):,}")

if len(file_list) != len(ml_data):
    print(
        "\nWARNING: The LiDAR frame count and trajectory row count "
        "differ. Verify timestamp alignment before scientific use."
    )

# ============================================================
# 3. TRANSFORM LIDAR POINTS INTO WORLD COORDINATES
# ============================================================

points_list = []
trajectory_positions = []

for i, file_path in enumerate(file_list):

    if i >= len(ml_data):
        break

    try:
        df_lidar = pd.read_csv(file_path)
    except Exception as error:
        print(f"Could not read {file_path}: {error}")
        continue

    if not all(
        column in df_lidar.columns
        for column in ["x", "y", "z"]
    ):
        print(f"Skipping CSV without XYZ columns: {file_path}")
        continue

    local_pts = df_lidar[
        ["x", "y", "z"]
    ].to_numpy(dtype=np.float64)

    local_pts = local_pts[
        np.isfinite(local_pts).all(axis=1)
    ]

    if len(local_pts) == 0:
        continue

    position = ml_data.loc[
        i, position_columns
    ].to_numpy(dtype=np.float64)

    angles = ml_data.loc[
        i, orientation_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(position).all():
        continue

    if not np.isfinite(angles).all():
        continue

    # Roll, pitch, yaw -> body-to-world rotation.
    # This assumes the xyz Euler convention.
    R_body_to_world = R.from_euler(
        "xyz",
        angles,
        degrees=ANGLES_IN_DEGREES
    )

    # Transform points from LiDAR coordinates to body coordinates,
    # then from body coordinates to world coordinates.
    points_body = R_lidar_to_body.apply(local_pts)

    points_world = R_body_to_world.apply(points_body)

    # Translate into the world coordinate system.
    global_pts = points_world + position

    global_pts = global_pts[
        np.isfinite(global_pts).all(axis=1)
    ]

    if len(global_pts) == 0:
        continue

    points_list.append(global_pts)
    trajectory_positions.append(position)

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1:,}/{len(file_list):,} LiDAR frames"
        )

if not points_list:
    raise RuntimeError(
        "No valid LiDAR points were accumulated. "
        "Check your CSV files and trajectory data."
    )

all_global_points = np.vstack(points_list)

trajectory_positions = np.asarray(
    trajectory_positions,
    dtype=np.float64
)

print(f"\nAccumulated LiDAR points: {len(all_global_points):,}")
print(f"Valid trajectory positions: {len(trajectory_positions):,}")

# Save the UAV trajectory coordinates.
trajectory_csv = os.path.join(
    output_dir,
    "uav_trajectory.csv"
)

pd.DataFrame(
    trajectory_positions,
    columns=position_columns
).to_csv(trajectory_csv, index=False)

# ============================================================
# 4. CREATE HEIGHT-COLORED LIDAR POINT CLOUD
# ============================================================

print("\nCreating height-colored LiDAR point cloud...")

global_pcd = o3d.geometry.PointCloud()

global_pcd.points = o3d.utility.Vector3dVector(
    all_global_points
)

global_pcd = global_pcd.voxel_down_sample(
    voxel_size=VOXEL_SIZE
)

points = np.asarray(global_pcd.points)

if len(points) == 0:
    raise RuntimeError("The downsampled point cloud is empty.")

print(f"Downsampled LiDAR points: {len(points):,}")

# Color points according to their vertical coordinate.
z = points[:, 2]

z_min, z_max = np.percentile(z, [2, 98])

z_normalized = np.clip(
    (z - z_min) / max(z_max - z_min, 1e-8),
    0.0,
    1.0
)

# Rainbow-style height coloring.
colors = np.zeros((len(points), 3))

colors[:, 0] = z_normalized

colors[:, 1] = (
    1.0 - np.abs(2.0 * z_normalized - 1.0)
)

colors[:, 2] = 1.0 - z_normalized

global_pcd.colors = o3d.utility.Vector3dVector(colors)

# Save the colored point cloud.
o3d.io.write_point_cloud(
    os.path.join(
        output_dir,
        "aerial03_height_colored.ply"
    ),
    global_pcd
)

# ============================================================
# 5. REMOVE THE DOMINANT PLANE AND FILTER NOISE
# ============================================================

print("\nRemoving the dominant plane...")

plane_model, inliers = global_pcd.segment_plane(
    distance_threshold=GROUND_DISTANCE_THRESHOLD,
    ransac_n=3,
    num_iterations=RANSAC_ITERATIONS
)

print(f"Dominant plane model: {plane_model}")
print(f"Points classified as plane: {len(inliers):,}")

objects_pcd = global_pcd.select_by_index(
    inliers,
    invert=True
)

print(
    f"Points remaining after plane removal: "
    f"{len(objects_pcd.points):,}"
)

if len(objects_pcd.points) >= 30:
    objects_pcd, _ = objects_pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

print(f"Points after noise filtering: {len(objects_pcd.points):,}")

if len(objects_pcd.points) < 100:
    raise RuntimeError(
        "Too few points remain for reconstruction. "
        "Adjust the plane-removal parameters."
    )

# ============================================================
# 6. RECONSTRUCT BUILDING SURFACES
# ============================================================

print("\nEstimating surface normals...")

objects_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=2.0,
        max_nn=30
    )
)

objects_pcd.orient_normals_consistent_tangent_plane(100)

print("Running Poisson surface reconstruction...")

mesh, densities = (
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        objects_pcd,
        depth=POISSON_DEPTH
    )
)

densities = np.asarray(densities)

# Remove low-density vertices to reduce reconstruction artifacts.
density_threshold = np.percentile(
    densities,
    POISSON_DENSITY_PERCENTILE
)

mesh.remove_vertices_by_mask(
    densities < density_threshold
)

mesh.compute_vertex_normals()

# Uniform gray building mesh.
mesh.paint_uniform_color(MESH_COLOR)

mesh_path = os.path.join(
    output_dir,
    "aerial03_building_mesh.ply"
)

o3d.io.write_triangle_mesh(mesh_path, mesh)

print(f"Mesh vertices: {len(mesh.vertices):,}")
print(f"Mesh triangles: {len(mesh.triangles):,}")

# ============================================================
# 7. CREATE BRIGHT MAGENTA 3D TRAJECTORY
# ============================================================

def create_tube_between_points(
    p1,
    p2,
    radius=TRAJECTORY_RADIUS,
    color=TRAJECTORY_COLOR
):
    """
    Create a solid 3D cylinder between two trajectory positions.
    The cylinder is explicitly colored magenta.
    """

    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-8:
        return None

    direction = direction / length

    tube = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length,
        resolution=12
    )

    # The cylinder's original longitudinal axis is +Z.
    z_axis = np.array([0.0, 0.0, 1.0])

    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)

    dot = np.clip(
        np.dot(z_axis, direction),
        -1.0,
        1.0
    )

    if axis_norm > 1e-8:

        axis = axis / axis_norm
        angle = np.arccos(dot)

        rotation_matrix = R.from_rotvec(
            axis * angle
        ).as_matrix()

        tube.rotate(
            rotation_matrix,
            center=np.zeros(3)
        )

    elif dot < 0:

        rotation_matrix = R.from_rotvec(
            [np.pi, 0.0, 0.0]
        ).as_matrix()

        tube.rotate(
            rotation_matrix,
            center=np.zeros(3)
        )

    # Center the cylinder between the two positions.
    tube.translate((p1 + p2) / 2.0)

    # Explicitly set the trajectory color.
    tube.paint_uniform_color(color)

    tube.compute_vertex_normals()

    return tube


# Remove consecutive duplicate positions.
if len(trajectory_positions) >= 2:

    keep = np.ones(
        len(trajectory_positions),
        dtype=bool
    )

    keep[1:] = (
        np.linalg.norm(
            np.diff(trajectory_positions, axis=0),
            axis=1
        ) > 1e-5
    )

    trajectory_positions = trajectory_positions[keep]

if len(trajectory_positions) < 2:
    raise RuntimeError(
        "At least two valid trajectory positions are required."
    )

# Downsample the trajectory only for rendering efficiency.
stride = max(
    1,
    int(
        np.ceil(
            (len(trajectory_positions) - 1)
            / MAX_TRAJECTORY_SEGMENTS
        )
    )
)

trajectory_sampled = trajectory_positions[::stride]

# Ensure the final UAV position is included.
if not np.allclose(
    trajectory_sampled[-1],
    trajectory_positions[-1]
):
    trajectory_sampled = np.vstack(
        [
            trajectory_sampled,
            trajectory_positions[-1]
        ]
    )

trajectory_geometries = []

print("\nCreating magenta trajectory tubes...")

for p1, p2 in zip(
    trajectory_sampled[:-1],
    trajectory_sampled[1:]
):

    tube = create_tube_between_points(
        p1,
        p2,
        radius=TRAJECTORY_RADIUS,
        color=TRAJECTORY_COLOR
    )

    if tube is not None:
        trajectory_geometries.append(tube)

print(f"Trajectory segments: {len(trajectory_geometries):,}")
print(f"Trajectory RGB: {TRAJECTORY_COLOR}")

# ============================================================
# 8. CREATE START AND END MARKERS
# ============================================================

start_marker = o3d.geometry.TriangleMesh.create_sphere(
    radius=MARKER_RADIUS,
    resolution=16
)

start_marker.translate(trajectory_positions[0])

start_marker.paint_uniform_color(START_COLOR)

start_marker.compute_vertex_normals()


end_marker = o3d.geometry.TriangleMesh.create_sphere(
    radius=MARKER_RADIUS,
    resolution=16
)

end_marker.translate(trajectory_positions[-1])

end_marker.paint_uniform_color(END_COLOR)

end_marker.compute_vertex_normals()

# ============================================================
# 9. SINGLE COMBINED 3D VISUALIZATION
# ============================================================

print("\nOpening the single combined 3D visualization...")

vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name=(
        "GRACO Aerial-03 | Buildings + LiDAR + "
        "MAGENTA UAV Trajectory"
    ),
    width=1600,
    height=1000
)

# Add the reconstructed buildings.
vis.add_geometry(mesh)

# Add the height-colored LiDAR points.
vis.add_geometry(objects_pcd)

# Add the magenta trajectory tubes.
for tube in trajectory_geometries:
    vis.add_geometry(tube)

# Add the start and end markers.
vis.add_geometry(start_marker)
vis.add_geometry(end_marker)

# Configure the renderer.
render = vis.get_render_option()

render.background_color = np.asarray(
    [1.0, 1.0, 1.0]
)

render.point_size = POINT_SIZE
render.light_on = True
render.mesh_show_back_face = True

# Open exactly one visualization window.
vis.run()

vis.destroy_window()

# ============================================================
# 10. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RECONSTRUCTION COMPLETED")
print("=" * 60)

print(f"Visualization output folder:\n{output_dir}")

print("\nSaved files:")
print(" - uav_trajectory.csv")
print(" - aerial03_height_colored.ply")
print(" - aerial03_building_mesh.ply")

print("\nOnly one 3D visualization window was opened.")


