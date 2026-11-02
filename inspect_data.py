import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

DATA_DIR = Path(
    r"C:\University Data\AI & LLMS (Aqeel Tariq)\Research\Data\GrAco_Flight3_Extracted\topic_data"
)

gps_path = DATA_DIR / "gnss__gps.csv"
imu_path = DATA_DIR / "gnss__imu.csv"
ground_truth_path = DATA_DIR / "gnss__ground_truth.csv"
simulated_measurements_path = DATA_DIR / "simulated_GPS_measurements.csv"

gps_df = pd.read_csv(gps_path)
imu_df = pd.read_csv(imu_path)
ground_truth_df = pd.read_csv(ground_truth_path)
simulated_measurements = pd.read_csv(simulated_measurements_path)
required_columns = ["latitude", "longitude", "altitude"]

# Convert coordinates to numeric values
for col in required_columns:
    gps_df[col] = pd.to_numeric(
        gps_df[col], errors="coerce"
    )


# ============================================================
#  Convert latitude and longitude to local ENU coordinates
#    using a local tangent-plane approximation.
# ============================================================

# Reference point: first valid GPS position
lat0 = np.radians(gps_df["latitude"].iloc[0])
lon0 = np.radians(gps_df["longitude"].iloc[0])

lat = np.radians(gps_df["latitude"].to_numpy())
lon = np.radians(gps_df["longitude"].to_numpy())

# Mean Earth radius in metres
EARTH_RADIUS = 6371000.0

# Local East and North coordinates
east = (
    EARTH_RADIUS
    * np.cos(lat0)
    * (lon - lon0)
)

north = EARTH_RADIUS * (lat - lat0)

# Relative altitude, referenced to the first sample
altitude = gps_df["altitude"].to_numpy()
up = altitude - altitude[0]
ml_data = pd.DataFrame();

ml_data["time"] = gps_df["bag_timestamp_sec"]
ml_data["east_m"] = east
ml_data["north_m"] = north
ml_data["up_m"] = up

temp = pd.DataFrame(columns=[
    'bag_timestamp_sec',
    'orientation.x',
    'orientation.y',
    'orientation.z',
    'orientation.w',
    'angular_velocity.x',
    'angular_velocity.y',
    'angular_velocity.z',
    'linear_acceleration.x',
    'linear_acceleration.y',
    'linear_acceleration.z'

])
for i in np.arange(
    imu_df['bag_timestamp_sec'].min(),
    imu_df['bag_timestamp_sec'].max(),
    0.1
):
    temp1 = imu_df[
        (imu_df['bag_timestamp_sec'] >= int(i*10)/10) &
        (imu_df['bag_timestamp_sec'] < int(i*10)/10 + 0.1)
    ]

    if not temp1.empty:
        temp.loc[len(temp)] = {
            'bag_timestamp_sec': int(i*10)/10,
            'orientation.x': temp1['orientation.x'].mean(),
            'orientation.y': temp1['orientation.y'].mean(),
            'orientation.z': temp1['orientation.z'].mean(),
            'orientation.w': temp1['orientation.w'].mean(),
            'angular_velocity.x' : temp1['angular_velocity.x'].mean(),
            'angular_velocity.y': temp1['angular_velocity.y'].mean(),
            'angular_velocity.z': temp1['angular_velocity.z'].mean(),
            'linear_acceleration.x': temp1['linear_acceleration.x'].mean(),
            'linear_acceleration.y': temp1['linear_acceleration.y'].mean(),
            'linear_acceleration.z': temp1['linear_acceleration.z'].mean(),
        }

temp["bag_timestamp_sec"] = round(temp["bag_timestamp_sec"] + 0.1,1)
# ============================================================
# Converting quaternians to Yaw,pitch,roll and populating other data
# ============================================================
# Convert quaternion to Euler angles
quaternions = np.array([
    temp["orientation.x"].to_numpy(),
    temp["orientation.y"].to_numpy(),
    temp["orientation.z"].to_numpy(),
    temp["orientation.w"].to_numpy()
]).T

print(quaternions.shape)  # (49063, 4)

rotation = R.from_quat(quaternions)

roll, pitch, yaw = rotation.as_euler("xyz", degrees=True).T

temp2 = pd.DataFrame(columns=[
    'bag_timestamp_sec',
    'east_m_ground',
    'north_m_ground',
    'up_m_ground'
])
for i in np.arange(
    ground_truth_df['bag_timestamp_sec'].min(),
    ground_truth_df['bag_timestamp_sec'].max(),
    0.1
):
    temp3 = ground_truth_df[
        (ground_truth_df['bag_timestamp_sec'] >= int(i*10)/10) &
        (ground_truth_df['bag_timestamp_sec'] < int(i*10)/10 + 0.1)
    ]

    if not temp3.empty:
        temp2.loc[len(temp2)] = {
            'bag_timestamp_sec': int(i*10)/10,
            'east_m_ground': temp3['pose.pose.position.x'].mean(),
            'north_m_ground': temp3['pose.pose.position.y'].mean(),
            'up_m_ground': temp3['pose.pose.position.z'].mean(),
        }

temp2["bag_timestamp_sec"] = round(temp2["bag_timestamp_sec"] + 0.1,1)


ml_data["roll"] = roll
ml_data["pitch"] = pitch
ml_data["yaw"] = yaw

ml_data["wbx"] = temp["angular_velocity.x"]
ml_data["wby"] = temp["angular_velocity.y"]
ml_data["wbz"] = temp["angular_velocity.z"]

ml_data["abx"] = temp["linear_acceleration.x"]
ml_data["aby"] = temp["linear_acceleration.y"]
ml_data["abz"] = temp["linear_acceleration.z"]

ml_data["east_m_ground"] = temp2["east_m_ground"]
ml_data["north_m_ground"] = temp2["north_m_ground"]
ml_data["up_m_ground"] = temp2["up_m_ground"]



# ============================================================
#  CALCULATE GNSS POSITIONING ERROR
# ============================================================

gnss = ml_data[
    ["east_m", "north_m", "up_m"]
].to_numpy(dtype=float)

ground = ml_data[
    [
        "east_m_ground",
        "north_m_ground",
        "up_m_ground"
    ]
].to_numpy(dtype=float)


# Horizontal error in metres.
error_east = gnss[:, 0] - ground[:, 0]
normalized_east = error_east - min(error_east)
error_east = normalized_east

error_north = gnss[:, 1] - ground[:, 1]
normalized_north = error_north - min(error_north)
error_north = normalized_north

horizontal_error = np.sqrt(
    error_east**2 + error_north**2
)


# Vertical error.
vertical_error = gnss[:, 2] - ground[:, 2]
normalized_vertical = vertical_error - min(vertical_error)
vertical_error = normalized_vertical;

# Full 3D positioning error.
error_3d = np.sqrt(
    error_east**2
    + error_north**2
    + vertical_error**2
)


# Choose which error controls the trajectory colors.
ERROR_TYPE = "horizontal"

if ERROR_TYPE == "horizontal":
    error = horizontal_error
    error_label = "Horizontal GNSS Error (m)"

elif ERROR_TYPE == "3d":
    error = error_3d
    error_label = "3D GNSS Error (m)"

else:
    raise ValueError(
        "ERROR_TYPE must be 'horizontal' or '3d'."
    )


# Save calculated errors in the DataFrame.
ml_data["error_east_m"] = error_east
ml_data["error_north_m"] = error_north
ml_data["error_up_m"] = vertical_error
ml_data["horizontal_error_m"] = horizontal_error
ml_data["error_3d_m"] = error_3d






# ============================================================
# Plot horizontal trajectory
# ============================================================

# Extract data from ml_data DataFrame
# (Adjust column names if your DataFrame uses different keys, e.g., 'acc_x', 'gyro_x', etc.)
east = ml_data["east_m"]
north = ml_data["north_m"]
up = ml_data["up_m"]
time = ml_data.index  # Or ml_data['time'] if you have a specific timestamp column


# ============================================================
# Calculate Full 3D Positioning Error
# ============================================================

error_3d = np.sqrt(
    error_east**2
    + error_north**2
    + vertical_error**2
)

# Ensure coordinates and errors are valid and aligned
trajectory = np.column_stack([
    np.asarray(east),
    np.asarray(north),
    np.asarray(up)
])

errors = np.asarray(error_3d)

valid = np.isfinite(trajectory).all(axis=1) & np.isfinite(errors)

trajectory = trajectory[valid]
errors = errors[valid]

if len(trajectory) < 2:
    raise ValueError("At least two valid trajectory points are required.")

# ============================================================
# Create 3D Trajectory with Error-Based Color Gradient
# ============================================================

fig = plt.figure(figsize=(12, 9))
ax_3d = fig.add_subplot(111, projection="3d")

# Create line segments between consecutive trajectory points
points = trajectory.reshape(-1, 1, 3)
segments = np.concatenate(
    [points[:-1], points[1:]],
    axis=1
)

# Assign each segment the mean error of its two endpoints
segment_errors = (errors[:-1] + errors[1:]) / 2

# Normalize error values for the color mapping
vmin = errors.min()
vmax = errors.max()

if np.isclose(vmin, vmax):
    vmax = vmin + 1e-9

norm = Normalize(vmin=vmin, vmax=vmax)

# Color trajectory according to positioning error
trajectory_line = Line3DCollection(
    segments,
    cmap="jet",
    norm=norm,
    linewidth=2.5
)

trajectory_line.set_array(segment_errors)
ax_3d.add_collection3d(trajectory_line)

# ============================================================
# Start and End Markers
# ============================================================

ax_3d.scatter(
    *trajectory[0],
    color="green",
    s=80,
    label="Start",
    depthshade=False
)

ax_3d.scatter(
    *trajectory[-1],
    color="black",
    s=80,
    marker="X",
    label="End",
    depthshade=False
)

# ============================================================
# Colorbar
# ============================================================

cbar = fig.colorbar(
    trajectory_line,
    ax=ax_3d,
    pad=0.12,
    shrink=0.75
)

cbar.set_label("Full 3D GNSS Positioning Error (m)")

# ============================================================
# Labels and Formatting
# ============================================================

ax_3d.set_xlabel("East (m)")
ax_3d.set_ylabel("North (m)")
ax_3d.set_zlabel("Relative Altitude (m)")

ax_3d.set_title(
    "GRACO Aerial Craft: 3D Trajectory Colored by GNSS Error"
)

ax_3d.legend(loc="upper left")

# Set limits to match the trajectory data
ax_3d.set_xlim(trajectory[:, 0].min(), trajectory[:, 0].max())
ax_3d.set_ylim(trajectory[:, 1].min(), trajectory[:, 1].max())
ax_3d.set_zlim(trajectory[:, 2].min(), trajectory[:, 2].max())

plt.tight_layout()
plt.show()

# ============================================================
# Print Error Statistics
# ============================================================

print(f"Minimum 3D error: {errors.min():.3f} m")
print(f"Maximum 3D error: {errors.max():.3f} m")
print(f"Mean 3D error:    {errors.mean():.3f} m")
print(f"Median 3D error:  {np.median(errors):.3f} m")

np.random.seed(42)  # Reproducible simulated values

error = ml_data["error_3d_m"].to_numpy()

# Normalize error between 0 and 1
error_min = np.nanmin(error)
error_max = np.nanmax(error)

if np.isclose(error_min, error_max):
    normalized_error = np.zeros(len(error))
else:
    normalized_error = (error - error_min) / (error_max - error_min)

# Higher error -> fewer satellites
# Lower error -> more satellites
satellite_count = (
    20 - normalized_error * 16
    + np.random.normal(0, 1.2, len(ml_data))
)

# Round and constrain to a plausible illustrative range
satellite_count = np.clip(
    np.rint(satellite_count),
    4,
    20
)

# Handle rows with missing positioning errors
satellite_count = np.where(
    np.isfinite(error),
    satellite_count,
    np.nan
)

ml_data["satellite_count"] = satellite_count

rng = np.random.default_rng(42)

pdop = (
    1.0
    + normalized_error * 5.0
    + rng.normal(0, 0.35, len(ml_data))
)


ml_data["pdop"] = np.clip(pdop, 0.8, 8.0)
ml_data["pdop"] = ml_data["pdop"].round()

cn0_mean = (
    48.0
    - normalized_error * 20.0
    + rng.normal(0, 2.0, len(ml_data))
)

ml_data["cn0_mean"] = np.clip(cn0_mean, 18.0, 52.0)
ml_data["cn0_mean"] = ml_data["cn0_mean"].round()




ml_data.to_csv("ml_data.csv", index=False)