
# =====================================================================
# ML_Model.py
# GRACO Aerial-03: XGBoost GNSS Error Prediction with LiDAR
# =====================================================================
#
# INPUTS:
#   1. Existing ML dataset: ml_data.csv
#   2. LiDAR point clouds: velodyne__points_######.csv
#
# TARGET:
#   error_3d_m (existing column; NOT recalculated)
#
# FEATURES:
#   GNSS position
#   IMU acceleration
#   IMU angular velocity
#   Roll, pitch, yaw
#   Satellite count
#   PDOP
#   Mean C/N0
#   Time-derived feature
#   LiDAR geometric features
#
# OUTPUTS:
#   Trained XGBoost model
#   Evaluation metrics
#   Test predictions
#   Feature importance
#   Training/testing dataset
#   Diagnostic plots
#
# IMPORTANT:
#   LiDAR frames are initially aligned by sequence.
#   Correct timestamp synchronization must be verified before
#   treating results as scientifically validated.
# =====================================================================


# =====================================================================
# 1. IMPORT LIBRARIES
# =====================================================================

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import TimeSeriesSplit


warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# 2. PATH CONFIGURATION
# =====================================================================

BASE_DIR = Path(
    r"C:\University Data\AI & LLMS (Aqeel Tariq)\Research"
)

ML_DATA_PATH = BASE_DIR / "ml_data.csv"

LIDAR_DIR = (
    BASE_DIR
    / "Data"
    / "GrAco_Flight3_Extracted"
    / "lidar_sampled_points"
)

OUTPUT_DIR = (
    BASE_DIR
    / "Data"
    / "GrAco_Flight3_Extracted"
    / "xgboost_final_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# 3. MODEL CONFIGURATION
# =====================================================================

TARGET = "error_3d_m"

TEST_FRACTION = 0.20

RANDOM_STATE = 42

# XGBoost parameters.
# These are initial hyperparameters, not optimized values.

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "rmse"
}

# Maximum number of points used to calculate geometric features
# from a single LiDAR frame.

MAX_POINTS_PER_FRAME = 50000

# Treat points at (0,0,0) as possible placeholders.
# They will be counted, then excluded from geometric statistics.

EXCLUDE_ORIGIN_FROM_GEOMETRY = True

ORIGIN_TOLERANCE = 1e-8

# If True, stop rather than interpolate when row counts differ.
# This is recommended for scientifically rigorous experiments.
STRICT_ALIGNMENT = False


# =====================================================================
# 4. ML FEATURES
# =====================================================================

# These are the known columns in your existing dataset.
#
# Ground-truth coordinates are deliberately excluded because they
# would provide information unavailable to a real-time UAV system.
#
# Existing GNSS error columns are also excluded from the inputs.
# error_3d_m is the target, not a feature.

SENSOR_FEATURES = [
    "east_m",
    "north_m",
    "up_m",

    "roll",
    "pitch",
    "yaw",

    "wbx",
    "wby",
    "wbz",

    "abx",
    "aby",
    "abz",

    "satellite_count",
    "pdop",
    "cn0_mean"
]


# =====================================================================
# 5. HELPER FUNCTIONS
# =====================================================================

def natural_sort_key(path):

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def finite_rows(array):

    return np.isfinite(array).all(axis=1)


def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "MAE_m": float(mae),
        "RMSE_m": float(rmse),
        "R2": float(r2)
    }


# =====================================================================
# 6. LOAD THE EXISTING ML DATASET
# =====================================================================

print("\n" + "=" * 75)
print("LOADING EXISTING ML DATASET")
print("=" * 75)

if not ML_DATA_PATH.exists():

    raise FileNotFoundError(
        f"ML dataset not found:\n{ML_DATA_PATH}"
    )

ml_data = pd.read_csv(
    ML_DATA_PATH
)

ml_data.columns = [
    str(col).strip()
    for col in ml_data.columns
]

print("Dataset path:", ML_DATA_PATH)
print("Dataset shape:", ml_data.shape)

print("\nAvailable columns:")

for col in ml_data.columns:
    print("  ", col)


# =====================================================================
# 7. VALIDATE THE TARGET AND INPUT FEATURES
# =====================================================================

if TARGET not in ml_data.columns:

    raise ValueError(
        f"Target '{TARGET}' was not found in ml_data.csv.\n"
        f"Available columns: {ml_data.columns.tolist()}"
    )


missing_features = [
    col
    for col in SENSOR_FEATURES
    if col not in ml_data.columns
]

if missing_features:

    raise ValueError(
        "The following expected ML parameters are missing:\n"
        f"{missing_features}"
    )


# =====================================================================
# 8. PROCESS THE TIME FEATURE
# =====================================================================

# Time is useful as a temporal indicator, but raw timestamps should
# not be passed directly to XGBoost as strings.
#
# The existing timestamp is converted into seconds within the day.
# The original timestamp is retained separately for diagnostics.

if "time" in ml_data.columns:

    raw_time = ml_data["time"].astype(str)

    parsed_time = pd.to_datetime(
        raw_time,
        errors="coerce",
        utc=True
    )

    # Fallback for timestamps represented as elapsed seconds.
    numeric_time = pd.to_numeric(
        ml_data["time"],
        errors="coerce"
    )

    seconds_of_day = (
        parsed_time.dt.hour * 3600
        + parsed_time.dt.minute * 60
        + parsed_time.dt.second
        + parsed_time.dt.microsecond / 1_000_000
    )

    seconds_of_day = seconds_of_day.astype(float)

    seconds_of_day = seconds_of_day.where(
        parsed_time.notna(),
        numeric_time
    )

    ml_data["time_seconds"] = seconds_of_day

    if ml_data["time_seconds"].notna().any():

        SENSOR_FEATURES.append(
            "time_seconds"
        )

        print(
            "\nTime feature successfully processed."
        )

    else:

        print(
            "\nWARNING: Time could not be converted. "
            "The model will exclude it."
        )


# =====================================================================
# 9. CHECK FOR TARGET LEAKAGE
# =====================================================================

FORBIDDEN_FEATURE_TOKENS = [
    "ground",
    "error",
    "filename",
    "file_name"
]

for feature in SENSOR_FEATURES:

    if any(
        token in feature.lower()
        for token in FORBIDDEN_FEATURE_TOKENS
    ):

        raise ValueError(
            f"Potential target leakage detected: {feature}"
        )


# =====================================================================
# 10. LOAD A LIDAR CSV FILE
# =====================================================================

def load_point_cloud(file_path):
    """
    Read an individual LiDAR CSV.

    Expected format:

        x,y,z
        0.0,0.0,-0.0
        1.2,2.3,0.4

    Only x, y, and z are used as coordinates.

    Filenames and other metadata are never treated as coordinates.
    """

    data = pd.read_csv(
        file_path,
        header=0,
        sep=",",
        dtype=str,
        encoding="utf-8-sig",
        on_bad_lines="skip"
    )

    data.columns = [
        str(col).strip().lower()
        for col in data.columns
    ]

    required = ["x", "y", "z"]

    if all(
        col in data.columns
        for col in required
    ):

        xyz = data[required].copy()

    else:

        # Fallback for an unusual or missing CSV header.
        data = pd.read_csv(
            file_path,
            header=None,
            sep=",",
            dtype=str,
            encoding="utf-8-sig",
            comment="#",
            on_bad_lines="skip"
        )

        if data.shape[1] < 3:

            raise ValueError(
                f"Expected three XYZ columns in "
                f"{file_path.name}; found {data.shape[1]}."
            )

        xyz = data.iloc[:, :3].copy()

    for col in xyz.columns:

        xyz[col] = pd.to_numeric(
            xyz[col].astype(str).str.strip(),
            errors="coerce"
        )

    xyz = xyz.dropna(
        subset=list(xyz.columns)
    )

    points = xyz.to_numpy(
        dtype=np.float64
    )

    points = points[
        finite_rows(points)
    ]

    if len(points) == 0:

        raise ValueError(
            f"No valid XYZ coordinates in {file_path.name}"
        )

    return points


# =====================================================================
# 11. EXTRACT LIDAR FEATURES
# =====================================================================

def extract_lidar_features(points):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    points = points[
        finite_rows(points)
    ]

    if len(points) == 0:

        raise ValueError(
            "Cannot extract features from an empty point cloud."
        )

    original_count = len(points)

    origin_mask = np.all(
        np.abs(points) <= ORIGIN_TOLERANCE,
        axis=1
    )

    origin_fraction = float(
        origin_mask.mean()
    )

    if EXCLUDE_ORIGIN_FROM_GEOMETRY:

        geometry = points[
            ~origin_mask
        ]

    else:

        geometry = points

    features = {
        "lidar_point_count": int(original_count),
        "lidar_origin_fraction": origin_fraction
    }

    if len(geometry) == 0:

        zero_features = [
            "lidar_valid_geometry_count",

            "lidar_x_mean",
            "lidar_y_mean",
            "lidar_z_mean",

            "lidar_x_std",
            "lidar_y_std",
            "lidar_z_std",

            "lidar_x_range",
            "lidar_y_range",
            "lidar_z_range",

            "lidar_z_p10",
            "lidar_z_p50",
            "lidar_z_p90",

            "lidar_xy_spread",

            "lidar_eigenvalue_1",
            "lidar_eigenvalue_2",
            "lidar_eigenvalue_3",

            "lidar_linearity",
            "lidar_planarity",
            "lidar_scattering"
        ]

        features.update({
            name: 0.0
            for name in zero_features
        })

        return features

    # Subsample very large point clouds.
    if len(geometry) > MAX_POINTS_PER_FRAME:

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        indices = rng.choice(
            len(geometry),
            size=MAX_POINTS_PER_FRAME,
            replace=False
        )

        geometry = geometry[indices]

    x = geometry[:, 0]
    y = geometry[:, 1]
    z = geometry[:, 2]

    features.update({
        "lidar_valid_geometry_count": int(len(geometry)),

        "lidar_x_mean": float(np.mean(x)),
        "lidar_y_mean": float(np.mean(y)),
        "lidar_z_mean": float(np.mean(z)),

        "lidar_x_std": float(np.std(x)),
        "lidar_y_std": float(np.std(y)),
        "lidar_z_std": float(np.std(z)),

        "lidar_x_range": float(np.ptp(x)),
        "lidar_y_range": float(np.ptp(y)),
        "lidar_z_range": float(np.ptp(z)),

        "lidar_z_p10": float(np.percentile(z, 10)),
        "lidar_z_p50": float(np.percentile(z, 50)),
        "lidar_z_p90": float(np.percentile(z, 90)),

        "lidar_xy_spread": float(
            np.sqrt(
                np.var(x) + np.var(y)
            )
        )
    })

    # Point-cloud covariance eigenvalues.
    if len(geometry) >= 3:

        centered = (
            geometry
            - geometry.mean(axis=0, keepdims=True)
        )

        covariance = np.cov(
            centered,
            rowvar=False
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        eigenvalues = np.sort(
            np.maximum(eigenvalues, 0.0)
        )[::-1]

        lambda1, lambda2, lambda3 = eigenvalues

    else:

        lambda1 = 0.0
        lambda2 = 0.0
        lambda3 = 0.0

    if lambda1 > 1e-12:

        linearity = (
            lambda1 - lambda2
        ) / lambda1

        planarity = (
            lambda2 - lambda3
        ) / lambda1

        scattering = (
            lambda3 / lambda1
        )

    else:

        linearity = 0.0
        planarity = 0.0
        scattering = 0.0

    features.update({
        "lidar_eigenvalue_1": float(lambda1),
        "lidar_eigenvalue_2": float(lambda2),
        "lidar_eigenvalue_3": float(lambda3),

        "lidar_linearity": float(linearity),
        "lidar_planarity": float(planarity),
        "lidar_scattering": float(scattering)
    })

    return features


# =====================================================================
# 12. DISCOVER LIDAR FILES
# =====================================================================

print("\n" + "=" * 75)
print("DISCOVERING LIDAR FILES")
print("=" * 75)

if not LIDAR_DIR.exists():

    raise FileNotFoundError(
        f"LiDAR directory not found:\n{LIDAR_DIR}"
    )

lidar_files = sorted(
    LIDAR_DIR.glob("velodyne__points_*.csv"),
    key=natural_sort_key
)

if not lidar_files:

    raise FileNotFoundError(
        "No LiDAR CSV files matching "
        "'velodyne__points_*.csv' were found."
    )

print("LiDAR directory:", LIDAR_DIR)
print("Number of LiDAR frames:", len(lidar_files))

print("\nFirst five frames:")

for path in lidar_files[:5]:
    print(" ", path.name)


# =====================================================================
# 13. EXTRACT FEATURES FROM EVERY LIDAR FRAME
# =====================================================================

print("\n" + "=" * 75)
print("EXTRACTING LIDAR FEATURES")
print("=" * 75)

lidar_rows = []

for index, file_path in enumerate(lidar_files):

    points = load_point_cloud(
        file_path
    )

    features = extract_lidar_features(
        points
    )

    # Keep filename and frame index as metadata only.
    lidar_rows.append({
        "lidar_frame_index": index,
        "lidar_filename": file_path.name,
        **features
    })

    if index < 3:

        print(
            f"{file_path.name}: "
            f"{len(points):,} points"
        )

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1:,}/"
            f"{len(lidar_files):,} frames"
        )


lidar_table = pd.DataFrame(
    lidar_rows
)

lidar_table.to_csv(
    OUTPUT_DIR / "lidar_extracted_features.csv",
    index=False
)

LIDAR_FEATURES = [
    col
    for col in lidar_table.columns
    if col.startswith("lidar_")
    and col not in {
        "lidar_frame_index",
        "lidar_filename"
    }
]

for col in LIDAR_FEATURES:

    lidar_table[col] = pd.to_numeric(
        lidar_table[col],
        errors="coerce"
    )

print("\nExtracted LiDAR features:", len(LIDAR_FEATURES))

print("\nLiDAR origin fraction:")
print(
    lidar_table["lidar_origin_fraction"].describe()
)


# =====================================================================
# 14. ALIGN LIDAR FEATURES WITH ML DATA
# =====================================================================

print("\n" + "=" * 75)
print("ALIGNING LIDAR AND ML DATA")
print("=" * 75)

n_ml = len(ml_data)
n_lidar = len(lidar_table)

print("ML dataset rows:", n_ml)
print("LiDAR frames:", n_lidar)

if n_ml == 0 or n_lidar == 0:

    raise ValueError(
        "The ML dataset or LiDAR feature table is empty."
    )

if STRICT_ALIGNMENT and n_ml != n_lidar:

    raise ValueError(
        "ML and LiDAR counts differ. "
        "Timestamp-based synchronization is required."
    )

if n_ml == n_lidar:

    aligned_lidar = (
        lidar_table[LIDAR_FEATURES]
        .reset_index(drop=True)
        .copy()
    )

    print(
        "Using provisional one-to-one sequence alignment."
    )

else:

    print(
        "\nWARNING: Row counts differ. "
        "Using normalized sequence interpolation."
    )

    print(
        "This does not guarantee correct temporal synchronization."
    )

    source_positions = np.linspace(
        0.0,
        1.0,
        n_lidar
    )

    target_positions = np.linspace(
        0.0,
        1.0,
        n_ml
    )

    aligned_columns = {}

    for col in LIDAR_FEATURES:

        values = lidar_table[col].to_numpy(
            dtype=np.float64
        )

        valid = np.isfinite(values)

        if not valid.any():

            aligned_columns[col] = np.zeros(
                n_ml
            )

            continue

        if not valid.all():

            values = np.interp(
                source_positions,
                source_positions[valid],
                values[valid]
            )

        aligned_columns[col] = np.interp(
            target_positions,
            source_positions,
            values
        )

    aligned_lidar = pd.DataFrame(
        aligned_columns
    )


# =====================================================================
# 15. CREATE COMPLETE MODEL DATASET
# =====================================================================

data = pd.concat(
    [
        ml_data.reset_index(drop=True),
        aligned_lidar.reset_index(drop=True)
    ],
    axis=1
)

if len(data) != len(ml_data):

    raise RuntimeError(
        "Combined dataset row count does not match the ML dataset."
    )

ALL_FEATURES = (
    SENSOR_FEATURES
    + LIDAR_FEATURES
)

# Make all model input columns numerical.
for col in ALL_FEATURES + [TARGET]:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

# Retain only rows with a valid existing target.
data = data.dropna(
    subset=[TARGET]
).reset_index(drop=True)

if len(data) < 100:

    raise ValueError(
        f"Only {len(data)} valid target rows remain."
    )

print("\nCombined dataset shape:", data.shape)
print("Total sensor features:", len(SENSOR_FEATURES))
print("Total LiDAR features:", len(LIDAR_FEATURES))
print("Total model features:", len(ALL_FEATURES))

data.to_csv(
    OUTPUT_DIR / "complete_ml_dataset_with_lidar.csv",
    index=False
)


# =====================================================================
# 16. CHRONOLOGICAL TRAIN/TEST SPLIT
# =====================================================================

# Do not randomly shuffle sequential flight observations.
# Train on the earlier segment and test on the later segment.

split_index = int(
    len(data) * (1.0 - TEST_FRACTION)
)

if split_index < 2 or split_index >= len(data):

    raise ValueError(
        "Invalid train/test split."
    )

train_data = data.iloc[
    :split_index
].copy()

test_data = data.iloc[
    split_index:
].copy()

y_train = train_data[TARGET].astype(float)
y_test = test_data[TARGET].astype(float)

# Calculate imputation values on training data only.
medians = train_data[
    ALL_FEATURES
].median()

# Features used for training.
X_train = train_data[
    ALL_FEATURES
].fillna(medians)

X_test = test_data[
    ALL_FEATURES
].fillna(medians)

test1_data = pd.DataFrame
test1_data = X_test
test1_data["error_3d_m"] = y_test
test1_data.to_csv(r"C:\University Data\AI & LLMS (Aqeel Tariq)\Research\test_data.csv",
    index=False)
X_test.drop(columns=["error_3d_m"], inplace=True)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("Training target mean:", round(y_train.mean(), 4))
print("Testing target mean:", round(y_test.mean(), 4))


# =====================================================================
# 17. TRAIN XGBOOST REGRESSION MODEL
# =====================================================================

print("\n" + "=" * 75)
print("TRAINING XGBOOST")
print("=" * 75)

model = XGBRegressor(
    **XGB_PARAMS
)

model.fit(
    X_train,
    y_train,
    verbose=False
)

print("\nXGBoost training completed.")


# =====================================================================
# 18. PREDICT ON THE TEST SET
# =====================================================================

predictions = model.predict(
    X_test
)

# Baseline: predict the training-set mean for every test row.
baseline_predictions = np.full(
    len(y_test),
    y_train.mean()
)


# =====================================================================
# 19. EVALUATE THE MODEL
# =====================================================================

model_metrics = calculate_metrics(
    y_test,
    predictions
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions
)

print("\n" + "=" * 75)
print("MODEL EVALUATION")
print("=" * 75)

print("\nXGBoost:")
print(f"MAE:  {model_metrics['MAE_m']:.6f} m")
print(f"RMSE: {model_metrics['RMSE_m']:.6f} m")
print(f"R²:   {model_metrics['R2']:.6f}")

print("\nTraining-mean baseline:")
print(f"MAE:  {baseline_metrics['MAE_m']:.6f} m")
print(f"RMSE: {baseline_metrics['RMSE_m']:.6f} m")
print(f"R²:   {baseline_metrics['R2']:.6f}")

comparison = pd.DataFrame([
    {
        "Model": "Training-mean baseline",
        **baseline_metrics
    },
    {
        "Model": "XGBoost sensor + LiDAR",
        **model_metrics
    }
])

comparison.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)

print("\nComparison:")
print(
    comparison.to_string(
        index=False
    )
)


# =====================================================================
# 20. SAVE TEST PREDICTIONS
# =====================================================================

test_predictions = pd.DataFrame({
    "actual_error_3d_m": y_test.to_numpy(),
    "predicted_error_3d_m": predictions,
    "residual_m": y_test.to_numpy() - predictions
})

test_predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# =====================================================================
# 21. FEATURE IMPORTANCE
# =====================================================================

importance = pd.DataFrame({
    "feature": ALL_FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 20 features:")

print(
    importance.head(20).to_string(
        index=False
    )
)

plt.figure(figsize=(12, 8))

top_features = importance.head(20).sort_values(
    "importance",
    ascending=True
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("XGBoost feature importance")
plt.ylabel("Feature")

plt.title(
    "Top 20 Features: GNSS Error Prediction"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300
)

plt.show()


# =====================================================================
# 22. ACTUAL VS PREDICTED ERROR
# =====================================================================

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.to_numpy(),
    label="Actual GNSS error",
    linewidth=1.5
)

plt.plot(
    predictions,
    label="XGBoost prediction",
    linewidth=1.2
)

plt.xlabel("Chronological test sample")
plt.ylabel("Existing 3D GNSS error (m)")

plt.title(
    "XGBoost: Actual vs Predicted GNSS Error"
)

plt.legend()
plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# =====================================================================
# 23. RESIDUAL ANALYSIS
# =====================================================================

residuals = (
    y_test.to_numpy()
    - predictions
)

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted GNSS error (m)")
plt.ylabel("Residual (m)")

plt.title(
    "XGBoost Residual Analysis"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residual_analysis.png",
    dpi=300
)

plt.show()


# =====================================================================
# 24. SAVE THE TRAINED MODEL
# =====================================================================

joblib.dump(
    model,
    OUTPUT_DIR / "xgboost_gnss_lidar_model.joblib"
)

joblib.dump(
    ALL_FEATURES,
    OUTPUT_DIR / "model_features.joblib"
)

joblib.dump(
    medians.to_dict(),
    OUTPUT_DIR / "feature_medians.joblib"
)

joblib.dump(
    XGB_PARAMS,
    OUTPUT_DIR / "xgboost_parameters.joblib"
)

metadata = {
    "target": TARGET,
    "number_of_rows": int(len(data)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "number_of_features": int(len(ALL_FEATURES)),
    "sensor_features": SENSOR_FEATURES,
    "lidar_features": LIDAR_FEATURES,
    "xgboost_parameters": XGB_PARAMS,
    "test_fraction": TEST_FRACTION,
    "random_state": RANDOM_STATE,
    "alignment_method": (
        "sequence-based one-to-one"
        if n_ml == n_lidar
        else "normalized sequence interpolation"
    ),
    "metrics": model_metrics
}

with open(
    OUTPUT_DIR / "model_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# =====================================================================
# 25. FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 75)
print("XGBOOST TRAINING FINISHED")
print("=" * 75)

print("\nModel saved to:")

print(
    OUTPUT_DIR
    / "xgboost_gnss_lidar_model.joblib"
)

print("\nResults saved to:")

print(
    OUTPUT_DIR.resolve()
)

print("\nFinal test metrics:")

print(
    f"MAE:  {model_metrics['MAE_m']:.6f} m"
)

print(
    f"RMSE: {model_metrics['RMSE_m']:.6f} m"
)

print(
    f"R²:   {model_metrics['R2']:.6f}"
)

print("\nRemember:")
print(
    "Verify LiDAR-to-GNSS timestamp alignment before "
    "using these results as research evidence."
)

print(
    "The reported metrics are from a chronological holdout "
    "test set; hyperparameter optimization and independent "
    "flight-level validation have not yet been performed."
)
