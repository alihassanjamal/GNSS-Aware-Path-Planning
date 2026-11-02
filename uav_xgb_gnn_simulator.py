# FIXED RELEASE: @staticmethod risk color and guarded slider callback.
"""
XGBoost + GNN GNSS-Risk-Aware UAV Path Planner
==============================================

Install:
    pip install numpy pandas scikit-learn joblib xgboost torch panda3d matplotlib

Run:
    1) Edit CONFIG paths below.
    2) Run: python uav_xgb_gnn_simulator.py

What this prototype does:
- Loads a trained XGBoost GNSS-error model and its feature list/medians.
- Reads a CSV with sensor features and ground-truth error labels.
- Produces out-of-sample XGBoost predictions if the supplied CSV is a held-out set.
- Builds a spatial graph from trajectory observations.
- Trains a small GraphSAGE-style GNN to estimate 3D GNSS error.
- Maps GNN estimates onto the simulated map using nearest-neighbour interpolation.
- Runs A* over a synthetic building map using the learned/interpolated risk field.
- Displays shortest, balanced, and risk-averse routes in Panda3D.

SCIENTIFIC LIMITATIONS:
- This is a prototype, not a validated research result.
- The CSV used for GNN training must have valid error_3d_m labels and all XGBoost features.
- For rigorous evaluation, XGBoost predictions used as GNN inputs should be out-of-fold
  or produced by a model that did not train on those rows. Do not report training-set
  predictions as unbiased test performance.
- The simulated map is not automatically geographically aligned with the real trajectory.
  The risk field is transferred from trajectory observations by nearest-neighbour mapping.
  This is an illustrative spatial transfer assumption, not a physical GNSS propagation model.
- The GNN is trained to predict node-level error; A* then chooses a route. It does not
  directly train a neural policy that outputs a complete path.
"""

import os
import math
import heapq
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectFrame, DirectButton, DirectSlider, DGG
from panda3d.core import (
    AmbientLight, DirectionalLight, Vec4, TextNode, WindowProperties,
    LineSegs, TransparencyAttrib, CardMaker, Point3
)
from direct.task import Task


# =============================================================================
# CONFIGURATION — EDIT THESE PATHS
# =============================================================================

DATA_CSV = r"C:\University Data\AI & LLMS (Aqeel Tariq)\Research\test_data.csv"
MODEL_PATH = r"C:\University Data\AI & LLMS (Aqeel Tariq)\Research\Data\GrAco_Flight3_Extracted\xgboost_final_results\xgboost_gnss_lidar_model.joblib"
FEATURES_PATH = r"C:\University Data\AI & LLMS (Aqeel Tariq)\Research\Data\GrAco_Flight3_Extracted\xgboost_final_results\model_features.joblib"
MEDIANS_PATH = r"C:\University Data\AI & LLMS (Aqeel Tariq)\Research\Data\GrAco_Flight3_Extracted\xgboost_final_results\feature_medians.joblib"

# CSV columns:
TARGET_COL = "error_3d_m"
EAST_COL = "east_m"
NORTH_COL = "north_m"
UP_COL = "up_m"
TIME_COL = "time"       # optional; not automatically used as a model feature

# If your CSV does not contain every model feature, the script stops with a clear error.
# Do not set this to True unless filling missing features with medians is scientifically
# justified for your particular feature set.
ALLOW_MEDIAN_IMPUTATION = True

# GNN settings
RANDOM_SEED = 42
TEST_SIZE = 0.20
EPOCHS = 180
LEARNING_RATE = 0.005
HIDDEN_DIM = 32
K_NEIGHBORS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GNN_MODEL_OUT = "gnn_gnss_error_model.pt"
GRAPH_PREDICTIONS_OUT = "gnn_graph_predictions.csv"
FIGURE_DIR = "gnn_training_figures"
PLOT_DPI = 160
RISK_UPDATE_HZ = 2.0
NODE_DISPLAY_LIMIT = 650
FLIGHT_LOG_OUT = "uav_flight_risk_log.csv"
NAV_NODE_OUT = "navigation_node_risk.csv"
ROUTE_COMPARISON_OUT = "route_comparison.csv"
GRAPH_PLOT_MAX_NODES = 350
SHOW_GRAPH_AT_START = False
# Dynamic replanning: illustrative online update in synthetic map coordinates.
AUTO_REPLAN_DEFAULT = True
REPLAN_RISK_THRESHOLD = 0.65
REPLAN_COOLDOWN_S = 4.0
REPLAN_MIN_IMPROVEMENT = 0.04  # fractional decrease in modeled remaining route cost
REPLAN_INFLUENCE_RADIUS_M = 13.0
REPLAN_MIN_CHANGE = 0.10
REPLAN_EVENTS_OUT = "uav_replanning_events.csv"
DYNAMIC_NODE_OUT = "dynamic_navigation_risk.csv"


# Simulator settings
WORLD_X = 100.0
WORLD_Y = 80.0
GRID_STEP = 4.0
FLIGHT_Z = 7.0
START = (8.0, 8.0)
GOAL = (92.0, 72.0)
# Route names now describe *geometric exposure scenarios*, not performance rankings.
RISK_WEIGHTS = {"dense urban": 2.0, "suburban": 5.0, "open terrain": 10.0}
ROUTE_COLORS = {"dense urban": (0.94, 0.25, 0.28, 1),
                "suburban": (0.98, 0.67, 0.16, 1),
                "open terrain": (0.12, 0.75, 0.58, 1)}
# Each scenario follows a separate set of spatial checkpoints; the A* segments
# still avoid all building footprints. The open path deliberately uses the
# outer park/perimeter, while dense follows the downtown street canyon.
SCENARIO_CHECKPOINTS = {
    "dense urban": [(12., 28.), (36., 28.), (56., 28.), (76., 32.), (92., 52.)],
    "suburban": [(12., 52.), (36., 52.), (60., 52.), (72., 56.), (88., 60.)],
    "open terrain": [(4., 4.), (4., 76.), (92., 76.)],
}
SCENARIO_DESCRIPTIONS = {
    "dense urban": "Urban street canyon | high building exposure",
    "suburban": "Mixed-density corridor | moderate exposure",
    "open terrain": "Outer park/perimeter | low building exposure",
}

# Building: center_x, center_y, width, depth, height
BUILDINGS = [
    (20, 16, 12, 12, 17), (40, 14, 14, 18, 25), (62, 15, 13, 12, 20),
    (82, 20, 12, 14, 16), (17, 39, 14, 15, 24), (39, 39, 13, 13, 31),
    (62, 40, 17, 14, 27), (84, 45, 11, 16, 22), (23, 62, 16, 12, 18),
    (48, 64, 14, 12, 23), (72, 64, 13, 12, 19),
]
# Physical proxy: building proximity is an *illustrative* obstruction/exposure
# score, not a GNSS ray-tracing model. It is kept separate from GNN error.
BUILDING_EXPOSURE_RADIUS_M = 16.0



# =============================================================================
# MODEL / GRAPH UTILITIES
# =============================================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_file(path, description):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {description}:\n  {path}\n"
            "Edit the CONFIGURATION paths at the top of this script."
        )


def load_xgb_artifacts():
    require_file(MODEL_PATH, "trained XGBoost model")
    require_file(FEATURES_PATH, "XGBoost feature list")
    require_file(MEDIANS_PATH, "feature medians")
    model = joblib.load(MODEL_PATH)
    model_features = list(joblib.load(FEATURES_PATH))
    medians = joblib.load(MEDIANS_PATH)
    return model, model_features, medians


def load_and_predict_csv():
    require_file(DATA_CSV, "input CSV")
    df = pd.read_csv(DATA_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"'{TARGET_COL}' is missing from {DATA_CSV}. "
            "A supervised GNN needs real ground-truth error labels for training."
        )
    for col in (EAST_COL, NORTH_COL):
        if col not in df.columns:
            raise ValueError(f"Required spatial column '{col}' is missing from the CSV.")

    model, model_features, medians = load_xgb_artifacts()
    missing = [c for c in model_features if c not in df.columns]
    if missing and not ALLOW_MEDIAN_IMPUTATION:
        raise ValueError(
            "CSV is missing model features:\n  " + "\n  ".join(missing) +
            "\nProvide a feature-complete CSV synchronized with the XGBoost training data."
        )
    if missing:
        print("\nWARNING: The following XGBoost features are missing and will be median-filled:")
        for c in missing:
            print("  -", c)
        print("These predictions may be unreliable; this is only a pipeline fallback.\n")

    X = pd.DataFrame(index=df.index)
    for col in model_features:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            X[col] = np.nan

    for col in model_features:
        fill_value = medians.get(col, np.nan) if hasattr(medians, "get") else np.nan
        if pd.isna(fill_value):
            fill_value = X[col].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(float(fill_value))

    pred = np.asarray(model.predict(X[model_features]), dtype=float)
    pred = np.maximum(pred, 0.0)
    df["xgb_predicted_error_3d_m"] = pred
    print(f"Loaded {len(df)} rows; XGBoost predictions generated.")
    print(df[["xgb_predicted_error_3d_m", TARGET_COL]].head())
    return df, model_features


def build_spatial_edges(coords, k=6):
    """Build undirected k-nearest-neighbour graph using EN coordinates."""
    n = len(coords)
    if n < 2:
        raise ValueError("At least two rows are required to build a spatial graph.")
    k = max(1, min(k, n - 1))
    nn_model = NearestNeighbors(n_neighbors=k + 1)
    nn_model.fit(coords)
    indices = nn_model.kneighbors(coords, return_distance=False)
    edges = set()
    for i, row in enumerate(indices):
        for j in row:
            j = int(j)
            if i != j:
                edges.add((min(i, j), max(i, j)))
    edge_pairs = sorted(edges)
    src = np.array([e[0] for e in edge_pairs] + [e[1] for e in edge_pairs], dtype=np.int64)
    dst = np.array([e[1] for e in edge_pairs] + [e[0] for e in edge_pairs], dtype=np.int64)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


class GraphSAGERegressor(nn.Module):
    """Small mean-aggregation GraphSAGE-style node regressor."""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lin_self1 = nn.Linear(input_dim, hidden_dim)
        self.lin_neigh1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.lin_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                 nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    @staticmethod
    def mean_neighbors(x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        counts = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        counts.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=x.device, dtype=x.dtype))
        return agg / counts.clamp(min=1.0)

    def forward(self, x, edge_index):
        neigh1 = self.mean_neighbors(x, edge_index)
        h = F.relu(self.lin_self1(x) + self.lin_neigh1(neigh1))
        neigh2 = self.mean_neighbors(h, edge_index)
        h = F.relu(self.lin_self2(h) + self.lin_neigh2(neigh2))
        return F.softplus(self.out(h)).squeeze(-1)


def train_gnn(df):
    """
    Train a node-level GNN. Inputs contain XGBoost predictions and spatial/sensor features.
    The target is the real error_3d_m column. Error components and ground-truth coordinates
    are excluded from input features to prevent direct target leakage.
    """
    set_seed(RANDOM_SEED)
    coords = df[[EAST_COL, NORTH_COL]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(float)

    feature_candidates = [
        "xgb_predicted_error_3d_m",
        EAST_COL, NORTH_COL, UP_COL,
        "roll", "pitch", "yaw", "wbx", "wby", "wbz", "abx", "aby", "abz",
        "satellite_count", "pdop", "cn0_mean",
    ]
    feature_cols = [c for c in feature_candidates if c in df.columns]
    if "xgb_predicted_error_3d_m" not in feature_cols:
        raise RuntimeError("XGBoost prediction column was not generated.")

    Xdf = df[feature_cols].copy()
    for col in Xdf:
        Xdf[col] = pd.to_numeric(Xdf[col], errors="coerce")
        Xdf[col] = Xdf[col].replace([np.inf, -np.inf], np.nan)
        Xdf[col] = Xdf[col].fillna(Xdf[col].median())
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(target) & np.isfinite(Xdf.to_numpy()).all(axis=1)
    if valid.sum() < 30:
        raise ValueError(f"Only {valid.sum()} valid rows; need at least 30 for a useful prototype.")

    # Keep only valid rows and create a graph over these rows.
    dfv = df.loc[valid].reset_index(drop=True)
    Xdf = Xdf.loc[valid].reset_index(drop=True)
    coords = coords[valid]
    target = target[valid]
    Xraw = Xdf.to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw).astype(np.float32)
    edge_index = build_spatial_edges(coords, K_NEIGHBORS)

    # Split by rows only for a baseline. For research, replace with flight-wise or
    # chronological split and ensure no nearby/time-adjacent leakage.
    indices = np.arange(len(target))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_mask = torch.zeros(len(target), dtype=torch.bool)
    test_mask = torch.zeros(len(target), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    x_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(target, dtype=torch.float32, device=DEVICE)
    edges_t = edge_index.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)

    gnn = GraphSAGERegressor(X.shape[1], HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Keep per-epoch diagnostics so training progress can be plotted and reviewed.
    history = {"epoch": [], "train_rmse": [], "holdout_rmse": [], "holdout_mae": [], "xgb_holdout_rmse": []}
    xgb_pred_t = torch.tensor(dfv["xgb_predicted_error_3d_m"].to_numpy(), dtype=torch.float32, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        gnn.train()
        optimizer.zero_grad()
        output = gnn(x_t, edges_t)
        loss = F.mse_loss(output[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()
        gnn.eval()
        with torch.no_grad():
            out = gnn(x_t, edges_t)
            train_rmse = torch.sqrt(F.mse_loss(out[train_mask], y_t[train_mask])).item()
            test_rmse = torch.sqrt(F.mse_loss(out[test_mask], y_t[test_mask])).item()
            test_mae = torch.mean(torch.abs(out[test_mask] - y_t[test_mask])).item()
            xgb_rmse = torch.sqrt(F.mse_loss(xgb_pred_t[test_mask], y_t[test_mask])).item()
        history["epoch"].append(epoch)
        history["train_rmse"].append(train_rmse)
        history["holdout_rmse"].append(test_rmse)
        history["holdout_mae"].append(test_mae)
        history["xgb_holdout_rmse"].append(xgb_rmse)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d}/{EPOCHS} | train RMSE={train_rmse:.3f} m "
                  f"| holdout GNN RMSE={test_rmse:.3f} m | holdout MAE={test_mae:.3f} m "
                  f"| XGB RMSE={xgb_rmse:.3f} m")

    gnn.eval()
    with torch.no_grad():
        gnn_pred = gnn(x_t, edges_t).cpu().numpy()

    torch.save({
        "state_dict": gnn.cpu().state_dict(),
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "hidden_dim": HIDDEN_DIM,
        "k_neighbors": K_NEIGHBORS,
    }, GNN_MODEL_OUT)

    dfv["gnn_predicted_error_3d_m"] = gnn_pred
    dfv["xgb_predicted_error_3d_m"] = np.maximum(
        dfv["xgb_predicted_error_3d_m"].to_numpy(float), 0.0
    )
    dfv["gnn_node_id"] = np.arange(len(dfv))
    dfv["dataset_split"] = np.where(test_mask.cpu().numpy(), "holdout", "train")
    dfv["gnn_residual_m"] = gnn_pred - target
    dfv.to_csv(GRAPH_PREDICTIONS_OUT, index=False)

    # -------------------------------------------------------------------------
    # Training figures and evaluation diagnostics
    # -------------------------------------------------------------------------
    fig_dir = Path(FIGURE_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(fig_dir / "training_history.csv", index=False)

    # Graph structure and per-node error diagnostics (full graph exported, subset plotted).
    undirected_edges = edge_index.cpu().numpy().T
    undirected_edges = undirected_edges[undirected_edges[:, 0] < undirected_edges[:, 1]]
    pd.DataFrame(undirected_edges, columns=["source_node_id", "target_node_id"]).to_csv(
        fig_dir / "gnn_graph_edges.csv", index=False)
    node_table = pd.DataFrame({
        "node_id": np.arange(len(dfv)), "east_m": coords[:, 0], "north_m": coords[:, 1],
        "actual_error_m": target, "xgb_error_m": dfv["xgb_predicted_error_3d_m"].to_numpy(),
        "gnn_error_m": gnn_pred, "gnn_residual_m": gnn_pred-target,
        "split": dfv["dataset_split"].to_numpy()})
    node_table.to_csv(fig_dir / "gnn_node_errors.csv", index=False)
    rng = np.random.default_rng(RANDOM_SEED)
    sample = np.sort(rng.choice(len(dfv), min(GRAPH_PLOT_MAX_NODES, len(dfv)), replace=False))
    selected = set(sample.tolist())
    fig, ax = plt.subplots(figsize=(11, 8))
    for a, b in undirected_edges:
        if int(a) in selected and int(b) in selected:
            ax.plot(coords[[a,b], 0], coords[[a,b], 1], color="gray", alpha=.17, linewidth=.6, zorder=1)
    dots = ax.scatter(coords[sample, 0], coords[sample, 1], c=gnn_pred[sample],
                      cmap="turbo", s=26, edgecolors="black", linewidths=.15, zorder=2)
    fig.colorbar(dots, ax=ax, label="GNN predicted 3D error (m)")
    ax.set(xlabel="East (m)", ylabel="North (m)",
           title=f"GNN spatial graph: {len(sample)} displayed / {len(dfv)} nodes")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(fig_dir / "06_gnn_nodes_and_edges.png", dpi=PLOT_DPI); plt.close(fig)

    ordered = np.argsort(coords[:, 0], kind="stable")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(dfv)), target[ordered], label="Ground truth", alpha=.8, linewidth=1)
    ax.plot(np.arange(len(dfv)), gnn_pred[ordered], label="GNN estimate", alpha=.8, linewidth=1)
    ax.set(xlabel="Node index (sorted by East coordinate)", ylabel="3D error (m)",
           title="Node-by-node GNSS error estimates")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(fig_dir / "07_per_node_error_estimation.png", dpi=PLOT_DPI); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    points = ax.scatter(coords[:,0], coords[:,1], c=gnn_pred-target, cmap="coolwarm", s=12)
    fig.colorbar(points, ax=ax, label="GNN residual: predicted - actual (m)")
    ax.set(xlabel="East (m)", ylabel="North (m)", title="Spatial distribution of node prediction residuals")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(fig_dir / "08_node_residual_map.png", dpi=PLOT_DPI); plt.close(fig)

    # Figure 1: training and holdout RMSE over epochs.
    plt.figure(figsize=(9, 5.5))
    plt.plot(history["epoch"], history["train_rmse"], label="Training RMSE")
    plt.plot(history["epoch"], history["holdout_rmse"], label="Holdout RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (m)")
    plt.title("GNN Training Progress")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "01_training_holdout_rmse.png", dpi=PLOT_DPI)
    plt.close()

    # Metrics calculated on the held-out rows.
    holdout_actual = target[test_mask.cpu().numpy()]
    holdout_gnn = gnn_pred[test_mask.cpu().numpy()]
    holdout_xgb = dfv.loc[test_mask.cpu().numpy(), "xgb_predicted_error_3d_m"].to_numpy(float)
    rmse = lambda pred, actual: float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = lambda pred, actual: float(np.mean(np.abs(pred - actual)))
    metrics = pd.DataFrame([
        {"model": "XGBoost", "holdout_RMSE_m": rmse(holdout_xgb, holdout_actual), "holdout_MAE_m": mae(holdout_xgb, holdout_actual)},
        {"model": "GNN", "holdout_RMSE_m": rmse(holdout_gnn, holdout_actual), "holdout_MAE_m": mae(holdout_gnn, holdout_actual)},
    ])
    metrics.to_csv(fig_dir / "holdout_metrics.csv", index=False)

    # Figure 2: XGBoost versus GNN metric comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(metrics["model"], metrics["holdout_RMSE_m"])
    axes[0].set_title("Holdout RMSE")
    axes[0].set_ylabel("RMSE (m)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(metrics["model"], metrics["holdout_MAE_m"])
    axes[1].set_title("Holdout MAE")
    axes[1].set_ylabel("MAE (m)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("XGBoost vs GNN on Holdout Rows")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_xgb_vs_gnn_metrics.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Figure 3: predicted versus actual on held-out rows.
    low = float(min(holdout_actual.min(), holdout_xgb.min(), holdout_gnn.min()))
    high = float(max(holdout_actual.max(), holdout_xgb.max(), holdout_gnn.max()))
    if high <= low:
        high = low + 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].scatter(holdout_actual, holdout_xgb, s=14, alpha=0.6)
    axes[0].plot([low, high], [low, high], linestyle="--")
    axes[0].set_title("XGBoost: Predicted vs Actual")
    axes[0].set_xlabel("Actual error (m)")
    axes[0].set_ylabel("Predicted error (m)")
    axes[1].scatter(holdout_actual, holdout_gnn, s=14, alpha=0.6)
    axes[1].plot([low, high], [low, high], linestyle="--")
    axes[1].set_title("GNN: Predicted vs Actual")
    axes[1].set_xlabel("Actual error (m)")
    axes[1].set_ylabel("Predicted error (m)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_predicted_vs_actual.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Figure 4: residual distributions on the holdout rows.
    plt.figure(figsize=(9, 5))
    plt.hist(holdout_xgb - holdout_actual, bins=35, alpha=0.6, label="XGBoost residual")
    plt.hist(holdout_gnn - holdout_actual, bins=35, alpha=0.6, label="GNN residual")
    plt.xlabel("Residual: predicted - actual (m)")
    plt.ylabel("Count")
    plt.title("Holdout Residual Distributions")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "04_residual_distributions.png", dpi=PLOT_DPI)
    plt.close()


    # Figure 5: spatial plots for observed trajectory coordinates.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [(target, "Ground-truth error"), (dfv["xgb_predicted_error_3d_m"].to_numpy(), "XGBoost predicted error"), (dfv["gnn_predicted_error_3d_m"].to_numpy(), "GNN predicted error")]
    for ax, (values, title) in zip(axes, panels):
        points = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=12)
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
        fig.colorbar(points, ax=ax, label="Error (m)")
    fig.suptitle("Spatial GNSS Error Along the Available Trajectory")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_spatial_error_maps.png", dpi=PLOT_DPI)
    plt.close(fig)

    print(f"\nSaved GNN checkpoint: {GNN_MODEL_OUT}")
    print(f"Saved node predictions: {GRAPH_PREDICTIONS_OUT}")
    print(f"Saved training history and figures in: {fig_dir.resolve()}")
    print(metrics.to_string(index=False))
    print("Research caution: random row split is only a pipeline check; use a chronological or flight-wise split for publication.")
    return dfv, feature_cols


# =============================================================================
# RISK MAP TRANSFER + A* PLANNING
# =============================================================================

def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class RiskMap:
    """
    Transfers predicted error values from real trajectory coordinates to simulator
    coordinates by nearest-neighbour mapping. This is only valid as a demonstration
    unless the coordinate frames are aligned and spatial transfer is justified.
    """
    def __init__(self, predicted_df):
        self.xy = predicted_df[[EAST_COL, NORTH_COL]].to_numpy(float)
        self.err = predicted_df["gnn_predicted_error_3d_m"].to_numpy(float)
        self.xgb_err = predicted_df["xgb_predicted_error_3d_m"].to_numpy(float)
        good = np.isfinite(self.xy).all(axis=1) & np.isfinite(self.err)
        self.xy, self.err = self.xy[good], np.maximum(self.err[good], 0.0)
        if len(self.xy) == 0:
            raise ValueError("No valid GNN predictions available for risk map.")
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.xy)
        # Robust scale for display/cost normalization.
        self.scale = max(float(np.percentile(self.err, 90)), 1.0)

    def error_at(self, x, y):
        _, idx = self.nn.kneighbors(np.array([[x, y]], dtype=float))
        return float(self.err[int(idx[0, 0])])

    def risk_at(self, x, y):
        return clamp(self.error_at(x, y) / self.scale, 0.0, 1.0)



class LiveGNNInference:
    """Real GraphSAGE forward pass on every sample, with replayed sensor rows.

    WARNING: A nearest recorded sensor observation is replayed at each synthetic
    UAV location. The simulation map is not georeferenced to the flight data.
    This is live MODEL INFERENCE, not live GNSS measurement or validated mapping.
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        checkpoint = torch.load(GNN_MODEL_OUT, map_location="cpu", weights_only=False)
        self.cols = checkpoint["feature_cols"]
        self.mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.model = GraphSAGERegressor(len(self.cols), checkpoint["hidden_dim"]).cpu().eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.coords = self.df[[EAST_COL, NORTH_COL]].to_numpy(float)
        self.lookup = NearestNeighbors(n_neighbors=1).fit(self.coords)
        self.neighbors = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(self.coords))).fit(self.coords)
        X = self.df[self.cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)
        self.base_features = torch.tensor(((X.to_numpy(np.float32) - self.mean) / np.maximum(self.scale, 1e-8)), dtype=torch.float32)
        self.base_edges = build_spatial_edges(self.coords, K_NEIGHBORS)
        self.xgb, self.xgb_cols, self.xgb_medians = load_xgb_artifacts()

    def infer(self, x, y):
        # DEMO REPLAY: choose a recorded sensor packet, not a simulated GNSS sensor.
        row_id = int(self.lookup.kneighbors([[x, y]], return_distance=False)[0, 0])
        record = self.df.iloc[row_id]
        row = {}
        for name in self.xgb_cols:
            value = pd.to_numeric(record.get(name, np.nan), errors="coerce")
            if not np.isfinite(value):
                value = self.xgb_medians.get(name, 0.0) if hasattr(self.xgb_medians, "get") else 0.0
            row[name] = float(value) if np.isfinite(value) else 0.0
        xgb_error = max(0.0, float(self.xgb.predict(pd.DataFrame([row], columns=self.xgb_cols))[0]))
        node = self.df.loc[row_id, self.cols].copy()
        node["xgb_predicted_error_3d_m"] = xgb_error
        values = pd.to_numeric(node, errors="coerce").to_numpy(np.float32)
        fallback = self.base_features[row_id].numpy() * self.scale + self.mean
        values = np.where(np.isfinite(values), values, fallback)
        feature = torch.tensor(((values - self.mean) / np.maximum(self.scale, 1e-8))[None, :], dtype=torch.float32)
        # Attach a fresh query node to its spatial neighbors and run the GNN now.
        # The query is positioned at the REPLAYED observation, not fictitious city ENU.
        replay_xy = self.coords[row_id]
        ids = self.neighbors.kneighbors([replay_xy], return_distance=False)[0]
        q = len(self.df)
        new_edges = torch.tensor([[q] * len(ids) + ids.tolist(), ids.tolist() + [q] * len(ids)], dtype=torch.long)
        edges = torch.cat((self.base_edges, new_edges), dim=1)
        with torch.inference_mode():
            output = self.model(torch.cat((self.base_features, feature), dim=0), edges)
            gnn_error = max(0.0, float(output[-1].item()))
        return {"gnn_error_m": gnn_error, "xgb_error_m": xgb_error,
                "source_row": row_id, "replay_east_m": float(replay_xy[0]),
                "replay_north_m": float(replay_xy[1])}

class AdaptiveRiskMap:
    """Online synthetic-map risk overlay.

    Each *live* GNN result updates a local risk patch around the current synthetic
    UAV position. This does NOT establish geospatial correspondence between
    recorded sensor rows and synthetic buildings; it is a demonstrator.
    """
    def __init__(self, base, nodes):
        self.base = base
        self.scale = base.scale
        self.nodes = np.asarray(nodes, dtype=float)
        self.base_values = np.asarray([base.risk_at(*p) for p in nodes], dtype=float)
        self.values = self.base_values.copy()
        self.node_index = {tuple(p): i for i, p in enumerate(nodes)}
        self.tree = NearestNeighbors(n_neighbors=1).fit(self.nodes)
        self.updates = 0

    def risk_at(self, x, y):
        key = (float(x), float(y))
        idx = self.node_index.get(key)
        if idx is None:
            idx = int(self.tree.kneighbors([[x, y]], return_distance=False)[0, 0])
        return float(self.values[idx])

    def error_at(self, x, y):
        return self.risk_at(x, y) * self.scale

    def assimilate(self, x, y, observed_risk):
        d = np.linalg.norm(self.nodes - np.array([x, y]), axis=1)
        weights = np.exp(-0.5 * (d / REPLAN_INFLUENCE_RADIUS_M) ** 2)
        weights[d > 2.5 * REPLAN_INFLUENCE_RADIUS_M] = 0.0
        before = self.values.copy()
        # Maintain a persistent, bounded local update; nearby nodes respond most.
        self.values = np.clip(self.values + 0.8 * weights * (observed_risk - self.values), 0, 1)
        self.updates += 1
        return float(np.max(np.abs(self.values - before)))


def modeled_path_cost(path, risk_weight, risk_map):
    if not path or len(path) < 2:
        return float("inf")
    return sum(distance(a, b) * (1 + risk_weight * risk_map.risk_at(*b))
               for a, b in zip(path[:-1], path[1:]))


def is_blocked(x, y, margin=1.6):
    for bx, by, width, depth, _height in BUILDINGS:
        if (bx - width / 2 - margin <= x <= bx + width / 2 + margin and
                by - depth / 2 - margin <= y <= by + depth / 2 + margin):
            return True
    return False


def segment_clear(a, b, samples=16):
    for t in np.linspace(0.0, 1.0, samples):
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if is_blocked(x, y):
            return False
    return True


def create_grid_nodes():
    return [(float(x), float(y))
            for y in np.arange(GRID_STEP, WORLD_Y, GRID_STEP)
            for x in np.arange(GRID_STEP, WORLD_X, GRID_STEP)
            if not is_blocked(float(x), float(y))]


def astar_path(start, goal, risk_weight, nodes, risk_map):
    node_set = set(nodes)
    start_node = min(nodes, key=lambda p: distance(p, start))
    goal_node = min(nodes, key=lambda p: distance(p, goal))
    offsets = [(-GRID_STEP, 0), (GRID_STEP, 0), (0, -GRID_STEP), (0, GRID_STEP),
               (-GRID_STEP, -GRID_STEP), (-GRID_STEP, GRID_STEP),
               (GRID_STEP, -GRID_STEP), (GRID_STEP, GRID_STEP)]
    open_heap = [(distance(start_node, goal_node), start_node)]
    came_from, g_score, closed = {}, {start_node: 0.0}, set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_node:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            path[0], path[-1] = start, goal
            return path
        closed.add(current)

        for dx, dy in offsets:
            neighbor = (round(current[0] + dx, 6), round(current[1] + dy, 6))
            if neighbor not in node_set or is_blocked(*neighbor):
                continue
            # Avoid diagonal corner-cutting through obstacle boundaries.
            if dx and dy and (is_blocked(current[0] + dx, current[1]) or
                              is_blocked(current[0], current[1] + dy)):
                continue
            d = math.hypot(dx, dy)
            risk = risk_map.risk_at(*neighbor)
            step_cost = d * (1.0 + risk_weight * risk)
            new_g = g_score[current] + step_cost
            if new_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = new_g
                heapq.heappush(open_heap, (new_g + distance(neighbor, goal_node), neighbor))
    return None


def building_exposure(x, y):
    """0..1 urban-density proxy based on distance to nearby footprints."""
    total = 0.0
    for bx, by, w, d, height in BUILDINGS:
        dx = max(abs(x-bx) - w/2, 0.0)
        dy = max(abs(y-by) - d/2, 0.0)
        edge_distance = math.hypot(dx, dy)
        total += math.exp(-0.5 * (edge_distance / 7.0)**2) * min(height / 28., 1.2)
    return float(np.clip(total / 2.7, 0., 1.))


def plan_scenario(name, start, goal, nodes, risk_map):
    """A* through fixed geographic checkpoints, preserving scenario meaning."""
    anchors = [start] + SCENARIO_CHECKPOINTS[name] + [goal]
    path = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        leg = astar_path(a, b, RISK_WEIGHTS[name], nodes, risk_map)
        if not leg or len(leg) < 2 or not segment_clear(leg[0], leg[1]):
            raise RuntimeError(f"No collision-free A* segment for {name}: {a} -> {b}")
        if not segment_clear(leg[-2], leg[-1]):
            raise RuntimeError(f"Unsafe goal connector in {name}: {a} -> {b}")
        path.extend(leg if not path else leg[1:])
    return path


def route_exposure_metrics(path):
    if not path or len(path) < 2:
        return {"mean_building_exposure": float('nan'), "max_building_exposure": float('nan')}
    distances, exposures = [], []
    for a,b in zip(path[:-1], path[1:]):
        length = distance(a,b)
        midpoint = ((a[0]+b[0])/2., (a[1]+b[1])/2.)
        distances.append(length)
        exposures.append(building_exposure(*midpoint))
    return {"mean_building_exposure": float(np.average(exposures, weights=distances)),
            "max_building_exposure": float(max(exposures))}


def route_metrics(path, risk_map):
    if not path or len(path) < 2:
        return {"distance": 0.0, "mean_error": float("nan"), "max_error": float("nan")}
    ds, errors = [], []
    for a, b in zip(path[:-1], path[1:]):
        d = distance(a, b)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ds.append(d)
        errors.append(risk_map.error_at(*mid))
    total = sum(ds)
    mean_error = float(np.average(errors, weights=ds)) if total else 0.0
    return {"distance": total, "mean_error": mean_error, "max_error": max(errors, default=0.0)}


# =============================================================================
# PANDA3D VISUALIZATION
# =============================================================================

def draw_line(parent, points, color, thickness=3.0, name="line"):
    segs = LineSegs(name)
    segs.setThickness(thickness)
    segs.setColor(*color)
    for i, p in enumerate(points):
        if i == 0:
            segs.moveTo(*p)
        else:
            segs.drawTo(*p)
    return parent.attachNewNode(segs.create())


class UAVRiskSimulator(ShowBase):
    def __init__(self, risk_map, predicted_df=None, live_model=None):
        super().__init__()
        self.risk_map = risk_map
        self.live_model = live_model
        self.last_inference = None
        self.disableMouse()
        self.setBackgroundColor(0.81, 0.87, 0.92, 1)
        wp = WindowProperties()
        wp.setTitle("URBAN UAV MISSION LAB | Dense / Mixed / Open | GraphSAGE + A*")
        wp.setSize(1600, 940)
        self.win.requestProperties(wp)
        self.camera_target = Point3(WORLD_X / 2, WORLD_Y / 2, 0)
        self.cam_yaw = 42.0
        self.cam_pitch = 38.0
        self.cam_distance = 126.0
        self.mouse_drag = False
        self.mouse_last = None
        self._update_camera()

        self.world = self.render.attachNewNode("world")
        self.risk_root = self.world.attachNewNode("risk-overlay")
        self.paths_root = self.world.attachNewNode("candidate-paths")
        self.graph_root = self.world.attachNewNode("gnn-observation-graph")
        self.predicted_df = predicted_df
        self._setup_lighting()
        self._create_ground()
        self._create_roads()
        self._create_open_park()
        self._create_buildings()
        self._create_graph_overlay()
        self._create_risk_overlay()

        self.grid_nodes = create_grid_nodes()
        # These are A* navigation nodes, not observed GNN graph nodes. The node risk
        # is a lookup of PRECOMPUTED GNN predictions via the illustrative RiskMap.
        self.nav_xy = np.asarray(self.grid_nodes, dtype=float)
        self.nav_errors = np.asarray([self.risk_map.error_at(*p) for p in self.grid_nodes])
        self.nav_risks = np.clip(self.nav_errors / self.risk_map.scale, 0, 1)
        self.nav_nn = NearestNeighbors(n_neighbors=1).fit(self.nav_xy)
        self.adaptive_map = AdaptiveRiskMap(self.risk_map, self.grid_nodes)
        self.auto_replan = AUTO_REPLAN_DEFAULT
        self.replan_events = []
        self.replan_count = 0
        self.last_replan_s = -REPLAN_COOLDOWN_S
        self.last_live_risk = None
        self.last_risk_change = 0.0
        self.nav_visuals = {}
        self._create_navigation_nodes()
        pd.DataFrame({"node_id": np.arange(len(self.grid_nodes)),
                      "x_m": self.nav_xy[:,0], "y_m": self.nav_xy[:,1],
                      "gnn_transferred_error_m": self.nav_errors,
                      "normalized_risk": self.nav_risks,
                      "coordinate_mapping": "illustrative_nearest_observation"}).to_csv(NAV_NODE_OUT, index=False)
        self.routes = {}
        for name, weight in RISK_WEIGHTS.items():
            path = plan_scenario(name, START, GOAL, self.grid_nodes, self.risk_map)
            if path:
                self.routes[name] = path
        if not self.routes:
            raise RuntimeError("No route found. Adjust start/goal or building positions.")

        comparison = []
        for route_name, path in self.routes.items():
            m = route_metrics(path, self.risk_map)
            m.update(route_exposure_metrics(path))
            comparison.append({"route": route_name, "risk_weight": RISK_WEIGHTS[route_name],
                               "distance_m": m["distance"],
                               "mean_predicted_error_m": m["mean_error"],
                               "max_predicted_error_m": m["max_error"],
                               "waypoints": len(path),
                               "mean_building_exposure_proxy": m["mean_building_exposure"],
                               "max_building_exposure_proxy": m["max_building_exposure"],
                               "evaluation": "illustrative model prediction; not measured flight error"})
        pd.DataFrame(comparison).to_csv(ROUTE_COMPARISON_OUT, index=False)
        print("\nROUTE COMPARISON (MODEL-BASED, NOT MEASURED):")
        print(pd.DataFrame(comparison).to_string(index=False))
        self.active_route = "dense urban"
        self.show_paths = True
        self.show_risk = False
        self.risk_root.hide()
        self.is_moving = False
        self.speed = 8.0
        self.travelled = 0.0
        self.flight_elapsed = 0.0
        self.sample_accum = 0.0
        self.flight_log = []
        self.replan_events = []
        self.replan_count = 0
        self.last_replan_s = -REPLAN_COOLDOWN_S
        self.adaptive_map.values = self.adaptive_map.base_values.copy()
        self.nav_risks = self.adaptive_map.values.copy()
        self.nav_errors = self.nav_risks * self.adaptive_map.scale
        self._refresh_dynamic_node_colors()
        self._create_risk_overlay()
        self.risk_root.hide()
        self.last_live_risk = None
        self.visited_node_ids = set()
        self.current_node_id = None
        self._create_markers()
        self._create_uav()
        self._create_hud()
        self._create_gui()
        self._select_route(self.active_route)

        self.accept("1", self._select_route, ["dense urban"])
        self.accept("2", self._select_route, ["suburban"])
        self.accept("3", self._select_route, ["open terrain"])
        self.accept("space", self._toggle_flight)
        self.accept("r", self._reset_flight)
        self.accept("h", self._toggle_risk)
        self.accept("p", self._toggle_paths)
        self.accept("g", self._toggle_graph)
        self.accept("n", self._toggle_nav_nodes)
        self.accept("v", self._toggle_node_labels)
        self.accept("f", self._toggle_follow_camera)
        self.accept("e", self._export_flight_log)
        self.accept("a", self._toggle_auto_replan)
        self.accept("b", self._manual_replan)
        self.accept("mouse1", self._mouse_down)
        self.accept("mouse1-up", self._mouse_up)
        self.accept("wheel_up", self._zoom, [0.88])
        self.accept("wheel_down", self._zoom, [1.13])
        self.accept("c", self._reset_camera)
        self.accept("arrow_left", self._orbit, [-8, 0])
        self.accept("arrow_right", self._orbit, [8, 0])
        self.accept("arrow_up", self._orbit, [0, 5])
        self.accept("arrow_down", self._orbit, [0, -5])
        self.accept("mouse3", self._inspect_node_under_cursor)
        self.accept("escape", self.userExit)
        self.taskMgr.add(self._animate, "animate-uav")
        self.taskMgr.add(self._camera_task, "camera-drag")
        self.follow_camera = False
        self.show_node_labels = False
        self._sample_risk(force=True)
        self._refresh_gui()

    def _setup_lighting(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.68, 0.70, 0.76, 1))
        self.render.setLight(self.render.attachNewNode(ambient))
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.95, 0.93, 0.86, 1))
        sn = self.render.attachNewNode(sun)
        sn.setHpr(-35, -55, 0)
        self.render.setLight(sn)

    def _box(self, parent, name, pos, scale, color):
        """Untextured cuboid with explicit lighting; all scales are half-extents."""
        obj = self.loader.loadModel("models/box")
        obj.setName(name)
        obj.reparentTo(parent)
        obj.setPos(*pos)
        obj.setScale(*scale)
        obj.setTextureOff(1)
        obj.setColor(*color)
        obj.setTwoSided(False)
        return obj

    def _sphere(self, name, pos, scale, color):
        obj = self.loader.loadModel("models/misc/sphere")
        obj.setName(name)
        obj.reparentTo(self.world)
        obj.setPos(*pos)
        obj.setScale(scale)
        obj.setTextureOff(1)
        obj.setColor(*color)
        return obj

    def _create_ground(self):
        # The base is a single matte surface, not a colored graph plane.
        self._box(self.world, "terrain", (WORLD_X/2, WORLD_Y/2, -0.18),
                  (WORLD_X/2+3, WORLD_Y/2+3, 0.18), (.64, .69, .67, 1))

    def _create_roads(self):
        """Connected road corridors with subtle curbs and painted lane markings."""
        asphalt = (.19,.23,.27,1)
        curb = (.74,.75,.72,1)
        marking = (.79,.79,.69,1)
        # Long uninterrupted corridors; do not add free-floating strips.
        for x in (8,30,52,74,96):
            self._box(self.world,"road-ns",(x,WORLD_Y/2,.018),(3.35,WORLD_Y/2,.022),asphalt)
            for edge in (-3.55,3.55):
                self._box(self.world,"curb-ns",(x+edge,WORLD_Y/2,.065),(.16,WORLD_Y/2,.055),curb)
            for y in np.arange(6,WORLD_Y-5,10):
                self._box(self.world,"stripe-ns",(x,float(y),.046),(.055,1.25,.009),marking)
        for y in (8,28,52,76):
            self._box(self.world,"road-ew",(WORLD_X/2,y,.025),(WORLD_X/2,3.35,.025),asphalt)
            for edge in (-3.55,3.55):
                self._box(self.world,"curb-ew",(WORLD_X/2,y+edge,.065),(WORLD_X/2,.16,.055),curb)
            for x in np.arange(6,WORLD_X-5,10):
                self._box(self.world,"stripe-ew",(float(x),y,.053),(1.25,.055,.009),marking)

    def _create_graph_overlay(self):
        if self.predicted_df is None:
            self.graph_root.hide()
            return
        df = self.predicted_df
        pts = df[[EAST_COL, NORTH_COL]].to_numpy(float)
        values = df["gnn_predicted_error_3d_m"].to_numpy(float)
        # Graph coordinates may not match the synthetic world; fit for DISPLAY ONLY.
        lo = np.nanmin(pts, axis=0); hi = np.nanmax(pts, axis=0)
        span = np.maximum(hi-lo, 1e-6)
        uv = (pts-lo)/span
        xy = np.column_stack((8 + uv[:,0]*(WORLD_X-16), 8 + uv[:,1]*(WORLD_Y-16)))
        rng = np.random.default_rng(RANDOM_SEED)
        ids = np.sort(rng.choice(len(xy), min(260, len(xy)), replace=False))
        chosen = xy[ids]
        k = min(5, len(chosen))
        if k >= 2:
            neighbors = NearestNeighbors(n_neighbors=k).fit(chosen).kneighbors(chosen, return_distance=False)
            edges = LineSegs("graph-links"); edges.setThickness(1.0)
            edges.setColor(.12, .26, .40, .42)
            for i, row in enumerate(neighbors):
                for j in row:
                    if j > i:
                        edges.moveTo(float(chosen[i,0]), float(chosen[i,1]), 1.7)
                        edges.drawTo(float(chosen[j,0]), float(chosen[j,1]), 1.7)
            self.graph_root.attachNewNode(edges.create())
        vmin = float(np.nanmin(values)); vmax = float(np.nanmax(values))
        for i in ids:
            t = clamp((float(values[i])-vmin)/max(vmax-vmin, 1e-8), 0, 1)
            self._box(self.graph_root, f"gnn-node-{i}",
                      (float(xy[i,0]), float(xy[i,1]), 1.8), (.26,.26,.26),
                      (t, .7*(1-t), .15, 1))
        self.graph_root.hide()

    def _create_navigation_nodes(self):
        self.nav_root=self.world.attachNewNode("navigation-risk-nodes")
        self.nav_label_root=self.world.attachNewNode("node-error-labels")
        stride=max(1,math.ceil(len(self.grid_nodes)/NODE_DISPLAY_LIMIT))
        for i in range(0,len(self.grid_nodes),stride):
            x,y=self.grid_nodes[i]
            marker=self._sphere(f"nav-node-{i}",(x,y,.34),.22,self._risk_color(self.nav_risks[i]))
            marker.reparentTo(self.nav_root)
            self.nav_visuals[i]=marker
            if i%(stride*12)==0:
                txt=TextNode(f"node-label-{i}")
                txt.setText(f"{self.nav_errors[i]:.1f} m")
                txt.setAlign(TextNode.ACenter)
                txt.setTextColor(.98,.98,.98,1)
                label=self.nav_label_root.attachNewNode(txt)
                label.setPos(x,y,1.3)
                label.setScale(.55)
                label.setBillboardPointEye()
        self.nav_root.hide()  # Clean cinematic view; toggle Nodes to inspect.
        self.nav_label_root.hide()
        self.current_node_marker=self._sphere("active-risk-node",(START[0],START[1],1.2),.48,(.1,.9,.8,1))

    @staticmethod
    def _risk_color(risk):
        risk = clamp(float(risk), 0, 1)
        if risk < .5:
            return (.12 + 1.6*risk, .82, .16, 1)
        return (.95, .82 - 1.4*(risk-.5), .12, 1)

    def _toggle_nav_nodes(self):
        if self.nav_root.isHidden():
            self.nav_root.show()
        else:
            self.nav_root.hide()

    def _toggle_node_labels(self):
        self.show_node_labels = not self.show_node_labels
        if self.show_node_labels: self.nav_label_root.show()
        else: self.nav_label_root.hide()

    def _toggle_follow_camera(self):
        self.follow_camera = not self.follow_camera
        if not self.follow_camera: self._reset_camera()

    def _sample_risk(self, force=False):
        """Periodic fresh GNN inference using replayed sensor observations."""
        x, y, _ = self.uav.getPos()
        _, nearest = self.nav_nn.kneighbors([[float(x), float(y)]])
        node_id = int(nearest[0, 0])
        # Fresh XGBoost + GraphSAGE forward pass, not a cached prediction lookup.
        inference_start = time.perf_counter()
        self.last_inference = self.live_model.infer(float(x), float(y))
        inference_ms = (time.perf_counter() - inference_start) * 1000
        error = self.last_inference["gnn_error_m"]
        risk = clamp(error / self.risk_map.scale, 0.0, 1.0)
        self.last_live_risk = risk
        self.last_risk_change = self.adaptive_map.assimilate(float(x), float(y), risk)
        self.nav_risks = self.adaptive_map.values.copy()
        self.nav_errors = self.nav_risks * self.adaptive_map.scale
        self._refresh_dynamic_node_colors()
        if node_id not in self.visited_node_ids:
            self.visited_node_ids.add(node_id)
            if node_id in self.nav_visuals:
                self.nav_visuals[node_id].setScale(.50, .50, .60)
        self.current_node_id = node_id
        nx, ny = self.grid_nodes[node_id]
        self.current_node_marker.setPos(nx, ny, 2.8)
        self.current_node_marker.setColor(*self._risk_color(risk))
        level = "LOW" if risk < .33 else ("MODERATE" if risk < .67 else "HIGH")
        self.live_readout.setText(
            f"LIVE GRAPH-SAGE INFERENCE (replayed sensors)\n"
            f"Navigation node: {node_id} / {len(self.grid_nodes)}\n"
            f"Position: E {x:.1f}m | N {y:.1f}m\n"
            f"GNN-estimated error: {error:.2f} m\n"
            f"Normalized risk: {risk:.2f} / 1.00 ({level})\n"
            f"Urban exposure proxy: {building_exposure(x,y):.2f} / 1.00\n"
            f"Visited nodes: {len(self.visited_node_ids)}\n"
            f"Flight time: {self.flight_elapsed:.1f}s\n"
            f"Inference latency: {inference_ms:.1f} ms\n"
            f"Mode: replayed sensors + live GNN\n"
            f"Auto-replan: {'ON' if self.auto_replan else 'OFF'} | replans: {self.replan_count}")
        if hasattr(self, "gui_live_rows"):
            fields=[f"NODE {node_id:03d}  |  {level}",
                    f"GNN predicted error   {error:.2f} m",
                    f"Normalized risk      {risk:.3f}",
                    f"Building exposure    {building_exposure(x,y):.2f}",
                    f"Position E/N         {x:.1f} / {y:.1f} m",
                    f"Visited nodes        {len(self.visited_node_ids)}",
                    f"Flight time          {self.flight_elapsed:.1f} s",
                    f"Model latency        {inference_ms:.1f} ms",
                    f"Auto A* {'ON' if self.auto_replan else 'OFF'}  |  Replans {self.replan_count}"]
            for widget,value in zip(self.gui_live_rows,fields):
                widget.setText(value)
            self._refresh_gui()
        if self.is_moving or force:
            self.flight_log.append({"elapsed_s": round(self.flight_elapsed, 3),
                                    "route": self.active_route, "uav_x_m": float(x),
                                    "uav_y_m": float(y), "navigation_node_id": node_id,
                                    "node_x_m": nx, "node_y_m": ny,
                                    "predicted_error_m": error, "normalized_risk": risk,
                                    "risk_level": level, "building_exposure_proxy": building_exposure(x,y),
                                    "source": "live_xgboost_graphsage_inference_with_replayed_sensor_row",
                                    "xgb_error_m": self.last_inference["xgb_error_m"],
                                    "sensor_replay_row": self.last_inference["source_row"],
                                    "gnn_inference_ms": round(inference_ms, 3),
                                    "adaptive_map_change": self.last_risk_change,
                                    "replans_so_far": self.replan_count})
        if self.is_moving and self.auto_replan:
            self._attempt_replan(force=False)

    def _refresh_dynamic_node_colors(self):
        # Recolor the existing node objects without rebuilding the entire scene.
        for i, obj in self.nav_visuals.items():
            obj.setColor(*self._risk_color(float(self.nav_risks[i])))

    def _toggle_auto_replan(self):
        self.auto_replan = not self.auto_replan
        if hasattr(self, "gui_auto"):
            self.gui_auto["text"] = f"AUTO A*: {'ON' if self.auto_replan else 'OFF'}"
        self.status.setText(f"Automatic risk-aware replanning {'enabled' if self.auto_replan else 'disabled'}")

    def _manual_replan(self):
        self._attempt_replan(force=True)

    def _remaining_path(self):
        # Position is not necessarily on a grid node. Start from the current
        # physical UAV position, then follow the not-yet-flown waypoints.
        x, y, _ = self.uav.getPos()
        remaining = self.travelled
        idx = 0
        for i, length in enumerate(self.seg_lengths):
            if remaining <= length:
                idx = i + 1
                break
            remaining -= length
        else:
            idx = len(self.path)
        return [(float(x), float(y))] + self.path[idx:]

    def _attempt_replan(self, force=False):
        if not force and self.flight_elapsed - self.last_replan_s < REPLAN_COOLDOWN_S:
            return False
        if not force and (self.last_live_risk is None or
                          self.last_live_risk < REPLAN_RISK_THRESHOLD or
                          self.last_risk_change < REPLAN_MIN_CHANGE):
            return False
        x, y, _ = self.uav.getPos()
        if distance((x, y), GOAL) < GRID_STEP * 1.5:
            return False
        weight = RISK_WEIGHTS[self.active_route]
        old_path = self._remaining_path()
        old_cost = modeled_path_cost(old_path, weight, self.adaptive_map)
        # The first waypoint is the true UAV position; A* connects through the
        # closest free grid node. Do not teleport the aircraft to the grid.
        # Preserve the selected scenario during replanning by retaining
        # checkpoints that the UAV has not reached yet.
        remaining_anchors = [p for p in SCENARIO_CHECKPOINTS[self.active_route]
                             if distance((x,y), p) + 1.0 < distance(START, p)]
        # Checkpoint progress is estimated by nearest location along the current
        # path; a completed checkpoint is not reintroduced.
        remaining_route = self._remaining_path()
        remaining_anchors = [p for p in SCENARIO_CHECKPOINTS[self.active_route]
                             if min(distance(p,q) for q in remaining_route) < GRID_STEP * 2]
        targets = remaining_anchors + [GOAL]
        candidate = []
        here = (float(x), float(y))
        for target in targets:
            leg = astar_path(here, target, weight, self.grid_nodes, self.adaptive_map)
            if not leg:
                candidate = []
                break
            candidate.extend(leg if not candidate else leg[1:])
            here = target
        if not candidate or len(candidate) < 2:
            return False
        new_cost = modeled_path_cost(candidate, weight, self.adaptive_map)
        improvement = (old_cost - new_cost) / max(old_cost, 1e-6)
        if not force and improvement < REPLAN_MIN_IMPROVEMENT:
            return False
        # The existing A* routine can connect start to a nearby grid node;
        # reject any direct connector that crosses a building.
        if any(not segment_clear(a,b) for a,b in zip(candidate[:-1],candidate[1:])):
            return False
        self.path = candidate
        self.seg_lengths = [distance(a, b) for a, b in zip(candidate[:-1], candidate[1:])]
        self.total_distance = sum(self.seg_lengths)
        self.travelled = 0.0
        self.last_replan_s = self.flight_elapsed
        self.replan_count += 1
        self.replan_events.append({
            "flight_time_s": self.flight_elapsed, "route": self.active_route,
            "uav_x_m": float(x), "uav_y_m": float(y),
            "live_gnn_risk": self.last_live_risk,
            "old_modeled_remaining_cost": old_cost,
            "new_modeled_remaining_cost": new_cost,
            "fractional_cost_improvement": improvement,
            "forced": bool(force), "replan_number": self.replan_count,
            "sensor_mode": "recorded_sensor_replay_synthetic_spatial_transfer"})
        self._redraw_routes()
        self._create_risk_overlay()
        self.status.setText(f"ROUTE UPDATED #{self.replan_count}: modeled cost improvement {improvement:.1%}")
        print(f"[A* REPLAN {self.replan_count}] t={self.flight_elapsed:.1f}s "
              f"risk={self.last_live_risk:.2f} modeled improvement={improvement:.1%}")
        return True

    def _export_flight_log(self):
        if not self.flight_log:
            return
        df = pd.DataFrame(self.flight_log)
        df.to_csv(FLIGHT_LOG_OUT, index=False)
        pd.DataFrame(self.replan_events, columns=[
            "flight_time_s", "route", "uav_x_m", "uav_y_m", "live_gnn_risk",
            "old_modeled_remaining_cost", "new_modeled_remaining_cost",
            "fractional_cost_improvement", "forced", "replan_number", "sensor_mode"
        ]).to_csv(REPLAN_EVENTS_OUT, index=False)
        pd.DataFrame({"node_id": np.arange(len(self.grid_nodes)),
                      "x_m": self.nav_xy[:, 0], "y_m": self.nav_xy[:, 1],
                      "baseline_risk": self.adaptive_map.base_values,
                      "updated_risk": self.adaptive_map.values,
                      "updated_predicted_error_m": self.adaptive_map.values * self.adaptive_map.scale
                      }).to_csv(DYNAMIC_NODE_OUT, index=False)
        # Explicitly plot estimated risk, not observed/actual GNSS error.
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(df.elapsed_s, df.predicted_error_m, color="#e66c31")
        axes[0].set_ylabel("Estimated GNSS error (m)")
        axes[1].plot(df.elapsed_s, df.normalized_risk, color="#2875b8")
        axes[1].set_ylabel("Normalized risk")
        axes[1].set_xlabel("Simulated flight time (s)")
        axes[1].set_ylim(0, 1.05)
        for event in self.replan_events:
            for ax in axes:
                ax.axvline(event["flight_time_s"], color="#b02a72", alpha=.7, linestyle="--")
        for ax in axes: ax.grid(alpha=.25)
        fig.suptitle("Live GNN inference on replayed sensor data (illustrative simulation)")
        fig.tight_layout()
        fig.savefig("flight_risk_timeseries.png", dpi=PLOT_DPI)
        plt.close(fig)
        self.status.setText(f"Saved {len(df)} samples to {FLIGHT_LOG_OUT}")
        if hasattr(self, "gui_inspect"):
            self.gui_inspect["text"] = f"Exported {len(df)} flight samples"
        print(f"Saved {FLIGHT_LOG_OUT} and flight_risk_timeseries.png")

    def _toggle_graph(self):
        if self.graph_root.isHidden(): self.graph_root.show()
        else: self.graph_root.hide()

    def _update_camera(self):
        yaw = math.radians(self.cam_yaw)
        pitch = math.radians(self.cam_pitch)
        r = self.cam_distance
        self.camera.setPos(self.camera_target + Point3(
            r*math.cos(pitch)*math.sin(yaw),
            -r*math.cos(pitch)*math.cos(yaw),
            r*math.sin(pitch)))
        self.camera.lookAt(self.camera_target)

    def _orbit(self, dyaw, dpitch):
        self.cam_yaw += dyaw
        self.cam_pitch = clamp(self.cam_pitch+dpitch, 8, 85)
        self._update_camera()

    def _zoom(self, multiplier):
        self.cam_distance = clamp(self.cam_distance*multiplier, 22, 380)
        self._update_camera()

    def _reset_camera(self):
        self.cam_yaw, self.cam_pitch, self.cam_distance = 42, 38, 126
        self.camera_target = Point3(WORLD_X/2, WORLD_Y/2, 0)
        self._update_camera()

    def _mouse_down(self):
        self.mouse_drag = True
        self.mouse_last = None

    def _mouse_up(self):
        self.mouse_drag = False
        self.mouse_last = None

    def _camera_task(self, task):
        if self.mouse_drag and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            if self.mouse_last is not None:
                self._orbit((m.x-self.mouse_last[0])*110, (m.y-self.mouse_last[1])*85)
            self.mouse_last = (m.x, m.y)
        return Task.cont

    def _create_open_park(self):
        """Restrained green buffers; avoid placing large raised slabs on the roads."""
        grass=(.30,.46,.35,1)
        for x,y,w,d in [(2.3,40,3.8,75),(50,79,95,1.8)]:
            self._box(self.world,"park",(x,y,.006),(w/2,d/2,.012),grass)
        for x,y in [(2.5,17),(2.5,40),(2.5,63),(18,79),(39,79),(65,79),(88,79)]:
            self._box(self.world,"trunk",(x,y,.9),(.16,.16,.9),(.34,.29,.22,1))
            self._sphere("tree",(x,y,2.5),1.05,(.18,.39,.25,1))

    def _create_buildings(self):
        """Architectural massing: aligned walls, window panels, roof and entrance."""
        palettes=[(.66,.71,.76,1),(.72,.73,.72,1),(.60,.68,.73,1),(.76,.75,.70,1)]
        glass=(.19,.34,.43,1)
        for i,(x,y,w,d,h) in enumerate(BUILDINGS):
            wall=palettes[i%len(palettes)]
            self._box(self.world,f"building-{i}",(x,y,h/2),(w/2,d/2,h/2),wall)
            self._box(self.world,f"roof-{i}",(x,y,h+.09),(w/2+.08,d/2+.08,.09),(.40,.46,.50,1))
            # Individual recessed-looking windows on BOTH visible street facades.
            floors=np.arange(3.3,h-1.0,3.4)
            xs=np.arange(x-w/2+1.55,x+w/2-0.4,2.7)
            ys=np.arange(y-d/2+1.55,y+d/2-0.4,2.7)
            for z in floors:
                for wx in xs:
                    for side in (-1,1):
                        self._box(self.world,"window",(float(wx),y+side*(d/2+.014),float(z)),
                                  (.76,.022,.87),glass)
                for wy in ys:
                    for side in (-1,1):
                        self._box(self.world,"window",(x+side*(w/2+.014),float(wy),float(z)),
                                  (.022,.76,.87),glass)
            self._box(self.world,"entrance",(x,y-d/2-.025,1.1),(1.1,.035,1.1),(.13,.21,.26,1))

    def _create_risk_overlay(self):
        """Optional translucent horizontal map, slightly above ground."""
        self.risk_root.node().removeAllChildren()
        source=self.adaptive_map if hasattr(self,"adaptive_map") else self.risk_map
        tile=4.0
        for x in np.arange(tile/2,WORLD_X,tile):
            for y in np.arange(tile/2,WORLD_Y,tile):
                if is_blocked(float(x),float(y),margin=0):
                    continue
                risk=source.risk_at(float(x),float(y))
                color=(.15+.75*risk,.72-.48*risk,.32-.18*risk,.22)
                cm=CardMaker("risk-tile")
                cm.setFrame(-tile/2,tile/2,-tile/2,tile/2)
                node=self.risk_root.attachNewNode(cm.generate())
                # CardMaker creates XY cards already: rotating made them vertical.
                node.setPos(float(x),float(y),.105)
                node.setColor(*color)
                node.setTransparency(TransparencyAttrib.MAlpha)
                node.setTwoSided(True)

    def _create_markers(self):
        self._sphere("start-marker", (START[0], START[1], 1), 1.5, (0.05, 0.8, 0.18, 1))
        self._sphere("goal-marker", (GOAL[0], GOAL[1], 1), 1.5, (0.95, 0.08, 0.08, 1))

    def _create_uav(self):
        self.uav = self.world.attachNewNode("simulated-uav")
        self._box(self.uav, "drone-body", (0, 0, 0), (0.85, 0.38, 0.20), (0.08, 0.10, 0.14, 1))
        for sx in (-1, 1):
            for sy in (-1, 1):
                self._box(self.uav, "drone-arm", (sx*0.70, sy*0.48, 0.02),
                          (0.65, 0.065, 0.055), (0.12, 0.14, 0.18, 1))
                rotor = self.loader.loadModel("models/misc/sphere")
                rotor.reparentTo(self.uav)
                rotor.setPos(sx*1.25, sy*0.82, 0.05)
                rotor.setScale(0.22, 0.22, 0.07)
                rotor.setColor(0.04, 0.04, 0.04, 1)
        self.uav.setScale(2.0)
        self.uav.setPos(START[0], START[1], FLIGHT_Z)

    def _create_hud(self):
        self.title=OnscreenText(text="URBAN UAV  /  GNSS RISK MISSION CONTROL",
            pos=(-1.28,.92),scale=.044,fg=(.08,.16,.22,1),align=TextNode.ALeft)
        self.help=OnscreenText(text="DRAG orbit  ·  SCROLL zoom  ·  RIGHT CLICK inspect  ·  F follow  ·  C reset",
            pos=(-1.28,-.95),scale=.025,fg=(.16,.24,.30,1),align=TextNode.ALeft)
        self.stats=OnscreenText(text="",pos=(2.5,.8),scale=.03,mayChange=True)
        self.live_readout=OnscreenText(text="",pos=(2.5,.8),scale=.03,mayChange=True)
        self.stats.hide(); self.live_readout.hide()
        self.disclaimer=OnscreenText(text="SIMULATED CITY  |  RECORDED SENSOR REPLAY  |  LIVE MODEL INFERENCE",
            pos=(-1.28,-.87),scale=.021,fg=(.43,.32,.24,1),align=TextNode.ALeft)
        self.status=OnscreenText(text="Ready",pos=(-1.28,-.81),scale=.031,
            fg=(.08,.24,.33,1),align=TextNode.ALeft,mayChange=True)

    def _create_gui(self):
        """Spaced dashboard with fixed text rows, no multiline widget overflow."""
        self.gui_panel=DirectFrame(parent=self.a2dTopRight,pos=(-.025,0,-.045),
            frameSize=(-.81,0,-1.82,0),frameColor=(.055,.083,.13,.98),relief=DGG.FLAT)
        def label(text,x,y,size=.027,color=(.86,.92,.97,1)):
            return OnscreenText(parent=self.gui_panel,text=text,pos=(x,y),scale=size,
                align=TextNode.ALeft,fg=color,mayChange=True)
        def btn(text,x,y,width,command,accent=False):
            return DirectButton(parent=self.gui_panel,text=text,pos=(x,0,y),scale=1,
                frameSize=(-width/2,width/2,-.036,.036),text_scale=.027,
                text_pos=(0,-.009),frameColor=(.08,.46,.59,1) if accent else (.14,.23,.34,1),
                text_fg=(.97,.98,1,1),relief=DGG.FLAT,command=command,
                rolloverSound=None,clickSound=None)
        label("MISSION CONTROL",-.75,-.085,.046,(.40,.83,.96,1))
        label("FLIGHT ENVIRONMENT",-.75,-.175,.025,(.53,.70,.82,1))
        self.gui_route=label("",-.75,-.226,.028)
        btn("DENSE",-.645,-.313,.205,lambda:self._select_route("dense urban"))
        btn("MIXED",-.405,-.313,.205,lambda:self._select_route("suburban"))
        btn("OPEN",-.165,-.313,.205,lambda:self._select_route("open terrain"))
        self.gui_fly=btn("START / PAUSE",-.56,-.415,.36,self._toggle_flight,True)
        btn("RESET",-.19,-.415,.25,self._reset_flight)
        label("SPEED",-.75,-.507,.025,(.53,.70,.82,1))
        self.gui_speed_text=label("8.0 m/s",-.24,-.507,.027)
        self.gui_speed_slider=DirectSlider(parent=self.gui_panel,pos=(-.405,0,-.577),
            range=(1.,20.),value=self.speed,pageSize=.5,scale=.28,
            frameColor=(.18,.29,.40,1),thumb_frameColor=(.31,.76,.94,1),command=self._set_speed)
        label("LIVE GNSS RISK / GRAPH-SAGE",-.75,-.666,.029,(.40,.83,.96,1))
        # One widget per line prevents Panda3D's font line-spacing collisions.
        self.gui_live_rows=[]
        for i in range(9):
            self.gui_live_rows.append(label("--",-.75,-.724-i*.047,.026))
        self.gui_live=self.gui_live_rows[0] # compatibility for existing callers
        self.gui_auto=btn("AUTO A*: ON",-.565,-1.235,.36,self._toggle_auto_replan,True)
        btn("REPLAN NOW",-.185,-1.235,.28,self._manual_replan)
        label("VIEW LAYERS",-.75,-1.320,.027,(.40,.83,.96,1))
        btn("Nodes",-.645,-1.395,.205,self._toggle_nav_nodes)
        btn("Labels",-.405,-1.395,.205,self._toggle_node_labels)
        btn("Graph",-.165,-1.395,.205,self._toggle_graph)
        btn("Risk",-.645,-1.477,.205,self._toggle_risk)
        btn("Route",-.405,-1.477,.205,self._toggle_paths)
        btn("Follow",-.165,-1.477,.205,self._toggle_follow_camera)
        btn("EXPORT CSV + PLOTS",-.405,-1.576,.68,self._export_flight_log,True)
        self.gui_inspect=label("Right-click a navigation node",-.75,-1.669,.023,(.65,.77,.86,1))
        label("MODEL ESTIMATE  /  NOT GROUND TRUTH",-.75,-1.744,.020,(.90,.70,.50,1))

    def _set_speed(self):
        # DirectSlider can invoke its callback during construction.
        if hasattr(self, "gui_speed_slider"):
            self.speed = float(self.gui_speed_slider["value"])
            if hasattr(self, "gui_speed_text"):
                self.gui_speed_text["text"] = f"{self.speed:.1f} m/s"

    def _refresh_gui(self):
        if hasattr(self, "gui_route"):
            self.gui_route.setText(self.active_route.upper())
            self.gui_fly["text"] = "PAUSE" if self.is_moving else "START FLIGHT"

    def _inspect_node_under_cursor(self):
        """Find nearest projected navigation node to the mouse; never changes risk data."""
        if not self.mouseWatcherNode.hasMouse():
            return
        mouse = self.mouseWatcherNode.getMouse()
        best = (0.075, None)
        from panda3d.core import Point2
        for i, (x,y) in enumerate(self.grid_nodes):
            projected = Point2()
            camera_space = self.cam.getRelativePoint(self.render, Point3(x,y,2.2))
            if not self.camLens.project(camera_space, projected):
                continue
            dist = math.hypot(float(projected.x-mouse.x), float(projected.y-mouse.y))
            if dist < best[0]:
                best = (dist, i)
        if best[1] is None:
            self.gui_inspect["text"] = "No nearby node: right-click closer to a dot"
            return
        i = best[1]
        self.gui_inspect["text"] = (
            f"Node {i}: error {self.nav_errors[i]:.2f}m | risk {self.nav_risks[i]:.2f}")
        self.current_node_marker.setPos(*self.grid_nodes[i], 2.8)
        self.current_node_marker.setColor(*self._risk_color(float(self.nav_risks[i])))

    def _select_route(self, name):
        if name not in self.routes:
            if hasattr(self, "status"):
                self.status.setText(f"Route '{name}' unavailable.")
            return
        self.active_route = name
        self.is_moving = False
        if getattr(self, "flight_log", None):
            self._export_flight_log()
        self.flight_log = []
        self.replan_events = []
        self.replan_count = 0
        self.last_replan_s = -REPLAN_COOLDOWN_S
        self.adaptive_map.values = self.adaptive_map.base_values.copy()
        self.nav_risks = self.adaptive_map.values.copy()
        self.nav_errors = self.nav_risks * self.adaptive_map.scale
        self._refresh_dynamic_node_colors()
        self._create_risk_overlay()
        self.risk_root.hide()
        self.last_live_risk = None
        self.visited_node_ids = set()
        self.flight_elapsed = 0.0
        self.sample_accum = 0.0
        self.travelled = 0.0
        self.path = self.routes[name]
        self.seg_lengths = [distance(a, b) for a, b in zip(self.path[:-1], self.path[1:])]
        self.total_distance = sum(self.seg_lengths)
        self.uav.setPos(self.path[0][0], self.path[0][1], FLIGHT_Z)
        self._redraw_routes()
        m = route_metrics(self.path, self.risk_map)
        m.update(route_exposure_metrics(self.path))
        self.stats.setText(
            f"ROUTE: {name.upper()}\n\nDistance: {m['distance']:.1f} m\n"
            f"Mean predicted error: {m['mean_error']:.2f} m\n"
            f"Peak predicted error: {m['max_error']:.2f} m\n"
            f"Waypoints: {len(self.path)}\nRisk weight: {RISK_WEIGHTS[name]:.1f}\n"
            "Risk: GNN + nearest-neighbour map transfer"
        )
        self.status.setText(f"Selected {name.upper()} route - press SPACE to fly")
        if hasattr(self, "gui_route"):
            self._refresh_gui()

    def _redraw_routes(self):
        self.paths_root.node().removeAllChildren()
        if not self.show_paths:
            self.paths_root.hide()
            return
        self.paths_root.show()
        # Only the selected route is shown: no magenta overdraw or route clutter.
        draw_line(self.paths_root,[(x,y,FLIGHT_Z+.12) for x,y in self.path],
                  ROUTE_COLORS[self.active_route],2.6,"selected-route")

    def _toggle_flight(self):
        if self.travelled >= self.total_distance:
            self._reset_flight()
        self.is_moving = not self.is_moving
        self.status.setText("UAV IN FLIGHT" if self.is_moving else "PAUSED")
        if hasattr(self, "gui_fly"):
            self._refresh_gui()

    def _reset_flight(self):
        self.is_moving = False
        if self.flight_log:
            self._export_flight_log()
        self.flight_log = []
        self.replan_events = []
        self.replan_count = 0
        self.last_replan_s = -REPLAN_COOLDOWN_S
        self.adaptive_map.values = self.adaptive_map.base_values.copy()
        self.nav_risks = self.adaptive_map.values.copy()
        self.nav_errors = self.nav_risks * self.adaptive_map.scale
        self._refresh_dynamic_node_colors()
        self._create_risk_overlay()
        self.risk_root.hide()
        self.last_live_risk = None
        self.visited_node_ids = set()
        self.flight_elapsed = 0.0
        self.sample_accum = 0.0
        self.travelled = 0.0
        self.uav.setPos(self.path[0][0], self.path[0][1], FLIGHT_Z)
        self.status.setText("RESET - press SPACE to fly")
        if hasattr(self, "gui_fly"):
            self._refresh_gui()

    def _toggle_risk(self):
        self.show_risk = not self.show_risk
        self.risk_root.show() if self.show_risk else self.risk_root.hide()

    def _toggle_paths(self):
        self.show_paths = not self.show_paths
        self._redraw_routes()

    def _animate(self, task):
        if self.is_moving:
            dt = min(globalClock.getDt(), .1)
            self.flight_elapsed += dt
            self.sample_accum += dt
            self.travelled += self.speed * dt
            if self.travelled >= self.total_distance:
                self.travelled = self.total_distance
                self.is_moving = False
                self.status.setText("DESTINATION REACHED - exporting flight telemetry")
            remaining = self.travelled
            for i, seg_len in enumerate(self.seg_lengths):
                if remaining <= seg_len or i == len(self.seg_lengths)-1:
                    a, b = self.path[i], self.path[i+1]
                    frac = 0 if seg_len <= 1e-9 else clamp(remaining/seg_len, 0, 1)
                    x = a[0] + frac*(b[0]-a[0])
                    y = a[1] + frac*(b[1]-a[1])
                    self.uav.setPos(x, y, FLIGHT_Z)
                    self.uav.setH(math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))
                    break
                remaining -= seg_len
            if self.sample_accum >= 1.0 / RISK_UPDATE_HZ or not self.is_moving:
                self._sample_risk()
                self.sample_accum = 0.0
            if not self.is_moving:
                self._export_flight_log()
        if self.follow_camera:
            x, y, _ = self.uav.getPos()
            self.camera_target = Point3(x, y, 1)
            self.cam_distance = 33
            self.cam_pitch = 23
            self._update_camera()
        return Task.cont


def main():
    set_seed(RANDOM_SEED)
    df, _ = load_and_predict_csv()
    predicted_df, _ = train_gnn(df)
    risk_map = RiskMap(predicted_df)
    print("\nStarting Panda3D simulator.")
    print("Important: make sure the trajectory coordinates and simulated map use the same frame/scale.")
    live_model = LiveGNNInference(predicted_df)
    app = UAVRiskSimulator(risk_map, predicted_df, live_model)
    app.run()


if __name__ == "__main__":
    main()

