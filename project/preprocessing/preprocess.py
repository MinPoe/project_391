# before running the file run the command 'pip install pandas scikit-learn joblib'
# run data processing by running the command 'python preprocesssing/preprocess.py'
import os
import sys
import subprocess
import tempfile
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# config
BAG_PATHS = [
    ("data/gap_following_data", 24),
    ("data/gap_following_data_v2", 24),
    ("data/gap_following_data_v3", 148),
]
OUTPUT_DIR = "processed"
LAP_DURATION_SEC = 148  # approximate lap time in seconds
LIDAR_STEP = 6          # keep every nth ray (1081 → 181 rays)
MAX_RANGE = 10.0        # cap lidar readings at this distance in meters
MIN_SPEED = 0.05        # drop rows where car is stationary
MAX_STEER = 0.5         # clip steering to ±0.5 rad (car max is ~0.42)

def bag_to_df(bag_path):
    # convert bag to a temporary csv then load it into a dataframe
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name
    try:
        subprocess.run(
            [sys.executable, "preprocessing/bag_to_csv.py", "--bag", bag_path, "--output", tmp_path],
            check=True
        )
        df = pd.read_csv(tmp_path)
    finally:
        os.remove(tmp_path)
    return df

def label_laps(df, session_id, lap_duration_sec):
    df = df.sort_values("timestamp").reset_index(drop=True)
    lap_duration_ns = lap_duration_sec * 1_000_000_000
    start_ts = df["timestamp"].iloc[0]
    df = df.copy()
    df["lap_id"] = ((df["timestamp"] - start_ts) // lap_duration_ns).astype(int)
    df["session"] = session_id
    # drop last partial lap (only if there are multiple laps)
    if df["lap_id"].max() > 0:
        df = df[df["lap_id"] < df["lap_id"].max()].reset_index(drop=True)
    return df

def clean(df):
    lidar_cols = [c for c in df.columns if c.startswith("lidar_")]
    df = df.copy()
    # cap lidar at max range and fill missing values
    df[lidar_cols] = (
        df[lidar_cols]
        .replace([np.inf, -np.inf], np.nan)
        .clip(0, MAX_RANGE)
        .fillna(MAX_RANGE)
    )
    # clip steering outliers (raw data can have ±π from bad readings)
    df["steering_angle"] = df["steering_angle"].clip(-MAX_STEER, MAX_STEER)
    # odom_vx is negative when moving forward on this vehicle
    df = df[df["odom_vx"].abs() > MIN_SPEED].reset_index(drop=True)
    return df

def downsample_lidar(df):
    lidar_cols = [c for c in df.columns if c.startswith("lidar_")]
    keep = lidar_cols[::LIDAR_STEP]
    meta = ["timestamp", "steering_angle", "speed", "odom_vx", "lap_id", "session"]
    return df[meta + keep]

def augment(df):
    # mirror lidar scans and flip steering to simulate driving the other direction
    lidar_cols = [c for c in df.columns if c.startswith("lidar_")]
    mirrored = df.copy()
    mirrored[lidar_cols] = df[lidar_cols].values[:, ::-1]
    mirrored["steering_angle"] = -df["steering_angle"]
    return pd.concat([df, mirrored], ignore_index=True)

def normalize(df):
    lidar_cols = [c for c in df.columns if c.startswith("lidar_")]
    scaler_lidar = MinMaxScaler()
    scaler_action = MinMaxScaler()
    df = df.copy()
    df[lidar_cols] = scaler_lidar.fit_transform(df[lidar_cols])
    df[["steering_angle", "speed"]] = scaler_action.fit_transform(df[["steering_angle", "speed"]])
    return df, scaler_lidar, scaler_action

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = []
    for i, (bag_path, lap_sec) in enumerate(BAG_PATHS):
        if not os.path.exists(bag_path):
            continue
        df = bag_to_df(bag_path)
        df = label_laps(df, session_id=i, lap_duration_sec=lap_sec)
        df = clean(df)
        df = downsample_lidar(df)
        frames.append(df)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    # Fill NaN from bags with different ray counts
    lidar_cols = [c for c in combined.columns if c.startswith("lidar_")]
    combined[lidar_cols] = combined[lidar_cols].fillna(MAX_RANGE)
    combined = augment(combined)
    combined, scaler_lidar, scaler_action = normalize(combined)

    combined.to_csv(f"{OUTPUT_DIR}/data.csv", index=False)
    joblib.dump(scaler_lidar, f"{OUTPUT_DIR}/scaler_lidar.pkl")
    joblib.dump(scaler_action, f"{OUTPUT_DIR}/scaler_action.pkl")

    # Save scalers as .npz for the ROS2 inference nodes
    np.savez(
        f"{OUTPUT_DIR}/scalers.npz",
        lidar_scale=scaler_lidar.scale_.astype(np.float32),
        lidar_min=scaler_lidar.min_.astype(np.float32),
        action_scale=scaler_action.scale_.astype(np.float32),
        action_min=scaler_action.min_.astype(np.float32),
    )
    print(f"Scalers saved to {OUTPUT_DIR}/scalers.npz")
    print(f"  Steering range: [{-MAX_STEER}, {MAX_STEER}] rad")
    print(f"  action_scale: {scaler_action.scale_}")
    print(f"  action_min:   {scaler_action.min_}")

if __name__ == "__main__":
    main()