import os
import pandas as pd

dataset_dir = "rgbd_dataset_freiburg2_desk"
rgb_file = os.path.join(dataset_dir, "rgb.txt")
accel_file = os.path.join(dataset_dir, "preprocessed_accel.csv")
output_synced_csv = os.path.join(dataset_dir, "synced_rgb_accel.csv")

if os.path.exists(output_synced_csv):
    print(f"File already exists: {output_synced_csv}")
    print("Skipping RGB + accelerometer sync to avoid duplication")
else:
    # Load RGB timestamps
    rgb_timestamps = []
    rgb_filenames = []

    with open(rgb_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                timestamp, filename = parts
                rgb_timestamps.append(float(timestamp))
                rgb_filenames.append(filename)

    # Load accelerometer DataFrame
    accel_df = pd.read_csv(accel_file)

    # Match each RGB timestamp with closest accel timestamp
    synced_data = []
    for ts, rgb_file in zip(rgb_timestamps, rgb_filenames):
        closest_idx = (accel_df['timestamp'] - ts).abs().idxmin()
        closest_row = accel_df.loc[closest_idx]

        synced_data.append({
            "rgb_timestamp": ts,
            "rgb_file" : rgb_file,
            "accel_timestamp" : closest_row["timestamp"],
            "accel_x" : closest_row["accel_x"],
            "accel_y" : closest_row["accel_y"],
            "accel_z" : closest_row["accel_z"]
        })

    # Convert to DataFrame and save
    synced_df = pd.DataFrame(synced_data)
    synced_df.to_csv(output_synced_csv, index=False)

    print(f"Saved synced RGB + accelerometer data to {output_synced_csv}")