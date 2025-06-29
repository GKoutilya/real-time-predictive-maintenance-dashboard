import os
import pandas as pd

# Path to the dataset folder
dataset_dir = "rgbd_dataset_freiburg2_desk"
accel_file = os.path.join(dataset_dir, "accelerometer.txt")
output_csv = os.path.join(dataset_dir, "resampled_accel.csv")

if os.path.exists(output_csv):
    print(f'File already exists: {output_csv}')
    print("Skipping preprocessing and resampling duplication.")
else:
    # Read accelerometer.txt
    with open(accel_file, "r") as f:
        lines = f.readlines()

    # Skip comments and parse data
    data = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 4:
            timestamp, ax, ay, az = parts
            data.append({
                "timestamp" : float(timestamp),
                "accel_x" : float(ax),
                "accel_y" : float(ay),
                "accel_z" : float(az)
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Normalize accel_x, accel_y, accel_z (optional, but useful)
    for axis in ["accel_x", "accel_y", "accel_z"]:
        mean = df[axis].mean()
        std = df[axis].std()
        df[axis] = (df[axis] - mean) / std
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    # Resample at 10Hz (100mx intervals) and interpolate missing values
    df_resampled = df.resample('100ms').mean().interpolate()

    # Reset index and convert timestamp back to float (seconds)
    df_resampled.reset_index(inplace=True)
    df_resampled['timestamp'] = df_resampled["timestamp"].astype('int64') / 1e9

    # Save to CSV
    df_resampled.to_csv(output_csv, index=False)
    print(f"Saved normalized and resampled accerlerometer data to {output_csv}")