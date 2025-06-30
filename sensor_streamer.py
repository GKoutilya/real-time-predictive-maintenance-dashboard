import pandas as pd
import time
import json
import os


# Path to preprocessed accerlerometer data file
input_csv = "rgbd_dataset_freiburg2_desk/resampled_accel.csv"

def stream_data():
    """
    Simulates a real-time stream of accerlerometer sensor data.

    Reads from a preprocessed CSV file containing timestamped IMU readings,and prints each data row to the console as a JSON object at 10 Hz (one row every 100ms).

    This function is useful for testing real-time ML inference or dashboard systems with realistic sensor input behavior.

    Raises:
        FileNotFoundError: If the input CSV file is not found at the expected path.
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
    
    df = pd.read_csv(input_csv)
    for _, row in df.iterrows():
        data = [row["accel_x"], row["accel_y"], row["accel_z"]]
        yield data

def load_imu_data(filepath: str) -> pd.DataFrame:
    """
    Loads IMU accelerometer data from the TUM RGB-D dataset.

    Args:
        filepath (str): Path to accelerometer.txt file

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'ax', 'ay', 'az']
    """
    df = pd.read_csv(
        filepath,
        delim_whitespace=True,
        comment='#',
        header=None,
        usecols=[0,1,2,3],
        names=['timestamp', 'ax', 'ay', 'az'],
        on_bad_lines='skip'  # skip malformed lines
    )
    return df

if __name__ == "__main__":
    """
    Main Method

    This function is only executed when the file is run directly.
    """
    stream_data()
    imu_df = load_imu_data("rgbd_dataset_freiburg2_desk/accelerometer.txt")