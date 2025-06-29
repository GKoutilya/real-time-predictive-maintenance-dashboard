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

    print("Starting real-time accelerometer stream...\n")
    for _, row in df.iterrows():
        data = {
            "timestamp" : row["timestamp"],
            "accel_x" : row["accel_x"],
            "accel_y" : row["accel_y"],
            "accel_z" : row["accel_z"]
        }
        print(json.dumps(data)) # Emits one row every 100ms
        time.sleep(0.1)

    print("\n Stream Complete.")

if __name__ == "__main__":
    """
    Main Method

    This function is only executed when the file is run directly.
    """
    stream_data()