import numpy as np
import time

# Simulates 5 sensors and outputs a new reading every time it's called
# Each reading will be a list of 5 floating-point numbers (decimals) drawn from a normal distribution
def generate_sensor_reading(num_sensors=5):
    '''
    Simulates a single reading from multiple sensors.
    
    Args:
    num_sensors (int): Number of sensors to simulate (default = 5)

    Returns:
    List of sensor values, each drawn from a normal distribution with mean 0 and standard deviation 1
    '''

    # loc= - mean (center of distribution), scale= - standard deviation (how spread out the values are around the mean), size= - how many values around this distribution
    readings = np.random.normal(loc=0, scale=1, size=num_sensors)
    return readings.tolist()


def stream_sensor_data(delay=0.1):
    '''
    Continuously streams sensor readings at regular time intervals.

    Conveyor belt of sensor data - data keeps on moving

    Args:
    delay (float (decimal)): Time (in seconds) to wait between readings (default = 0.1)
    '''

    # Creates infinite loop to keep the stream going, information is infinitely generated for the model to study/predict
    while True:
        # Gets a new sensor reading
        reading = generate_sensor_reading()
        # Prints for now (can log, visualize, or store it later)
        print(reading)
        # Waits before generating the next reading
        time.sleep(delay)





# stream_sensor_data()