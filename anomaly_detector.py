import math

class RunningStats:
    """
    Keeps track of running statistics (mean, variance) using Welford's algorithm. 
    This lets us update mean and std dv incrementally without storing all the data.
    """

    def __init__(self, num_sensors=5):
        # Number of sensors to track stats for
        self.num_sensors = num_sensors

        # Initialize counts of readings seen for each sensor (list of ints)
        self.n = [0] * num_sensors

        # Initialize running means for each sensor (list of floats)
        self.mean = [0.0] * num_sensors

        # Initialize sum of squares of differences from the mean (list of floats)
        self.M2 = [0.0] * num_sensors

    
    def update(self, readings):
        """
        Update running stats with new sensor readings.

        Args:
        readings (list of floats): latest sensor values (length == num_sensors)
        """

        for i, x in enumerate(readings):
            self.n[i] += 1
            delta = x - self.mean[i]
            self.mean[i] += delta / self.n[i]
            delta2 = x - self.mean[i]
            self.M2[i] += delta * delta2

    
    def variance(self):
        """
        Returns the variance for each sensor.
        Variance is undefined for fewer than 2 data points, so return 0 for those.
        """

        var = []
        for i in range(self.num_sensors):
            if self.n[i] < 2:
                var.append(0.0)
            else:
                var.append(self.M2[i] / (self.n[i] - 1))
        return var
    

    def std_dev(self):
        """
        Return the standard deviation (sqrt of variance) for each sensor.
        """
        return [math.sqrt(v) for v in self.variance()]
    


def detect_anomalies(readings, running_stats, threshold=3):
    """
    Detects anomalies based on how far each sensor reading is from the running mean.

    Args:
        readings (list of floats): current sensor readings
        running_stats (RunningStats): object tracking running stats
        threshold (float): number of std deviations away from mean to flag anomaly

    Returns:
        List of bools, True if sensor reading is anomalous, else False
    """

    anomalies = []
    mean = running_stats.mean
    stds = running_stats.std_dev()

    for i, x in enumerate(readings):
        # If std dev is 0 (not enough data), treat as no anomaly
        if stds[i] == 0:
            anomalies.append(False)
        else:
            # Check if reading is outside mean plus or minus threshold * std dev
            if abs(x - mean[i]) > threshold * stds[i]:
                anomalies.append(True)
            else:
                anomalies.append(False)
    return anomalies