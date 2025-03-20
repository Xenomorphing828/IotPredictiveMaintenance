# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate timestamps for 30 days at 1-second intervals
timestamps = pd.date_range(start="2023-01-01", end="2023-01-30", freq="S")

# Step 2: Generate synthetic sensor data
np.random.seed(42)  # For reproducibility
num_samples = len(timestamps)

# Temperature: Random values with a baseline and some noise
temperature = 50 + 10 * np.sin(2 * np.pi * np.arange(num_samples) / (24 * 60 * 60)) + np.random.normal(0, 2, num_samples)

# Vibration: Random values with occasional spikes
vibration = np.random.normal(0, 1, num_samples)
vibration_spikes = np.random.choice([0, 5], size=num_samples, p=[0.95, 0.05])  # 5% chance of spikes
vibration += vibration_spikes

# Pressure: Random values with a gradual increase over time
pressure = 100 + 0.01 * np.arange(num_samples) + np.random.normal(0, 5, num_samples)

# Step 3: Generate labels (0 = normal, 1 = failure)
failure_condition = (temperature > 70) | (vibration > 6) | (pressure > 150)
labels = np.where(failure_condition, 1, 0)

# Step 4: Combine data into a DataFrame
data = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "failure": labels
})

# Step 5: Save the data to a CSV file
data.to_csv("synthetic_sensor_data.csv", index=False)

# Step 6: Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data["timestamp"], data["temperature"], label="Temperature")
plt.scatter(data[data["failure"] == 1]["timestamp"], data[data["failure"] == 1]["temperature"], color="red", label="Failure")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Over Time with Failures")
plt.legend()
plt.show()