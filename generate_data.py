import numpy as np
import random
import csv

# Function to generate ts
def generate_ts(length=45, start=3.46, increment_range=(0.8, 1.5)):
    ts = [start]
    for i in range(1, length):
        ts.append(ts[-1] + random.uniform(*increment_range))
    return ts

# Function to generate Vs
def generate_Vs(ts, base_value=0.0158, max_fluctuation=0.5, growth_rate=0.1):
    Vs = []
    for t in ts:
        fluctuation = random.uniform(-max_fluctuation, max_fluctuation)
        Vs.append(base_value + (growth_rate * t) + fluctuation)
        base_value = Vs[-1]
    return Vs

# Get the filename from the user
filename = input("Enter the name of the file to save the data (e.g., 'data.csv'): ")

# Generate the data
ts = generate_ts()
Vs = generate_Vs(ts)

# Write the data to the file specified by the user
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ts", "Vs"])
    for t, v in zip(ts, Vs):
        writer.writerow([t, v])

print(f"Data has been generated and saved in {filename}")
