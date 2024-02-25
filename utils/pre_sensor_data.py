import re
from datetime import datetime
from collections import Counter
import numpy as np

def regex_get_acc(set_num = 0, choosing="shaking", seconds=3):
    pattern = re.compile(r'Accelerometer\[\d\] :([-\d.]+)')
    # choosing = 0: shaking
    # choosing = 1: nodding
    # choosing = 2: s_n

    if set_num == 0 and choosing == "s_n":
        return "Error: no s_n dataset in dataset0"

    accelerometer_data = []
    path = f"C:\\Users\\xueyu\\Desktop\\evasion\\dataset{str(set_num)}\\{choosing}\\accel.txt"
    with open(path, "r") as file:
        for line in file:
            matches = pattern.findall(line)
            if matches:
                accelerometer_data.append([float(x) for x in matches])

    return accelerometer_data, int(count_entries_per_second(path))

# print(regex_get_acc()[0])

def regex_get_gyro(set_num = 0, choosing="shaking", seconds=3):
    pattern = re.compile(r'Gyroscope\[\d\] :([-\d.]+)')
    # choosing = 0: shaking
    # choosing = 1: nodding
    # choosing = 2: s_n

    if set_num == 0 and choosing == "s_n":
        return "Error: no s_n dataset in dataset0"

    gyroscope_data = []
    path = f"C:\\Users\\xueyu\\Desktop\\evasion\\dataset{str(set_num)}\\{choosing}\\gyro.txt"
    with open(path, "r") as file:
        for line in file:
            matches = pattern.findall(line)
            if matches:
                gyroscope_data.append([float(x) for x in matches])

    return gyroscope_data, int(count_entries_per_second(path))

def regex_get_mag(set_num = 0, choosing="shaking", seconds=3):
    pattern = re.compile(r'Magnetometer\[\d\] :([-\d.]+)')
    # choosing = 0: shaking
    # choosing = 1: nodding
    # choosing = 2: s_n

    if set_num == 0 and choosing == "s_n":
        return "Error: no s_n dataset in dataset0"

    magnetometer_data = []
    path = f"C:\\Users\\xueyu\\Desktop\\evasion\\dataset{str(set_num)}\\{choosing}\\mag.txt"
    with open(path, "r") as file:
        for line in file:
            matches = pattern.findall(line)
            if matches:
                magnetometer_data.append([float(x) for x in matches])

    return magnetometer_data, int(count_entries_per_second(path))

def get(set_num=0, seconds=3, choosing="nodding", sensor="accel"):
    path = f"C:\\Users\\xueyu\\Desktop\\evasion\\dataset{str(set_num)}\\{choosing}\\{sensor}.txt"
    frequency_x = int(count_entries_per_second(path))
    pattern = re.compile('')
    if sensor == "accel":
        pattern = re.compile(r'Accelerometer\[\d\] :([-\d.]+)')
    elif sensor == "gyro":
        pattern = re.compile(r'Gyroscope\[\d\] :([-\d.]+)')
    elif sensor == "mag":
        pattern = re.compile(r'Magnetometer\[\d\] :([-\d.]+)')
    data = []
    with open(path, "r") as file:
        for line in file:
            matches = pattern.findall(line)
            if matches:
                data.append([float(x) for x in matches])

    # trim the result to fit the model
    data = data[:-(len(data) % (seconds * frequency_x))]

    # reshape the data
    data = np.array(data).reshape(-1, seconds * frequency_x, 3)

    # normalize the data
    min_value, max_value = np.min(data), np.max(data)
    data = (data - min_value) / (max_value - min_value)

    return data, frequency_x


# Function to convert log timestamp to a simplified datetime object (ignoring milliseconds)
def parse_timestamp_to_second(hour, minute, second):
    # Use a fixed date for all entries since only time is relevant here
    return datetime(2000, 1, 1, int(hour), int(minute), int(second))

# Function to count log entries per second
def count_entries_per_second(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    timestamps = []
    # Regex to match the timestamp format
    # regex = r"Time: (\d{2}) (\d{2}) (\d{2}) \d{3}"
    regex = r"Log Entry :  Time: (\d{2}) (\d{2}) (\d{2}) \d{2,3}"
    
    for line in content:
        match = re.search(regex, line)
        if match:
            # Ignore milliseconds for this part
            timestamps.append(parse_timestamp_to_second(*match.groups()[:3]))

    # Count occurrences of each second
    counts = Counter(timestamps)
    
    # Calculate the total number of log entries and the number of unique seconds
    total_entries = sum(counts.values())
    unique_seconds = len(counts)
    
    # Calculate the average number of entries per second
    if unique_seconds > 0:
        average_entries_per_second = total_entries / unique_seconds
        print(f"Average count per second: {average_entries_per_second}")
    else:
        print("No log entries found.")
        average_entries_per_second = 0

    return average_entries_per_second
    
# count_entries_per_second("C:\\Users\\xueyu\\Desktop\\evasion\\dataset0\\shaking\\accel.txt")
if __name__ == "__main__":
    result = get(set_num=0, seconds=3, choosing="nodding", sensor="accel")
    print(result.shape)
