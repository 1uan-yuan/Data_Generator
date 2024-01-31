import re

pattern = re.compile(r'Accelerometer\[\d\] :([-\d.]+)')

def regex_get_acc(set_num = 0, choosing="shaking"):
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

    return accelerometer_data

# print(regex_get_acc()[0])