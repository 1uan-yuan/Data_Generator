import matplotlib.pyplot as plt
import pandas as pd

def plot_3D(real_values, fake_values):
    # print("real_values.shape: ", real_values.shape, "fake_values.shape: ", fake_values.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, y1, z1 = real_values[::3], real_values[1::3], real_values[2::3]
    x2, y2, z2 = fake_values[::3], fake_values[1::3], fake_values[2::3]

    ax.scatter(x1, y1, z1, c='r', label='real')
    ax.scatter(x2, y2, z2, c='b', label='fake')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()

def plot_1D(real_values, fake_values):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    x1, y1, z1 = real_values[::3], real_values[1::3], real_values[2::3]
    x2, y2, z2 = fake_values[::3], fake_values[1::3], fake_values[2::3]

    # Plotting x1 and x2 in the first subplot
    axs[0].plot(x1, label='Real x', marker='o')
    axs[0].plot(x2, label='Fake x', marker='^')
    axs[0].set_title('X comparison')
    axs[0].legend()

    # Plotting y1 and y2 in the second subplot
    axs[1].plot(y1, label='Real y', marker='o')
    axs[1].plot(y2, label='Fake y', marker='^')
    axs[1].set_title('Y comparison')
    axs[1].legend()

    # Plotting z1 and z2 in the third subplot
    axs[2].plot(z1, label='Real z', marker='o')
    axs[2].plot(z2, label='Fake z', marker='^')
    axs[2].set_title('Z comparison')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def plot_time_series(real_values, fake_values):
    ts1 = pd.Series(real_values) # real data
    ts2 = pd.Series(fake_values) # fake data

    plt.figure(figsize=(12, 6))
    plt.plot(ts1.index, ts1.values, label='Real data', linewidth=2, linestyle='-')
    plt.plot(ts2.index, ts2.values, label='Fake data', linewidth=2, linestyle='--')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()