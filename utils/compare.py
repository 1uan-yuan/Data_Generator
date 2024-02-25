import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_data():
    df_cvae_real = pd.read_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cvae_real.csv')
    df_cvae_fake = pd.read_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cvae_fake.csv')
    
    df_cgan_cvae_real = pd.read_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cgan_cvae_real.csv')
    df_cgan_cvae_fake = pd.read_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cgan_cvae_fake.csv')

    df_cvae_real_numpy = df_cvae_real.to_numpy()
    df_cvae_fake_numpy = df_cvae_fake.to_numpy()
    df_cgan_cvae_real_numpy = df_cgan_cvae_real.to_numpy()
    df_cgan_cvae_fake_numpy = df_cgan_cvae_fake.to_numpy()

    return df_cvae_real_numpy, df_cvae_fake_numpy, df_cgan_cvae_real_numpy, df_cgan_cvae_fake_numpy

def plot_3D_4():
    cvae_real, cvae_fake, cgan_cvae_real, cgan_cvae_fake = extract_data()

    x1, y1, z1 = cvae_real[::3], cvae_real[1::3], cvae_real[2::3]
    x2, y2, z2 = cvae_fake[::3], cvae_fake[1::3], cvae_fake[2::3]

    x3, y3, z3 = cgan_cvae_real[::3], cgan_cvae_real[1::3], cgan_cvae_real[2::3]
    x4, y4, z4 = cgan_cvae_fake[::3], cgan_cvae_fake[1::3], cgan_cvae_fake[2::3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, z1, c='r', label='cvae_real')
    ax.scatter(x2, y2, z2, c='b', label='cvae_fake')
    ax.scatter(x3, y3, z3, c='g', label='cgan_cvae_real')
    ax.scatter(x4, y4, z4, c='y', label='cgan_cvae_fake')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    plt.show()

def plot_1D_4():
    cvae_real, cvae_fake, cgan_cvae_real, cgan_cvae_fake = extract_data()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    x1, y1, z1 = cvae_real[::3], cvae_real[1::3], cvae_real[2::3]
    x2, y2, z2 = cvae_fake[::3], cvae_fake[1::3], cvae_fake[2::3]
    x3, y3, z3 = cgan_cvae_real[::3], cgan_cvae_real[1::3], cgan_cvae_real[2::3]
    x4, y4, z4 = cgan_cvae_fake[::3], cgan_cvae_fake[1::3], cgan_cvae_fake[2::3]

    # Plotting x1 and x2 in the first subplot
    axs[0].plot(x1, label='cvae_real x', marker='o')
    axs[0].plot(x2, label='cvae_fake x', marker='^')
    axs[0].plot(x3, label='cgan_cvae_real x', marker='s')
    axs[0].plot(x4, label='cgan_cvae_fake x', marker='p')
    axs[0].set_title('X comparison')
    axs[0].legend()

    # Plotting y1 and y2 in the second subplot
    axs[1].plot(y1, label='cvae_real y', marker='o')
    axs[1].plot(y2, label='cvae_fake y', marker='^')
    axs[1].plot(y3, label='cgan_cvae_real y', marker='s')
    axs[1].plot(y4, label='cgan_cvae_fake y', marker='p')
    axs[1].set_title('Y comparison')
    axs[1].legend()

    # Plotting z1 and z2 in the third subplot
    axs[2].plot(z1, label='cvae_real z', marker='o')
    axs[2].plot(z2, label='cvae_fake z', marker='^')
    axs[2].plot(z3, label='cgan_cvae_real z', marker='s')
    axs[2].plot(z4, label='cgan_cvae_fake z', marker='p')
    axs[2].set_title('Z comparison')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3D_4()
    plot_1D_4()