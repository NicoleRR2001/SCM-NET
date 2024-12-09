import math
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
import pandas as pd

'''
file name: DE_3D_Feature
input: the path of saw EEG file in SEED-VIG dataset

output: the 3D feature of all subjects
'''

# step1: input raw data
# step2: decompose frequency bands
# step3: calculate DE
# step4: stack them into 3D featrue
"""
分成5个频带，每100分为1个DE进行交叉熵计算，将得到的结果进行垂直堆叠
EEG信号的分布大致符合Gauss分布，交叉熵等于能量谱的log
"""

def butter_bandpass(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# calculate DE
def calculate_DE(saw_EEG_signal):
    variance = np.var(saw_EEG_signal, ddof=1) #求得方差
    return math.log(2 * math.pi * math.e * variance) / 2 #微分熵求取公式，简化版


# filter for 5 frequency bands
def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decompose_to_DE(file):
    # read data  sample * channel [1416000, 17]
    data = loadmat(file)['EEG']['data'][0, 0]
    print(data.shape)
    # sampling rate 采样率
    frequency = 200
    # samples 1416000
    samples = data.shape[0]
    print(samples)
    # 100 samples = 1 DE
    num_sample = int(samples/100)
    channels = data.shape[1]
    print(channels)
    bands = 5
    # init DE [14160, 17, 5]
    DE_3D_feature = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    for channel in range(channels):
        trial_signal = data[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4,   frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8,   frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14,  frequency, order=3)
        beta  = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)
        df = pd.DataFrame({
            'delta Data': delta,
            'theta Data': theta,
            'alpha Data': alpha,
            'beta Data': beta,
            'gamma Data': gamma
        })
        print(df.shape)
        max_rows_per_sheet = 800000
        num_sheets = len(df) // max_rows_per_sheet + (1 if len(df) % max_rows_per_sheet else 0)
        print(num_sheets)

        # 将DataFrame分割并保存到多个工作表
        for i in range(num_sheets):
            start_row = i * max_rows_per_sheet
            end_row = start_row + max_rows_per_sheet
            df[start_row:end_row].to_excel(f'filtered_data_sheet_{i + 1}.xlsx', index=False)
        # df.to_excel('filtered_data.xlsx', index=False)

        # DE
        DE_delta = np.zeros(shape=[0], dtype=float)
        DE_theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)
        # DE of delta, theta, alpha, beta and gamma，每100个是1个DE
        for index in range(num_sample):
            DE_delta = np.append(DE_delta, calculate_DE(delta[index * 100:(index + 1) * 100]))
            DE_theta = np.append(DE_theta, calculate_DE(theta[index * 100:(index + 1) * 100]))
            DE_alpha = np.append(DE_alpha, calculate_DE(alpha[index * 100:(index + 1) * 100]))
            DE_beta  = np.append(DE_beta,  calculate_DE(beta[index * 100:(index + 1) * 100]))
            DE_gamma = np.append(DE_gamma, calculate_DE(gamma[index * 100:(index + 1) * 100]))
            df2 = pd.DataFrame({
                'delta Data': DE_delta,
                'theta Data': DE_theta,
                'alpha Data': DE_alpha,
                'beta Data': DE_beta,
                'gamma Data': DE_gamma
            })
            print(df2.shape)
            max_rows_per_sheet = 800000
            num_sheets2 = len(df2) // max_rows_per_sheet + (1 if len(df2) % max_rows_per_sheet else 0)
            print(num_sheets2)

            # 将DataFrame分割并保存到多个工作表
            for i in range(num_sheets2):
                start_row2 = i * max_rows_per_sheet
                end_row2 = start_row2 + max_rows_per_sheet
                df[start_row2:end_row2].to_excel(f'filtered_data_sheet2_{i + 1}.xlsx', index=False)
            # df.to_excel('filtered_data.xlsx', index=False)
        # 按照垂直方向堆叠构成一个新的数组
        temp_de = np.vstack([temp_de, DE_delta])
        temp_de = np.vstack([temp_de, DE_theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])


    temp_trial_de = temp_de.reshape(-1, 5, num_sample)
    temp_trial_de = temp_trial_de.transpose([2, 0, 1])
    DE_3D_feature = np.vstack([temp_trial_de])

    return DE_3D_feature


if __name__ == '__main__':
    # Fill in your SEED-VIG dataset path
    filePath = '/home/lizr/SFT-Net/SEED-VIG/Raw_Data/'
    dataName = ['1_20151124_noon_2.mat', '2_20151106_noon.mat', '3_20151024_noon.mat', '4_20151105_noon.mat',
                '4_20151107_noon.mat', '5_20141108_noon.mat', '5_20151012_night.mat', '6_20151121_noon.mat',
                '7_20151015_night.mat', '8_20151022_noon.mat', '9_20151017_night.mat', '10_20151125_noon.mat',
                '11_20151024_night.mat', '12_20150928_noon.mat', '13_20150929_noon.mat', '14_20151014_night.mat',
                '15_20151126_night.mat', '16_20151128_night.mat', '17_20150925_noon.mat', '18_20150926_noon.mat',
                '19_20151114_noon.mat', '20_20151129_night.mat', '21_20151016_noon.mat']

    X = np.empty([0, 17, 5])

    for i in range(len(dataName)):
        dataFile = filePath + dataName[i]
        print('processing {}'.format(dataName[i]))
        # every subject DE feature
        DE_3D_feature = decompose_to_DE(dataFile)
        print(DE_3D_feature.shape)
        # all subjects
        X = np.vstack([X, DE_3D_feature])
        print(X.shape)

    # save .npy file
    np.save("./processedData/data_3d.npy", X)