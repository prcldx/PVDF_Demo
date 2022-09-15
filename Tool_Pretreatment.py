import pandas as pd
import os
import numpy as np
from scipy import signal
import math

def read_data(file_path):
    """
    read raw data
    :param file_path: Path to the folder where the data is located
    :return: movedata_0 (list, each element represents the relevant content in one csv file);

             samplingpoint_num_list (list, Number of sampling points of data in each csv file).
    """
    movedata_0 = []
    samplingpoint_num_list = []
    for root, dir, filenames_seque in os.walk(file_path):   #in order to get all the name strings of csv files
        filenames_seque.sort(key=lambda x:int(x.split('.')[0].split('-')[0]))#排序
        for file in filenames_seque:
            if os.path.splitext(file)[1] =='.csv':
                df = pd.read_csv(file_path + '//' + file, header=None)
                data_arraysum = np.array(df)
                movedata_0.append(data_arraysum)
                samplingpoint_num_list.append(data_arraysum.shape[0])
        break
    return movedata_0, samplingpoint_num_list


def butter_lowpass_filter(data, cutoff, fs, order=3):
    """
    数字低通巴特沃斯滤波过程：对一维数据滤波（只需调用这一个函数即可滤波）
    :param data: 一维数组
    :param cutoff:
    :param fs:
    :param order:
    :return: 一维数组
    """
    wn = 2*cutoff / fs    # 归一化截止频率=截止频率/信号频率，即MATLAB中butter 的 Wn
    b, a = signal.butter(order, wn, btype='lowpass', analog=False)   # 原本butter只能输出一个值，但用2个变量接收也无报错
    filted_data=signal.filtfilt(b,a,data)
    return filted_data

def butter_bandpass_filter(data,cutoff, fs, order=4):
    """
    数字带通滤波器，对一维数据进行滤波（只需调用这一个函数即可滤波）
    :param data:
    :param cutoff: type<list>,length = 2
    :param fs:
    :param order:
    :return:
    """
    wn = [2*i/fs for i in cutoff]
    b,a = signal.butter(order,wn,'bandpass',analog=False)#滤波器构造函数
    filted_data = signal.filtfilt(b,a,data)#进行滤波
    return filted_data

def butter_highpass_filter(data, cutoff, fs, order=3):
    """
    数字高通巴特沃斯滤波：对一维数据滤波（只需调用这一个函数即可滤波）
    :param data: 一维数组
    :param cutoff:
    :param fs:
    :param order:
    :return: 一维数组
    """
    wn = 2*cutoff / fs    # 归一化截止频率=截止频率/信号频率，即MATLAB中butter 的 Wn
    b, a = signal.butter(order, wn, btype='highpass', analog=False)   # 原本butter只能输出一个值，但用2个变量接收也无报错
    filted_data=signal.filtfilt(b,a,data)
    return filted_data

def smooth_filter(data, alpha=0.1):
    """平滑滤波"""
    for i in range(1, len(data)):
        last = data[i-1]
        current = data[i]
        data[i] = last * (1.0 - alpha) + current * alpha
    return data

def smooth(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def calculate_energy(judge_axis):
    """
    :param judge_axis:
    :return: energy:
    """
    energy = 0
    for i in range(judge_axis.shape[0]):
        energy = energy + judge_axis[i]**2
    return energy


def calculate_peak(series):
    """
    计算数组中的极大值位置
    :param series:
    :return: 极大值的位置，从零开始
    """
    num = 0
    current_max = 0
    current_min = 0
    index_max = 0
    index_min = 0
    index_list = []
    up = True
    for i in range(0, len(series)):
        if i == 0:
            current_max = series[i]
            current_min = series[i]
            if series[i+1] >= current_max:
                up = True
            else:
                up = False
        if up:
            if series[i] >= current_max:
                current_max = series[i]
                if i+1 < len(series):
                    if series[i+1] <= current_max:
                        up = False
                        num = num+1
                        index_max = i
                        current_min = series[i+1]
                        index_list.append(i)
        else:
            if series[i] <= current_min:
                current_min = series[i]
                if i+1 < len(series):
                    if series[i+1] > current_min:
                        up = True
                        num = num + 1
                        index_min = i
                        current_max = series[i+1]
        if index_max != 0:
            if index_min != 0:
                Dis_current = index_max - index_min
                if Dis_current > 0:
                    index_min = 0
                else:
                    index_max = 0
    index_list = np.array(index_list)
    return index_list



