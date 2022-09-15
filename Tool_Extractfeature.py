import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.fftpack import fft


def number_alternate_pn(array):
    """
    The number of alternating occurrences of positive or negative values.
    :param array: A column of data (array)
    :return: int
    """
    n = array.shape[0]
    temp = 0
    for i in range(n-1):
        if array[i] * array[i + 1] < 0:
            temp += 1
    return temp

def slope_sign_change (array):
    """
    The number of alternating occurrences of slope signs.
    :param array: A column of data (array)
    :return: int
    """
    n = array.shape[0]
    temp = 0
    for i in range(1,n-1):
        if (array[i]-array[i-1])*(array[i+1]-array[i]) < 0:
            temp += 1
    return temp

def waveform_length (array):
    """
    calculate the waveform_length of the signal
    :param array: A column of data (array)
    :return: int
    """
    n = array.shape[0]
    temp = 0
    for i in range(1,n):
        temp = temp + abs(array[i] - array[i-1])
    return temp






def statistical(data, sta_char):
    """
    Extract statistical features from the data in each window.
    :param data: A column of data (array)
    :param sta_char: String flag.
    Mean: 'MEAN'.
    Unbiased standard deviation: 'USD'.
    Skewness: 'SK'.
    Kurtosis: 'KU'.
    Root mean square: 'RMS'.
    Mean absolute deviation: 'MAD'.
    Interquartile range: 'IR'.
    Rectified mean: 'RM'.
    Waveform factor: 'WF'.
    Spectral peak: 'SP'.
    Spectrum peak frequency: 'SPF'.
    Number of alternating occurrences of positive or negative values: 'APN'.
    slope_sign_change: 'SSC'
    waveform_length: 'WL'

    Trend (specially for air pressure data): 'TREND'.

    If you don't want any features, use [] or ''.
    If you want one feature, use ['USD'].
    If you want several features, use ['USD', 'WF'].
    (The order of the features has been determined by the program and has nothing to do with the order of the string flags entered by the user.)
    If you want all the features (not including 'TREND'), we recommend using 'ALL'.
    :return: sta_data (list, statistical features extracted)
    """
    if sta_char == [] or sta_char == '':          # If you don't want any features
        return []
    else:
        mean = data.mean()  # Mean, np.mean(data)
        standard_deviation = np.std(data, ddof=1)  # Unbiased standard deviation
        data_series = pd.Series(data)
        skewness = data_series.skew()  # Skewness
        kurtosis = data_series.kurt()  # Kurtosis
        root_mean_square = math.sqrt(sum([x ** 2 for x in data]) / len(data))  # Root mean square
        mean_absolute_deviation = data_series.mad()  # Mean absolute deviation
        interquartile_range = stats.scoreatpercentile(data_series, 75) - stats.scoreatpercentile(data_series, 25)  # interquartile range
        rectified_mean = np.mean(abs(data))  # Rectified mean
        waveform_factor = root_mean_square / rectified_mean  # Waveform factor
        alternate_pn = number_alternate_pn(data)
        trend = data[-1] - data[0]      # For air pressure data
        SSC = slope_sign_change(data)
        WL = waveform_length(data)

        # Spectral peak
        N = data.shape[0]
        data = data - mean # 计算FFT之前需要先减去直流漂移
        spectral = fft(data)
        abs_spectral = np.abs(spectral)/N
        abs_spectral_half = abs_spectral[range(int(N / 2))]
        spectral_peak = abs_spectral_half.max()
        spectrum_peak_freq = int(np.argwhere(abs_spectral_half == spectral_peak))/3     # frequency conversion: freq=n*fs/N

        sta_char_dict = {'MEAN': mean, 'USD': standard_deviation, 'SK': skewness, 'KU': kurtosis,
                         'RMS': root_mean_square, 'MAD': mean_absolute_deviation,
                         'IR': interquartile_range, 'RM': rectified_mean, 'WF': waveform_factor,
                         'SP': spectral_peak, 'SPF': spectrum_peak_freq, 'APN': alternate_pn,
                         'TREND': trend, 'SSC': SSC, 'WL': WL}

        if sta_char == 'ALL':                           # If you want all the features
            sta_data = [mean, standard_deviation, skewness, kurtosis, root_mean_square,
                        mean_absolute_deviation, interquartile_range, rectified_mean, waveform_factor,
                        spectral_peak, spectrum_peak_freq, alternate_pn, SSC, WL]
            return sta_data
        else:
            sta_data = []
            for string_flag in sta_char:                # If you want several features or all the features
                sta_data.append(sta_char_dict[string_flag])
            return sta_data
