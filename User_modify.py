"""
Users only need to modify this document to adjust some parameters in the initial_testing and data process programs to view the results of the classification, without having to fully understand the details of the program.
The parts that users can modify mainly include:
1. The type of training set and test set,
2. Parameters related to low pass filter,
3. Feature types extracted from the data of each sensor,
4. Machine learning model for classification



Neither of these parameters need to be modified, because the data is contained in the python project,
If you want to move the data folder to another location, you can modify these two parameters.)

'cutoff_imu': It is recommended that the cut-off frequency <=30Hz.
'order_imu': The order of the low-pass filter.
'cutoff_pvdf':The cut-off frequency for bandpass filter
'order_pvdf': The order of the band-pass filter.

'accx_feature': Feature types extracted from x-axis acceleration data.
'accy_feature': Feature types extracted from y-axis acceleration data.
'accz_feature': Feature types extracted from z-axis acceleration data.
'resultantacc_feature': Feature types extracted from resultant acceleration data.
'gyrox_feature': Feature types extracted from x-axis angular velocity data.
'gyroy_feature': Feature types extracted from y-axis angular velocity data.
'gyroz_feature': Feature types extracted from z-axis angular velocity data.
'resultantgyro_feature': Feature types extracted from resultant angular velocity data.

The optional features are as follows: 均值、无偏标准差、偏度、峰度、均方根、平均绝对偏差、四分位差、整流平均值,波形因子,频谱峰值,频谱峰值频率,正负数据交替出现的次数. 趋势(仅仅用为气压数据的特征)
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
    Spectrum peak frequency: 'SPF'
    Number of alternating occurrences of positive or negative values: 'APN'.

    Trend (specially for air pressure data, extracting this feature from data of other types of sensors is useless): 'TREND'.
Please strictly follow the format below to modify the parameters:
    If you don't want any features for the data of one certain sensor, use [] or ''.
    If you want one feature, use ['MEAN'].
    If you want several features, use ['MEAN', 'USD'].
    (The order of the features has been determined by the program and has nothing to do with the order of the string flags entered by the user.)
    If you want all the features (not including 'TREND') for acc or gyro data, please use 'ALL'.
    Note that for air pressure data, the "TREND" feature is enough. The air pressure data does not have more information.

'model': Machine learning model for classification.
The optional models are as follows: K最近邻, 支持向量机, 随机森林, 线性判别分析，人工神经网络，高斯贝叶斯
    K-NearestNeighbor: 'KNN'.
    Support vector machines: 'SVM'.
    Random Forest: 'RF'.
    Linear Discriminant Analysis: 'LDA'.
    Artificial Neural Network: 'MLP'.
    TrAdaboost: 'TrA' (only for Transfer Task).
Please strictly follow the format below to modify the parameter:
    Can not be empty.
    If you want one model, use ['KNN'].
    If you want all models, use ['KNN', 'SVM', 'RF', 'LDA', 'GNB'].
    For Transfer Task, you can use ['KNN', 'SVM', 'RF', 'LDA', 'GNB', 'TrA'] to compare performance of ML (Machine Learning) and TL (Transfer Learning) models.
"""

CONFIGURATION = {
                 'train_data_path': "./data/train",             # Data path
                 'test_data_path': "./data/test",
                 'cutoff_imu': 20.0, 'order_imu': 3,'cutoff_pvdf': [1.0, 20.0], 'order_pvdf': 3,'fs': 100.0,  # Filter Configuration
                 'pvdf_feature': ['IR','MAD'],                            # feature set
                 'accx_feature': ['IR','MAD', 'USD','RM','WL'],
                 'accy_feature': ['WF','RM','WL'],
                 'accz_feature': ['IR', 'USD', 'MAD','RM','WL'],
                 'resultantacc_feature': ['WF', 'MAD', 'IR', 'MEAN', 'RM', 'USD', 'RMS'],
                 'gyrox_feature': ['RM','WL'],
                 'gyroy_feature': ['RM','WL'],
                 'gyroz_feature': ['RM','WL'],
                 'resultantgyro_feature': '',
                 'model': ['SVM']}        # 'KNN', , 'LDA'
