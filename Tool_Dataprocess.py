import numpy as np
from Tool_Pretreatment import *
import matplotlib.pyplot as plt
from Tool_Extractfeature import statistical
from User_modify import CONFIGURATION

def produce_sample_label(group_category):
    """
    Process raw experimental data of 8 channels'PVDF and 9 axis IMU.
    :param group_category: 'train' or 'test'
    :return: movedata_feature (Two-dimensional array, non-normalized feature matrix)<samples*features>,
             labels (One-dimensional array, gesture 1,2,3,4 is labeled by 1,2,3,4 sequentially),

    """
    # Specify the file_path: train or test
    move_num: int = 4
    if group_category == 'train':
        file_path = CONFIGURATION['train_data_path']
    elif group_category == 'test':
        file_path = CONFIGURATION['test_data_path']

    # Select the sensor parameter
    axis_num = 3                   # Axis number of IMU
    col_PVDF_filtered = 8              # Number of PVDF channel
    col_IMU_filtered = 3*axis_num      # Columns need to be filtered: acc, gyro
    fs = CONFIGURATION['fs']                             # Sensor sampling frequency (Hz). Here the frequency of all the 3 sensors is 100Hz
    cutoff_imu = CONFIGURATION['cutoff_imu']             # Cutoff  frequency of low pass filter
    order_imu = CONFIGURATION['order_imu']               # Order of  low pass filter
    cutoff_pvdf = CONFIGURATION['cutoff_pvdf']             # Cutoff frequency of filter
    order_pvdf = CONFIGURATION['order_pvdf']               # Order of  filter

    # Specify the parameters related to extracting features
    sta_char_pvdf = CONFIGURATION['pvdf_feature']           #pvdf feature
    sta_char_acc_x = CONFIGURATION['accx_feature']          #Acc Feature
    sta_char_acc_y = CONFIGURATION['accy_feature']
    sta_char_acc_z = CONFIGURATION['accz_feature']
    # sta_char_resultantacc = CONFIGURATION['resultantacc_feature']
    sta_char_gyro_x = CONFIGURATION['gyrox_feature']        #Gyro Feature
    sta_char_gyro_y = CONFIGURATION['gyroy_feature']
    sta_char_gyro_z = CONFIGURATION['gyroz_feature']
    # sta_char_resultantgyro = CONFIGURATION['resultantgyro_feature']

    """data read and save in movedata_0 variable"""
    movedata_0, samplingpoint_num_list = read_data(file_path)

    """
    Data filter: Configure different filters for PVDF signal and IMU signal.
    The filtered data is saved in movedata_filter variable.
    """
    movedata_filter = []
    for data in movedata_0:
        data_after = np.zeros((data.shape[0],data.shape[1]))
        for i in range(col_PVDF_filtered+col_IMU_filtered):
            if i<col_PVDF_filtered:
                data_after[:,i] = butter_lowpass_filter(data[:,i],cutoff_imu,fs,order_imu)
            else:
                data_after[:,i] = butter_bandpass_filter(data[:,i],cutoff_pvdf,fs,order_pvdf)#注意需要满足采样定理
        movedata_filter.append(data_after)

    """
    Data Segmentation: if needed: using different segmentation strategy for PVDF and IMU
    基于现在的能量极值的有效手势段检索策略
    """
    window_time = 3                    # Window length (s)
    window_len = window_time * fs     # Number of sample points in the window window_len=300
    ini_pass_len = 200          # Necessary. Need to discard the initial "ini_pass_len" sample points of the data stream
    increment = window_time*fs/2


    movedata_pre = []
    movedata = []                                             # store the valid segment
    label_pre = []
    index_pre = 0
    for data in movedata_filter:
        energy_list = []                                      # store the energy value of each segment
        index_pre = index_pre +1                              #检索得到train/test文件夹内csv文件的编号:顺序编号
        index = index_pre % 16                                #4 个手势，每个手势4次trail
        multip = int((data.shape[0]-ini_pass_len-window_len)/increment+1)    #Determine the number of segment in a trail
        for k in range(multip):
            med = ini_pass_len + int(increment)*k
            datapacket = data[med:med+int(window_len),:]
            movedata_pre.append(datapacket)
            """有效手势段检测算法"""
            judge_axis = np.sqrt(datapacket[:,8]**2+datapacket[:,9]**2+datapacket[:,10]**2)    # 选择a_resultance作为手势段能量的判别标准
            energy = sum(judge_axis**2)                                         #  计算手势段的能量
            energy_list.append(energy)
        energy_list = np.array(energy_list)
        """先进行平滑滤波，寻找能量峰值在那个窗里面，因为窗长大于手势长度，因此会有很多个极大能量窗"""
        energy_list = smooth_filter(energy_list, 0.1)
        peak_index = calculate_peak(energy_list)
        """提取能量峰值的segment """
        for n in peak_index:
            movedata.append(movedata_pre[n])
        """ 对segment加标签"""
        for m in range(peak_index.shape[0]):
            label_pre.append(index)


    """Extract features from data and finally get a non-normalized feature matrix.
    Non-normalized feature matrix: each row (eigenvector) of the matrix represents a sample, and each column of the eigenvector represents a feature.
    """
    movedata_feature = []
    for datapacket in movedata:
        feat_packet = []
        # Extract the statistical characteristics of acceleration (8 channels)
        for i in range(8):
            pvdf_feat = statistical(datapacket[:, i], sta_char_pvdf)
            feat_packet.append(pvdf_feat)
        # Extract the statistical characteristics of acceleration (3 axis)
        acc_xfeat = statistical(datapacket[:, 8], sta_char_acc_x)
        feat_packet.append(acc_xfeat)
        acc_yfeat = statistical(datapacket[:, 9], sta_char_acc_y)
        feat_packet.append(acc_yfeat)
        acc_zfeat = statistical(datapacket[:, 10], sta_char_acc_z)
        feat_packet.append(acc_zfeat)
        # Extract the statistical characteristics of angular velocity (3 axis)
        gyro_xfeat = statistical(datapacket[:, 11], sta_char_gyro_x)
        feat_packet.append(gyro_xfeat)
        gyro_yfeat = statistical(datapacket[:, 12], sta_char_gyro_y)
        feat_packet.append(gyro_yfeat)
        gyro_zfeat = statistical(datapacket[:, 13], sta_char_gyro_z)
        feat_packet.append(gyro_zfeat)

        # Reduce the dimension of feat_packet to a one-dimensional list
        feat_packet = [element for element in feat_packet if element != []]     # Remove empty elements in feat_packet
        med_vari = str(feat_packet)
        med_vari = med_vari.replace('[', '')
        med_vari = med_vari.replace(']', '')
        med_vari = eval(med_vari)
        # Store feat_packet
        if type(med_vari) == tuple:
            feat_packet = list(med_vari)
        elif type(med_vari) == float:
            feat_packet = [med_vari]
        movedata_feature.append(feat_packet)
    movedata_feature = np.array(movedata_feature)
    """Generate labels for all samples (one-dimensional array) 
        应根据训练集和测试集数据量的不同及时做出调整
    """
    label = []
    for var in label_pre:
        if (var >= 1) and (var <= 4):
            label.append(1)
        elif (var >= 5) and (var <= 8):
            label.append(2)
        elif (var >= 9) and (var <= 12):
            label.append(3)
        else:
            label.append(4)
    label = np.array(label)
    return movedata_feature, label











