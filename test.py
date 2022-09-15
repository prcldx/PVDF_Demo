import numpy as np

# fs = 100.0
# data = np.random.randn(1000,3)
# print(data)
# window_time = 3  # Window length (s)
# window_len = window_time * fs  # Number of sample points in the window window_len=300
# ini_pass_len = 200  # Necessary. Need to discard the initial "ini_pass_len" sample points of the data stream
# overlap = window_time * fs / 3
# print('1')
# multi = multip = int((data.shape[0]-ini_pass_len-window_len)/overlap+1)
# print(multi)


# movedata_filter = [1,1,1,4,5]
# data = 1
# index_pre = movedata_filter.index(data)
# print(index_pre)
#
# print(movedata_filter[3])


subject = [0]
hand_gesture = [1, 2, 3, 4]
trail = [1, 2, 3, 4, 5]
label_pre = []
window_number_list = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
print(len(window_number_list))

for i in subject:
    for j in hand_gesture:
        for k in trail:
            label = [i, j, k]
            index = i * 20 + (j - 1) * 5 + k-1
            for seg in range(window_number_list[index]):
                label_pre.append(label)

label_pre = np.array(label_pre)
print(1)