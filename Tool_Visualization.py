"""可视化模块"""

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt    # 绘图库
import matplotlib.ticker as ticker
font1 = {'family': 'Times New Roman',
'weight': 'normal',

}
def plot_confusion_matrix(cm, labels_name, title):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 8))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化 np.newaxis是增加维度，这里的除法用的是数组广播的性质

    #plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)    # 在特定的窗口上显示图像
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)    # imshow用以绘制热力图，interprolation参数控制热图的显示形式
    plt.title(title,font1, fontsize=16)    # 图像标题
    cb1 =plt.colorbar(fraction=0.04)
    cb1.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(nbins=6)
    cb1.locator = tick_locator
    cb1.update_ticks()
    num_local = np.array(range(len(labels_name)))

    plt.xticks(num_local, labels_name, fontproperties ='Times New Roman',fontsize=16)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontproperties ='Times New Roman',fontsize=16)    # 将标签印在y轴坐标上
    plt.ylabel('True Label',font1, fontsize=16)
    plt.xlabel('Predicted Label',font1, fontsize=16)
    cm = cm.T
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            a = cm[first_index][second_index]
            b = "%.1f%%" % (a * 100)
            #
            if first_index == second_index: #and first_index < 5:
                plt.text(first_index, second_index, b, fontsize=15,  color="w", va='center', ha='center')
            #elif first_index == 4 and second_index ==5:
                #plt.text(first_index, second_index, b, fontsize=15,  color="w", va='center', ha='center')
            #elif first_index == 4 and second_index ==6:
                #plt.text(first_index, second_index, b, fontsize=15,  color="w", va='center', ha='center')
            else:
                plt.text(first_index, second_index, b, fontsize=15, va='center', ha='center')


def np_fig(length, data):
    """输入一维数组进行绘图"""
    plt.figure()
    plt.plot(np.arange(length), data)
    plt.show()


def df_row_fig(df, row_list, label_list, object_num, x_start, x_step_length, y_start, y_stop):
    """
    :param df: 输入的Data frame
    :param row_list: 所需要绘图的行 e.g.,[1310, 2710, 3710, 5110, 6160, 7360, 8860]
    :param label_list: 行对应的坐标['140[deg]','130[deg]','120[deg]','110[deg]','100[deg]','90[deg]','80[deg]']
    :param object_num: 点数
    :param x_start: X轴坐标起始点
    :param x_step_length: X轴坐标步长
    :param y_start: Y轴坐标起始点
    :param y_stop: Y轴坐标终止点
    :return:
    """
    plt.figure()
    my_x_ticks = np.arange(x_start, x_start+object_num*x_step_length, x_step_length)
    for i in range(0, len(row_list)):
        subject_draw = np.array(df.loc[row_list[i], 1:object_num])
        plt.plot(my_x_ticks, subject_draw, label=label_list[i])
    plt.legend()
    plt.ylim(y_start, y_stop)
    plt.ylabel()
    plt.xlabel()
    # plt.xticks(my_x_ticks)
    plt.show()


def df_column_fig(df, column_list, label_list, object_num, x_start, x_step_length, y_start, y_stop):
    """
    :param df:输入的Data frame
    :param column_list:所需要绘图的列 E.g.,['ACC_X','ACC_Y','ACC_Z','GYR_X','GYR_Y','GYR_Z','BIO']
    :param label_list:列对应的坐标
    :param object_num:点数
    :param x_start:X轴坐标起始点
    :param x_step_length:X轴坐标步长
    :param y_start:Y轴坐标起始点
    :param y_stop:Y轴坐标终止点
    :return:
    """
    plt.figure()
    my_x_ticks = np.arange(x_start, x_start+object_num*x_step_length, x_step_length)
    for i in range(0, len(column_list)):
        subject_draw = np.array(df.iloc[column_list[i], 1:object_num])
        plt.plot(my_x_ticks, subject_draw, label=label_list[i])
    plt.legend()
    plt.ylim(y_start, y_stop)
    plt.ylabel()
    plt.xlabel()
    # plt.xticks(my_x_ticks)
    plt.show()


