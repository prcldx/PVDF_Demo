from User_modify import *
#from User_modify import CONFIGURATION
from Tool_Dataprocess import produce_sample_label
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as sm
import numpy as np
from Tool_Visualization import *
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


"""
Build various classification models
KNN; SVM;  RF; LDA;
"""
knn = KNeighborsClassifier()                #KNN Modeling
clf = svm.SVC(kernel='rbf')
#clf = svm.LinearSVC(dual=False)             #Linear SVM Modeling
rf = RandomForestClassifier(n_estimators=16, max_depth=8)       #RF Modeling
lda = LinearDiscriminantAnalysis()


model_dict = {'KNN':knn,'SVM':clf,'RF':rf,'LDA':lda}

"""Read the model type that the user wants to train and test"""
if CONFIGURATION['model'] == '' or []:
    print('Error! At least one model should be entered')
else:
    selected_model = []
    for string_flag in CONFIGURATION['model']:  # If you want several models or all the alternative models
        selected_model.append(model_dict[string_flag])#把一个实例化的模型加入列表

labelstr_list = ['HG1', 'HG2', 'HG3', 'HG4']#对四个手势进行分类

#if needed, write a function to divide train data and test data

train_dataset, train_label = produce_sample_label('train')
test_dataset, test_label = produce_sample_label('test')

"""
Classification Model validation
需要继续改进，加入k-fold validation
"""
j = 0
for model in selected_model:
    model_scorelist = []
    #先进行归一化
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(train_dataset)
    train_dataset = scaler.transform(train_dataset)
    test_dataset = scaler.transform(test_dataset)
    #划分批次，送入进行训练
    #if model == clf:
    print(CONFIGURATION['model'])
    if CONFIGURATION['model'] == ['SVM']:
        print('1')
        kf = KFold(n_splits=2)
        turned_parameter = tuned_parameters = [{"kernel": ["rbf"], "C": [10, 100]}]
        model = GridSearchCV(model, tuned_parameters, cv=kf)#这里验证一下gridsearchCV能不能这么用，结果证明，可以这么用，可以用CV传递交叉验证参数
        #val_scores = cross_val_score(model, train_dataset, train_label, cv=kf)#虽然model有很多参数，在单独调用交叉验证时，只是对第一个参数模型进行了交叉验证
        model.fit(X=train_dataset, y=train_label)
        means = model.cv_results_["mean_test_score"]#返回的结果是每个参数交叉验证得到的平均值
    else:
        kf = KFold(n_splits=5)
        val_scores = cross_val_score(model, train_dataset, train_label, cv=kf)
        model.fit(X=train_dataset,y=train_label)

    #testing
    model_scorelist.append(model.score(test_dataset,test_label))
    pred_y_test = model.predict(test_dataset)

    #打印正确率
    print("%s model" % (CONFIGURATION['model'][j]))
    print("test the samples of each child subject", model_scorelist, sep=':')
    print("mean score: %.1f%%\n" % (np.mean(model_scorelist) * 100))
    j += 1


# 获取混淆矩阵和分类报告
labels_name = ['1','2','3','4']
m = sm.confusion_matrix(test_label, pred_y_test)
#plot_confusion_matrix(m, labels_name, "4 gestures")

print('混淆矩阵为：', m, sep='\n')
r = sm.classification_report(test_label, pred_y_test)
print('分类报告为：', r, sep='\n')

print('1')







