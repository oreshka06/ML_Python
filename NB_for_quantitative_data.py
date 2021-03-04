# загрузка библиотек и файла

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("D:\\Users\\User\\Downloads\\mlbootcamp5_train.csv", 
                 sep=";", 
                 index_col="id")
print (df.head())

# 1. Построить наивный байесовский классификатор для количественных полей age, height, weight, ap_hi, ap_lo. 
# Исправить данные, если это необходимо. 

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

train = df[["age", "weight", "height", "ap_hi", "ap_lo"]]
target = df["cardio"]

model = gnb.fit(train, target)
predict = model.predict(train)
print(df.shape[0],
     (target == predict).sum() / df.shape[0])

predict

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(target, predict)

import itertools
class_names = ["Здоров", "Болен"]
def plot_confusion_matrix(cm, classes, normalize=False, title='Матрица неточностей', cmap=plt.cm.Blues):    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.4f' if normalize else 'd'
    thresh = cm.min() + (cm.max() - cm.min()) * 2 / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Истина')
    plt.xlabel('Предсказание')
    plt.tight_layout()

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Матрица неточностей, нормализована')
plt.show()
