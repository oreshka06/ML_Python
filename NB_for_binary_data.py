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

# 2. Написать свой наивный байесовский классификатор для бинарных полей gender, smoke, alco, active. 
# Привести матрицу неточностей и сравнить с предыдущими значениями.

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()

train = df[["gender", "smoke", "alco", "active"]]
target = df["cardio"]

model = clf.fit(train, target) 
predict = model.predict(train)
print(df.shape[0], (target == predict).sum() / df.shape[0])

predict