# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 18:13:43 2014

@author: 深津恵梨子
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer # 欠損値を埋める
from sklearn.svm import SVC

print u'訓練データ読み込み中'
df=pd.read_csv("C:\\Users\\fukazu\\Documents\\IPython Notebooks\\deepanAlytics\\train.csv",header=None)

x = df.loc[:, 2:]
y = df[1]

x.loc[:, 4:9] = x.loc[:, 4:9].astype(str)

print u'数量化中'
vectorizer = DictVectorizer(sparse=True)
x = vectorizer.fit_transform(x.to_dict('records'))

print u'欠損値を埋める'
inp = Imputer()
x = inp.fit_transform(x)

print u'特徴ベクトル生成中'
gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=7, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
gbdt.fit(x,toarray(),y)
x_new = gbdt.transform(x)

clf = SVC()
clf.fit(x_new, y)

print u'テストデータ読み込み中'
df=pd.read_csv("C:\\Users\\fukazu\\Documents\\IPython Notebooks\\deepanAlytics\\test.csv",header=None)

x = df.loc[:, 1:]

x.loc[:, 4:9] = x.loc[:, 4:9].astype(str)

print u'数量化中'
x = vectorizer.transform(x.to_dict('records'))

print u'欠損値を埋める'
x = inp.transform(x)

print u'予測中'
x_new = gbdt.transform(x)
answer = clf.predict(x_new)

print u'サブミットファイル生成'
index = df[0]
data = np.array([index.astype(int), answer.astype(float)], dtype=object).T # インデックスと予測値(実数に変換)を結合
np.savetxt("C:\\Users\\fukazu\\Documents\\IPython Notebooks\\deepanAlytics\\submit.csv", data, delimiter=',')
