#!/usr/bin/env python
# coding: utf-8

from catboost import Pool, CatBoostClassifier, cv
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime,timedelta  
import matplotlib.pyplot as plt
import time

import gc
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack, vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from skopt.space import Integer, Categorical, Real, Log10
from skopt.utils import use_named_args
from skopt import gp_minimize
import re


train = pd.read_csv('../dataset/deviceid_train.tsv', sep='\t', names=['device_id', 'sex', 'age'])
all_feat = pd.read_csv('../dataset/all_feat.csv')

train['label'] = train['sex'].astype(str) + '-' + train['age'].astype(str)
label_le = preprocessing.LabelEncoder()
train['label'] = label_le.fit_transform(train['label'])
data_all = pd.merge(left=all_feat, right=train, on='device_id', how='left')


train = data_all[:50000]
test = data_all[50000:]
train = train.fillna(-1)
test = test.fillna(-1)
del data_all
gc.collect()

use_feats = all_feat.columns[1:]
use_feats

X_train = train[use_feats]
X_test = test[use_feats]
Y = train['label']
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
sub = np.zeros((X_test.shape[0], 22))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)):
    X_tr, X_vl, y_tr, y_vl = X_train.iloc[train_index], X_train.iloc[test_index],                                 Y.iloc[train_index], Y.iloc[test_index]
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=[-1])
    dvalid = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    params = {
        'boosting_type': 'gbdt',
        'max_depth':6,
        'metric': {'multi_logloss'},
        'num_class':22,
        'objective':'multiclass',
        'num_leaves':7,
        'subsample': 0.9,
        'colsample_bytree': 0.2,
        'lambda_l1':0.0001,
        'lambda_l2':0.00111,
        'subsample_freq':12,
        'learning_rate': 0.012,
        'min_child_weight':12

    }

    model = lgb.train(params,
                        dtrain,
                        num_boost_round=6000,
                        valid_sets=dvalid,
                        early_stopping_rounds=100,
                        verbose_eval=100)


    sub += model.predict(X_test, num_iteration=model.best_iteration)/kfold.n_splits


sub = pd.DataFrame(sub)
cols = [x for x in range(0, 22)]
cols = label_le.inverse_transform(cols)
sub.columns = cols
sub['DeviceID'] = test['device_id'].values
sub = sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
sub.to_csv('lgb_22.csv', index=False)





