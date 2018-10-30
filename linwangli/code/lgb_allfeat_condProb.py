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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
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


# 读入数据
train = pd.read_csv('../dataset/deviceid_train.tsv', sep='\t', names=['device_id', 'sex', 'age'])
all_feat = pd.read_csv('../dataset/all_feat.csv')


data_all = pd.merge(left=all_feat, right=train, on='device_id', how='left')
train = data_all[:50000]
test = data_all[50000:]
train = train.fillna(-1)
test = test.fillna(-1)
del data_all
gc.collect()
use_feats = all_feat.columns[1:]
use_feats


# P(age)

Y = train['sex'] - 1
X_train = train[use_feats]
X_test = test[use_feats]
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
oof_preds1 = np.zeros((X_train.shape[0], ))
sub1 = np.zeros((X_test.shape[0], ))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)):
    X_tr, X_vl, y_tr, y_vl = X_train.iloc[train_index], X_train.iloc[test_index],                                 Y.iloc[train_index], Y.iloc[test_index]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    params = {
        'boosting_type': 'gbdt',
        'max_depth':6,
        'objective':'binary',
        'num_leaves':31,
        'subsample': 0.85,
        'colsample_bytree': 0.2,
        'lambda_l1':0.00007995302080034896,
        'lambda_l2':0.0003648648811380991,
        'subsample_freq':12,
        'learning_rate': 0.012,
        'min_child_weight':5.5
    }

    model = lgb.train(params,
                        dtrain,
                        num_boost_round=4000,
                        valid_sets=dvalid,
                        early_stopping_rounds=100,
                        verbose_eval=100)

    oof_preds1[test_index] = model.predict(X_vl, num_iteration=model.best_iteration)
    sub1 += model.predict(X_test, num_iteration=model.best_iteration)/kfold.n_splits


# P(age|sex = 2)

train['sex_pred'] = train['sex']
test['sex_pred'] = 1

use_feats = list(train.columns[1:-3])
use_feats = use_feats + ['sex_pred']

X_train = train[use_feats]
X_test = test[use_feats]

Y = train['age']
kfold = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
oof_preds2_1 = np.zeros((X_train.shape[0], 11))
sub2_1 = np.zeros((X_test.shape[0], 11))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)):
    X_tr, X_vl, y_tr, y_vl = X_train.iloc[train_index], X_train.iloc[test_index],                                 Y.iloc[train_index], Y.iloc[test_index]

    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    params = {
        'boosting_type': 'gbdt',
        'max_depth':6,
        'metric': {'multi_logloss'},
        'num_class':11,
        'objective':'multiclass',
        'num_leaves':31,
        'subsample': 0.9,
        'colsample_bytree': 0.2,
        'lambda_l1':0.0001,
        'lambda_l2':0.00111,
        'subsample_freq':10,
        'learning_rate': 0.012,
        'min_child_weight':10
    }

    model = lgb.train(params,
                        dtrain,
                        num_boost_round=4000,
                        valid_sets=dvalid,
                        early_stopping_rounds=100,
                        verbose_eval=100)

    oof_preds2_1[test_index] = model.predict(X_vl, num_iteration=model.best_iteration)
    sub2_1 += model.predict(X_test, num_iteration=model.best_iteration)/kfold.n_splits


# P(age|sex = 2)

train['sex_pred'] = train['sex']
test['sex_pred'] = 2

use_feats = list(train.columns[1:-3])
use_feats = use_feats + ['sex_pred']

X_train = train[use_feats]
X_test = test[use_feats]


Y = train['age']
kfold = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
oof_preds2_2 = np.zeros((X_train.shape[0], 11))
sub2_2 = np.zeros((X_test.shape[0], 11))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)):
    X_tr, X_vl, y_tr, y_vl = X_train.iloc[train_index], X_train.iloc[test_index],                                 Y.iloc[train_index], Y.iloc[test_index]

    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    params = {
        'boosting_type': 'gbdt',
        'max_depth':6,
        'metric': {'multi_logloss'},
        'num_class':11,
        'objective':'multiclass',
        'num_leaves':31,
        'subsample': 0.9,
        'colsample_bytree': 0.2,
        'lambda_l1':0.0001,
        'lambda_l2':0.00111,
        'subsample_freq':10,
        'learning_rate': 0.012,
        'min_child_weight':10
    }

    model = lgb.train(params,
                        dtrain,
                        num_boost_round=4000,
                        valid_sets=dvalid,
                        early_stopping_rounds=100,
                        verbose_eval=100)

    oof_preds2_2[test_index] = model.predict(X_vl, num_iteration=model.best_iteration)
    sub2_2 += model.predict(X_test, num_iteration=model.best_iteration)/kfold.n_splits


# 保存测试集的预测结果
sub1 = pd.DataFrame(sub1, columns=['sex2'])

sub1['sex1'] = 1-sub1['sex2']
sub2 = pd.DataFrame(sub2_1, columns=['age%s'%i for i in range(11)])
sub = pd.DataFrame(test['device_id'].values, columns=['DeviceID'])

for i in ['sex1', 'sex2']:
    for j in ['age%s'%i for i in range(11)]:
        sub[i+'_'+j] = sub1[i] * sub2[j]
sub.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

sub.to_csv('test_pred.csv', index=False)


# 保存训练集五折的预测结果
oof_preds1 = pd.DataFrame(oof_preds1, columns=['sex2'])
oof_preds1['sex1'] = 1-oof_preds1['sex2']

oof_preds2_1 = pd.DataFrame(oof_preds2_1, columns=['age%s'%i for i in range(11)])
oof_preds2_2 = pd.DataFrame(oof_preds2_2, columns=['age%s'%i for i in range(11)])

oof_preds = train[['device_id']]
oof_preds.columns = ['DeviceID']

for i in ['age%s'%i for i in range(11)]:
    oof_preds['sex1_'+i] = oof_preds1['sex1'] * oof_preds2_1[i]
for i in ['age%s'%i for i in range(11)]:
    oof_preds['sex2_'+i] = oof_preds1['sex2'] * oof_preds2_2[i]   

oof_preds.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

oof_preds.to_csv('train_pred.csv', index=False)





