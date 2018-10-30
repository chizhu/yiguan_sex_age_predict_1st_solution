# -*- coding:utf-8 -*-

import pandas as pd
import scipy.sparse
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train = pd.read_csv('Demo/deviceid_train.tsv', sep='\t', header=None)
test = pd.read_csv('Demo/deviceid_test.tsv', sep='\t', header=None)

data_all = pd.concat([train, test], axis=0)
data_all = data_all.rename({0:'id'}, axis=1)
del data_all[1],data_all[2]

start_close_time = pd.read_csv('Demo/deviceid_package_start_close.tsv', sep='\t', header=None)
start_close_time = start_close_time.rename({0:'id', 1:'app_name', 2:'start_time', 3:'close_time'}, axis=1)

start_close_time = start_close_time.sort_values(by='start_time')

start_close_time['start_time'] = map(int,start_close_time['start_time']/1000)
start_close_time['close_time'] = map(int,start_close_time['close_time']/1000)

unique_app_name = np.unique(start_close_time['app_name'])
dict_label = dict(zip(list(unique_app_name), list(np.arange(0, len(unique_app_name), 1))))
import time
start_close_time['app_name'] = start_close_time['app_name'].apply(lambda row: str(dict_label[row]))

del start_close_time['start_time'], start_close_time['close_time']

from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())
def dealed_row(row):
    app_name_list = list(row['app_name'])
    return ' '.join(app_name_list)

data_feature = start_close_time.groupby('id').progress_apply(lambda row:dealed_row(row)).reset_index()
data_feature = pd.merge(data_all, data_feature, on='id', how='left')
del data_feature['id']

count_vec = CountVectorizer(ngram_range=(1,3))
count_csr_basic = count_vec.fit_transform(data_feature[0])
tfidf_vec = TfidfVectorizer(ngram_range=(1,3))
tfidf_vec_basic = tfidf_vec.fit_transform(data_feature[0])

data_feature = scipy.sparse.csr_matrix(scipy.sparse.hstack([count_csr_basic, tfidf_vec_basic]))


from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold

train = pd.read_csv('Demo/deviceid_train.tsv', sep='\t', header=None)
test = pd.read_csv('Demo/deviceid_test.tsv', sep='\t', header=None)
def get_label(row):
    if row[1] == 1:
        return row[2]
    else:
        return row[2] + 11
train['label'] = train.apply(lambda row:get_label(row), axis=1)
data_all = pd.concat([train, test], axis=0)
data_all = data_all.rename({0:'id'}, axis=1)
del data_all[1],data_all[2]

train_feature = data_feature[:len(train)]
score = train['label']
test_feature = data_feature[len(train):]
number = len(np.unique(score))

# 五则交叉验证
n_folds = 5
print('处理完毕')

########################### lr(LogisticRegression) ################################
print('lr stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    clf = LogisticRegression(random_state=1017, C=8)
    clf.fit(train_feature[tr], score[tr])
    score_va = clf.predict_proba(train_feature[va])
    score_te = clf.predict_proba(test_feature)
    print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_lr_2_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_lr_1_3_error_single_classfiy.csv', index=None, encoding='utf8')
print('lr特征已保存\n')

########################### SGD(随机梯度下降) ################################
print('sgd stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    sgd = SGDClassifier(random_state=1017, loss='log')
    sgd.fit(train_feature[tr], score[tr])
    score_va = sgd.predict_proba(train_feature[va])
    score_te = sgd.predict_proba(test_feature)
    print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_2_sgd_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_sgd_1_3_error_single_classfiy.csv', index=None, encoding='utf8')
print('sgd特征已保存\n')

########################### pac(PassiveAggressiveClassifier) ################################
print('PAC stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    pac = PassiveAggressiveClassifier(random_state=1017)
    pac.fit(train_feature[tr], score[tr])
    score_va = pac._predict_proba_lr(train_feature[va])
    score_te = pac._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_pac_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_pac_1_3_error_single_classfiy.csv', index=None, encoding='utf8')
print('pac特征已保存\n')


########################### ridge(RidgeClassfiy) ################################
print('RidgeClassfiy stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    ridge = RidgeClassifier(random_state=1017)
    ridge.fit(train_feature[tr], score[tr])
    score_va = ridge._predict_proba_lr(train_feature[va])
    score_te = ridge._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_ridge_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_ridge_1_3_error_single_classfiy.csv', index=None, encoding='utf8')
print('ridge特征已保存\n')


########################### bnb(BernoulliNB) ################################
print('BernoulliNB stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    bnb = BernoulliNB()
    bnb.fit(train_feature[tr], score[tr])
    score_va = bnb.predict_proba(train_feature[va])
    score_te = bnb.predict_proba(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], bnb.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_bnb_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_bnb_1_3_error_single_classfiy.csv', index=None, encoding='utf8')
print('BernoulliNB特征已保存\n')

########################### mnb(MultinomialNB) ################################
print('MultinomialNB stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    mnb = MultinomialNB()
    mnb.fit(train_feature[tr], score[tr])
    score_va = mnb.predict_proba(train_feature[va])
    score_te = mnb.predict_proba(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], mnb.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_mnb_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_mnb_1_3_error_single_classfiy.csv', index=None, encoding='utf8')
print('MultinomialNB特征已保存\n')
