# -*- coding:utf-8 -*-


#######  尝试骚操作，单独针对这个表
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC

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

deviceid_packages = pd.read_csv('Demo/deviceid_packages.tsv', sep='\t', header=None)
deviceid_packages = deviceid_packages.rename({0: 'id', 1: 'packages_names'}, axis=1)
package_label = pd.read_csv('Demo/package_label.tsv', sep='\t', header=None)
package_label = package_label.rename({0:'packages_name', 1:'packages_type'},axis=1)
# package_label['packages_type'] = package_label.apply(lambda row:row['packages_type'] + ' ' + row[2], axis=1)
dict_label = dict(zip(list(package_label['packages_name']), list(package_label['packages_type'])))

data_all = pd.merge(data_all, deviceid_packages, on='id', how='left')

feature = pd.DataFrame()

import numpy as np

# app个数
# 毒特征？
# feature['app_count'] = data_all['packages_names'].apply(lambda row: len(str(row).split(',')))

# 对此数据做countvector,和tfidfvector,并在一起跑几个学习模型
# 引申出来的count和tfidf，跑基本机器学习分类模型
data_all['package_str'] = data_all['packages_names'].apply(lambda row: str(row).replace(',', ' '))
def get_more_information(row):
    result = ' '
    start = True
    row_list = row.split(',')
    for i in row_list:
        try:
            if start:
                result = dict_label[i]
                start = False
            else:
                result = result + ' ' + dict_label[i]
        except KeyError:
            pass
    return result
data_all['package_str_more_information'] = data_all['packages_names'].apply(lambda row: get_more_information(str(row)))

print(data_all)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from sklearn.cross_validation import StratifiedKFold


count_vec = CountVectorizer()
count_csr_basic = count_vec.fit_transform(data_all['package_str'])
tfidf_vec = TfidfVectorizer()
tfidf_vec_basic = tfidf_vec.fit_transform(data_all['package_str'])

count_vec = CountVectorizer()
count_csr_more = count_vec.fit_transform(data_all['package_str_more_information'])

tfidf_vec = TfidfVectorizer()
tfidf_vec_more = tfidf_vec.fit_transform(data_all['package_str_more_information'])

data_feature = scipy.sparse.csr_matrix(scipy.sparse.hstack([count_csr_basic, tfidf_vec_basic,
                     count_csr_more, tfidf_vec_more]))

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
    df_stack['tfidf_lr_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_lr_error_single_classfiy.csv', index=None, encoding='utf8')
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
    df_stack['tfidf_sgd_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_sgd_error_single_classfiy.csv', index=None, encoding='utf8')
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
df_stack.to_csv('feature/tfidf_pac_error_single_classfiy.csv', index=None, encoding='utf8')
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
    stack_test +=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
        score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_ridge_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_ridge_error_single_classfiy.csv', index=None, encoding='utf8')
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
df_stack.to_csv('feature/tfidf_bnb_error_single_classfiy.csv', index=None, encoding='utf8')
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
df_stack.to_csv('feature/tfidf_mnb_error_single_classfiy.csv', index=None, encoding='utf8')
print('MultinomialNB特征已保存\n')

############################ Linersvc(LinerSVC) ################################
print('LinerSVC stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    lsvc = LinearSVC(random_state=1017)
    lsvc.fit(train_feature[tr], score[tr])
    score_va = lsvc._predict_proba_lr(train_feature[va])
    score_te = lsvc._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['tfidf_lsvc_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv('feature/tfidf_lsvc_error_single_classfiy.csv', index=None, encoding='utf8')
print('LSVC特征已保存\n')


kmeans_result = pd.DataFrame()
###### kmeans ###
def get_cluster(num_clusters):
    print('开始' + str(num_clusters))
    name = 'kmean'
    print(name)
    model = KMeans(n_clusters=num_clusters, max_iter=300, n_init=1, \
                        init='k-means++', n_jobs=10, random_state=1017)
    result = model.fit_predict(data_feature)
    kmeans_result[name + 'word_' + str(num_clusters)] = result

get_cluster(5)
get_cluster(10)
get_cluster(19)
get_cluster(30)
get_cluster(40)
get_cluster(50)
get_cluster(60)
get_cluster(70)
kmeans_result.to_csv('feature/cluster_tfidf_feature.csv', index=False)



feature.to_csv('feature/deviceid_package_feature.csv', index=False)
