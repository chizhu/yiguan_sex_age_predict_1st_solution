# -*- coding:utf-8 -*-


#######  尝试骚操作，单独针对这个表
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, Ridge, \
    PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC, LinearSVR

train = pd.read_csv('Demo/deviceid_train.tsv', sep='\t', header=None)
test = pd.read_csv('Demo/deviceid_test.tsv', sep='\t', header=None)
test_id = test[0]
def get_label(row):
    return row[2]
train['label'] = train.apply(lambda row:get_label(row), axis=1)
data_all = pd.concat([train, test], axis=0)
data_all = data_all.rename({0:'id'}, axis=1)
del data_all[1],data_all[2]

deviceid_packages = pd.read_csv('Demo/deviceid_packages.tsv', sep='\t', header=None)
deviceid_packages = deviceid_packages.rename({0: 'id', 1: 'packages_names'}, axis=1)
package_label = pd.read_csv('Demo/package_label.tsv', sep='\t', header=None)
package_label = package_label.rename({0:'packages_name', 1:'packages_type'},axis=1)
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

X = train_feature
test = test_feature
y = score

n_flods = 5
kf = KFold(n_splits=n_flods,shuffle=True,random_state=1017)
kf = kf.split(X)

def xx_mse_s(y_true,y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res': list(y_pre)})
    return mean_squared_error(y_true,y_pre['res'].values)

######################## ridge reg #########################3
cv_pred = []
xx_mse = []
stack = np.zeros((len(y),1))
stack_te = np.zeros((len(test_id),1))
model_1 = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=250, normalize=False, tol=0.01,random_state=1017)
for i ,(train_fold,test_fold) in enumerate(kf):
    X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
    model_1.fit(X_train, label_train)
    val_ = model_1.predict(X=X_validate)
    stack[test_fold] = np.array(val_).reshape(len(val_),1)
    print(xx_mse_s(label_validate, val_))
    cv_pred.append(model_1.predict(test))
    xx_mse.append(xx_mse_s(label_validate, val_))
import numpy as np
print('xx_result',np.mean(xx_mse))
s = 0
for i in cv_pred:
    s = s+i
s = s/n_flods
print(stack)
print(s)
df_stack1 = pd.DataFrame(stack)
df_stack2 = pd.DataFrame(s)
df_stack = pd.concat([df_stack1,df_stack2
                ], axis=0)
df_stack.to_csv('feature/tfidf_ling_reg.csv', encoding='utf8', index=None)

######################## par reg #########################
kf = KFold(n_splits=n_flods,shuffle=True,random_state=1017)
kf = kf.split(X)
cv_pred = []
xx_mse = []
stack = np.zeros((len(y),1))
model_1 = PassiveAggressiveRegressor(fit_intercept=True, max_iter=280, tol=0.01,random_state=1017)
for i ,(train_fold,test_fold) in enumerate(kf):
    X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
    model_1.fit(X_train, label_train)
    val_ = model_1.predict(X=X_validate)
    stack[test_fold] = np.array(val_).reshape(len(val_),1)
    print(xx_mse_s(label_validate, val_))
    cv_pred.append(model_1.predict(test))
    xx_mse.append(xx_mse_s(label_validate, val_))
import numpy as np
print('xx_result',np.mean(xx_mse))
s = 0
for i in cv_pred:
    s = s+i
s = s/n_flods
print(stack)
print(s)
df_stack1 = pd.DataFrame(stack)
df_stack2 = pd.DataFrame(s)
df_stack = pd.concat([df_stack1,df_stack2
                ], axis=0)
df_stack.to_csv('feature/tfidf_par_reg.csv', encoding='utf8', index=None)

######################## svr reg #########################
kf = KFold(n_splits=n_flods,shuffle=True,random_state=1017)
kf = kf.split(X)
cv_pred = []
xx_mse = []
stack = np.zeros((len(y),1))
model_1 = LinearSVR(random_state=1017)
for i ,(train_fold,test_fold) in enumerate(kf):
    X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
    model_1.fit(X_train, label_train)
    val_ = model_1.predict(X=X_validate)
    stack[test_fold] = np.array(val_).reshape(len(val_),1)
    print(xx_mse_s(label_validate, val_))
    cv_pred.append(model_1.predict(test))
    xx_mse.append(xx_mse_s(label_validate, val_))
import numpy as np
print('xx_result',np.mean(xx_mse))
s = 0
for i in cv_pred:
    s = s+i
s = s/n_flods
print(stack)
print(s)
df_stack1 = pd.DataFrame(stack)
df_stack2 = pd.DataFrame(s)
df_stack = pd.concat([df_stack1,df_stack2
                ], axis=0)
df_stack.to_csv('feature/tfidf_svr_reg.csv', encoding='utf8', index=None)

