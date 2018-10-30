
# coding: utf-8

# In[ ]:


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
# %matplotlib inline

#add
import gc
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack, vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from skopt.space import Integer, Categorical, Real, Log10
from skopt.utils import use_named_args
from skopt import gp_minimize
from gensim.models import Word2Vec, FastText
import gensim 
import re


# In[ ]:


test = pd.read_csv('../input/yiguan/demo/Demo/deviceid_test.tsv', sep='\t', names=['device_id'])
train = pd.read_csv('../input/yiguan/demo/Demo/deviceid_train.tsv', sep='\t', names=['device_id', 'sex', 'age'])
brand = pd.read_table('../input/yiguan/demo/Demo/deviceid_brand.tsv', names=['device_id', 'vendor', 'version'])
packtime = pd.read_table('../input/yiguan/demo/Demo/deviceid_package_start_close.tsv', 
                         names=['device_id', 'app', 'start', 'close'])
packages = pd.read_csv('../input/yiguan/demo/Demo/deviceid_packages.tsv', sep='\t', names=['device_id', 'apps'])


# In[ ]:


packtime['period'] = (packtime['close'] - packtime['start'])/1000
packtime['start'] = pd.to_datetime(packtime['start'], unit='ms')
app_use_time = packtime.groupby(['app'])['period'].agg('sum').reset_index()
# 试试看200
app_use_top100 = app_use_time.sort_values(by='period', ascending=False)[:100]['app']
device_app_use_time = packtime.groupby(['device_id', 'app'])['period'].agg('sum').reset_index()
use_time_top100_statis = device_app_use_time.set_index('app').loc[list(app_use_top100)].reset_index()
top100_statis = use_time_top100_statis.pivot(index='device_id', columns='app', values='period').reset_index()


# In[ ]:


top100_statis = top100_statis.fillna(0)


# In[ ]:


# 手机品牌预处理
brand['vendor'] = brand['vendor'].astype(str).apply(lambda x : x.split(' ')[0].upper())
brand['ph_ver'] = brand['vendor'] + '_' + brand['version']

ph_ver = brand['ph_ver'].value_counts()
ph_ver_cnt = pd.DataFrame(ph_ver).reset_index()
ph_ver_cnt.columns = ['ph_ver', 'ph_ver_cnt']

brand = pd.merge(left=brand, right=ph_ver_cnt,on='ph_ver')


# In[ ]:


# 针对长尾分布做的一点处理
mask = (brand.ph_ver_cnt < 100)
brand.loc[mask, 'ph_ver'] = 'other' 

train = pd.merge(brand[['device_id', 'ph_ver']], train, on='device_id', how='right')
test = pd.merge(brand[['device_id', 'ph_ver']], test, on='device_id', how='right')
train['ph_ver'] = train['ph_ver'].astype(str)
test['ph_ver'] = test['ph_ver'].astype(str)

# 将 ph_ver 进行 label encoder
ph_ver_le = preprocessing.LabelEncoder()
train['ph_ver'] = ph_ver_le.fit_transform(train['ph_ver'])
test['ph_ver'] = ph_ver_le.transform(test['ph_ver'])
train['label'] = train['sex'].astype(str) + '-' + train['age'].astype(str)
label_le = preprocessing.LabelEncoder()
train['label'] = label_le.fit_transform(train['label'])


# In[ ]:


test['sex'] = -1
test['age'] = -1
test['label'] = -1
data = pd.concat([train, test], ignore_index=True)
data.shape


# In[ ]:


ph_ver_dummy = pd.get_dummies(data['ph_ver'])
ph_ver_dummy.columns = ['ph_ver_' + str(i) for i in range(ph_ver_dummy.shape[1])]


# In[ ]:


data = pd.concat([data, ph_ver_dummy], axis=1)


# In[ ]:


del data['ph_ver']


# In[ ]:


train = data[data.sex != -1]
test = data[data.sex == -1]
train.shape, test.shape


# In[ ]:


# 每个app的总使用次数统计
app_num = packtime['app'].value_counts().reset_index()
app_num.columns = ['app', 'app_num']
packtime = pd.merge(left=packtime, right=app_num, on='app')
# 同样的，针对长尾分布做些处理（尝试过不做处理，或换其他阈值，这个100的阈值最高）
packtime.loc[packtime.app_num < 100, 'app'] = 'other'


# In[ ]:


# 统计每台设备的app数量
df_app = packtime[['device_id', 'app']]
apps = df_app.drop_duplicates().groupby(['device_id'])['app'].apply(' '.join).reset_index()
apps['app_length'] = apps['app'].apply(lambda x:len(x.split(' ')))

train = pd.merge(train, apps, on='device_id', how='left')
test = pd.merge(test, apps, on='device_id', how='left')


# In[ ]:


# 获取每台设备所安装的apps的tfidf
tfidf = CountVectorizer(lowercase=False, min_df=3, stop_words=top100_statis.columns.tolist()[1:7])
apps['app'] = tfidf.fit_transform(apps['app'])

X_tr_app = tfidf.transform(list(train['app']))
X_ts_app = tfidf.transform(list(test['app']))


# In[ ]:


'''
svd = TruncatedSVD(n_components=100, random_state=42)
X = vstack([X_tr_app, X_ts_app])
svd.fit(X)
X_tr_app = svd.fit_transform(X_tr_app)
X_ts_app = svd.fit_transform(X_ts_app)
X_tr_app = pd.DataFrame(X_tr_app)
X_ts_app = pd.DataFrame(X_ts_app)
X_tr_app.columns = ['app_' + str(i) for i in range(0, 100)]
X_ts_app.columns = ['app_' + str(i) for i in range(0, 100)]
'''


# ### 利用word2vec得到每台设备所安装app的embedding表示

# In[ ]:


packages['apps'] = packages['apps'].apply(lambda x:x.split(','))
packages['app_length'] = packages['apps'].apply(lambda x:len(x))


# In[ ]:


embed_size = 128
fastmodel = Word2Vec(list(packages['apps']), size=embed_size, window=4, min_count=3, negative=2,
                 sg=1, sample=0.002, hs=1, workers=4)  

embedding_fast = pd.DataFrame([fastmodel[word] for word in (fastmodel.wv.vocab)])
embedding_fast['app'] = list(fastmodel.wv.vocab)
embedding_fast.columns= ["fdim_%s" % str(i) for i in range(embed_size)]+["app"]
embedding_fast.head()


# In[ ]:


id_list = []
for i in range(packages.shape[0]):
    id_list += [list(packages['device_id'])[i]]*packages['app_length'].iloc[i]


app_list = [word for item in packages['apps'] for word in item]

app_vect = pd.DataFrame({'device_id':id_list})        
app_vect['app'] = app_list


# In[ ]:


app_vect = app_vect.merge(embedding_fast, on='app', how='left')
app_vect = app_vect.drop('app', axis=1)

seqfeature = app_vect.groupby(['device_id']).agg('mean')
seqfeature.reset_index(inplace=True)


# In[ ]:


seqfeature.head()


# ### 用户一周七天玩手机的时长情况

# In[ ]:


# packtime['period'] = (packtime['close'] - packtime['start'])/1000
# packtime['start'] = pd.to_datetime(packtime['start'], unit='ms')
packtime['dayofweek'] = packtime['start'].dt.dayofweek
packtime['hour'] = packtime['start'].dt.hour
# packtime = packtime[(packtime['start'] < '2017-03-31 23:59:59') & (packtime['start'] > '2017-03-01 00:00:00')]


# In[ ]:


app_use_time = packtime.groupby(['device_id', 'dayofweek'])['period'].agg('sum').reset_index()
week_app_use = app_use_time.pivot_table(values='period', columns='dayofweek', index='device_id').reset_index()
week_app_use = week_app_use.fillna(0)
week_app_use.columns = ['device_id'] + ['week_day_' + str(i) for i in range(0, 7)]

week_app_use['week_max'] = week_app_use.max(axis=1)
week_app_use['week_min'] = week_app_use.min(axis=1)
week_app_use['week_sum'] = week_app_use.sum(axis=1)
week_app_use['week_std'] = week_app_use.std(axis=1)

'''
for i in range(0, 7):
    week_app_use['week_day_' + str(i)] = week_app_use['week_day_' + str(i)] / week_app_use['week_sum']
'''


# In[ ]:


'''
app_use_time = packtime.groupby(['device_id', 'hour'])['period'].agg('sum').reset_index()
hour_app_use = app_use_time.pivot_table(values='period', columns='hour', index='device_id').reset_index()
hour_app_use = hour_app_use.fillna(0)
hour_app_use.columns = ['device_id'] + ['hour_' + str(i) for i in range(0, 24)]

# hour_app_use['hour_max'] = hour_app_use.max(axis=1)
# hour_app_use['hour_min'] = hour_app_use.min(axis=1)
# hour_app_use['hour_sum'] = hour_app_use.sum(axis=1)
# hour_app_use['hour_std'] = hour_app_use.std(axis=1)

# for i in range(0, 24):
#     hour_app_use['hour_' + str(i)] = hour_app_use['hour_' + str(i)] / hour_app_use['hour_sum']
'''


# ### 将各个特征整合到一块

# In[ ]:


train.columns[4:]


# In[ ]:


user_behavior = pd.read_csv('../input/yg-user-behavior/user_behavior.csv')
user_behavior['app_len_max'] = user_behavior['app_len_max'].astype(np.float64)
del user_behavior['app']
train = pd.merge(train, user_behavior, on='device_id', how='left')
test = pd.merge(test, user_behavior, on='device_id', how='left')


# In[ ]:


train = pd.merge(train, seqfeature, on='device_id', how='left')
test = pd.merge(test, seqfeature, on='device_id', how='left')


# In[ ]:


train = pd.merge(train, week_app_use, on='device_id', how='left')
test = pd.merge(test, week_app_use, on='device_id', how='left')


# In[ ]:


'''
app_top50_list = list(packtime.groupby(by='app')['period'].sum().sort_values(ascending=False)[:50].index)

for app in app_top50_list:
    app_cnt = packtime[packtime['app'] == app]
    start_num_app = app_cnt.groupby(by='device_id')['start'].count().reset_index()
    start_num_app.columns = ['device_id', 'start_num_app_' + app[0:4]]
    train = train.merge(start_num_app, on='device_id', how='left')
    test = test.merge(start_num_app, on='device_id', how='left')
    print(app + ' done')   
'''


# In[ ]:


'''
# all_top50 : 使用总时长最高的50款app，每个人的使用时间统计
all_top50 = pd.read_csv('../input/yg-feature/all_top50_statis.csv')
train = pd.merge(train, all_top50, on='device_id', how='left')
test = pd.merge(test, all_top50, on='device_id', how='left')
'''


# In[ ]:


top100_statis.columns = ['device_id'] + ['top100_statis_' + str(i) for i in range(0, 100)]
train = pd.merge(train, top100_statis, on='device_id', how='left')
test = pd.merge(test, top100_statis, on='device_id', how='left')


# In[ ]:


train.to_csv('train_feature.csv', index=None)
test.to_csv('test_feature.csv', index=None)


# In[ ]:


feats = train.columns[4:]
feats


# In[ ]:


feats = feats.delete(153)
feats[153]


# In[ ]:


'''
train = pd.merge(train, hour_app_use, on='device_id', how='left')
test = pd.merge(test, hour_app_use, on='device_id', how='left')
'''


# In[ ]:


X_train = hstack([X_tr_app, train[feats].astype(float)])
X_test = hstack([X_ts_app, test[feats].astype(float)])

X_train = X_train.tocsr().astype('float')
X_test = X_test.tocsr().astype('float')


# ### 开始训练模型

# In[ ]:


Y = train['sex'] - 1
kfold = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
oof_preds1 = np.zeros((X_train.shape[0], ))
sub1 = np.zeros((X_test.shape[0], ))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)): 
    X_tr, X_vl, y_tr, y_vl = X_train[train_index], X_train[test_index],                                 Y[train_index], Y[test_index]
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


# In[ ]:


Y = train['age']
kfold = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
oof_preds2 = np.zeros((X_train.shape[0], 11))
sub2 = np.zeros((X_test.shape[0], 11))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)):
    X_tr, X_vl, y_tr, y_vl = X_train[train_index], X_train[test_index],                                 Y[train_index], Y[test_index]
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

    oof_preds2[test_index] = model.predict(X_vl, num_iteration=model.best_iteration)
    sub2 += model.predict(X_test, num_iteration=model.best_iteration)/kfold.n_splits


# In[ ]:


oof_preds1 = pd.DataFrame(oof_preds1, columns=['sex2'])

oof_preds1['sex1'] = 1-oof_preds1['sex2']
oof_preds2 = pd.DataFrame(oof_preds2, columns=['age%s'%i for i in range(11)])
oof_preds = train[['device_id']]
oof_preds.columns = ['DeviceID']

for i in ['sex1', 'sex2']:
    for j in ['age%s'%i for i in range(11)]:
        oof_preds[i+'_'+j] = oof_preds1[i] * oof_preds2[j]
oof_preds.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

oof_preds.to_csv('train.csv', index=False)


# In[ ]:


sub1 = pd.DataFrame(sub1, columns=['sex2'])

sub1['sex1'] = 1-sub1['sex2']
sub2 = pd.DataFrame(sub2, columns=['age%s'%i for i in range(11)])
sub = test[['device_id']]
sub.columns = ['DeviceID']

for i in ['sex1', 'sex2']:
    for j in ['age%s'%i for i in range(11)]:
        sub[i+'_'+j] = sub1[i] * sub2[j]
sub.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

sub.to_csv('lgb_l_v54.csv', index=False)


# In[ ]:


'''
Y = train['label']
#best params: [31, 11, 0.015955854914003094, 0.12122664084283229, 0.7645440142264772, 24, 1048, 0.00552258737237652, 0.005810068328090833, 7]
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
sub = np.zeros((X_test.shape[0], 22))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)):
    X_tr, X_vl, y_tr, y_vl = X_train[train_index], X_train[test_index], Y[train_index], Y[test_index]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    params = {
        'boosting_type': 'gbdt',
        'max_depth':7,
        'objective':'multiclass',
        'metric': {'multi_logloss'},
        'num_class':22,
        'num_leaves':20,
        'subsample': 0.86,
        'colsample_bytree': 0.8,
        #'lambda_l1':0.00007995302080034896,
        'lambda_l2':0.005,
        'subsample_freq':11,
        'learning_rate': 0.01,
        'min_child_weight':5.5,

    }
    
    model = lgb.train(params,
                        dtrain,
                        num_boost_round=6000,
                        valid_sets=dvalid,
                        early_stopping_rounds=20,
                        verbose_eval=100)


    sub += model.predict(X_test, num_iteration=model.best_iteration)/kfold.n_splits
'''


# In[ ]:


'''
sub = pd.DataFrame(sub)
cols = [x for x in range(0, 22)]
cols = label_le.inverse_transform(cols)

sub.columns = cols
sub['DeviceID'] = test['device_id'].values

sub = sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

sub.to_csv('30.csv', index=False)
'''

