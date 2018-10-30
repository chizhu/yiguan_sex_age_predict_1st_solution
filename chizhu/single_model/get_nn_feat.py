import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
%matplotlib inline

#add
import gc
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack, vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# from skopt.space import Integer, Categorical, Real, Log10
# from skopt.utils import use_named_args
# from skopt import gp_minimize
from gensim.models import Word2Vec, FastText
import gensim
import re
from config import path
# path = "/dev/shm/chizhu_data/data/"
###这里是原始文件的地址，务必修改这里的路径

test = pd.read_csv(path+'deviceid_test.tsv', sep='\t', names=['device_id'])
train = pd.read_csv(path+'deviceid_train.tsv', sep='\t',
                    names=['device_id', 'sex', 'age'])
brand = pd.read_table(path+'deviceid_brand.tsv',
                      names=['device_id', 'vendor', 'version'])
packtime = pd.read_table(path+'deviceid_package_start_close.tsv',
                         names=['device_id', 'app', 'start', 'close'])
packages = pd.read_csv(path+'deviceid_packages.tsv',
                       sep='\t', names=['device_id', 'apps'])

packtime['period'] = (packtime['close'] - packtime['start'])/1000
packtime['start'] = pd.to_datetime(packtime['start'], unit='ms')
app_use_time = packtime.groupby(['app'])['period'].agg('sum').reset_index()
app_use_top100 = app_use_time.sort_values(
    by='period', ascending=False)[:100]['app']
device_app_use_time = packtime.groupby(['device_id', 'app'])[
    'period'].agg('sum').reset_index()
use_time_top100_statis = device_app_use_time.set_index(
    'app').loc[list(app_use_top100)].reset_index()
top100_statis = use_time_top100_statis.pivot(
    index='device_id', columns='app', values='period').reset_index()

top100_statis = top100_statis.fillna(0)

# 手机品牌预处理
brand['vendor'] = brand['vendor'].astype(
    str).apply(lambda x: x.split(' ')[0].upper())
brand['ph_ver'] = brand['vendor'] + '_' + brand['version']

ph_ver = brand['ph_ver'].value_counts()
ph_ver_cnt = pd.DataFrame(ph_ver).reset_index()
ph_ver_cnt.columns = ['ph_ver', 'ph_ver_cnt']

brand = pd.merge(left=brand, right=ph_ver_cnt, on='ph_ver')

# 针对长尾分布做的一点处理
mask = (brand.ph_ver_cnt < 100)
brand.loc[mask, 'ph_ver'] = 'other'

train = pd.merge(brand[['device_id', 'ph_ver']],
                 train, on='device_id', how='right')
test = pd.merge(brand[['device_id', 'ph_ver']],
                test, on='device_id', how='right')
train['ph_ver'] = train['ph_ver'].astype(str)
test['ph_ver'] = test['ph_ver'].astype(str)

# 将 ph_ver 进行 label encoder
ph_ver_le = preprocessing.LabelEncoder()
train['ph_ver'] = ph_ver_le.fit_transform(train['ph_ver'])
test['ph_ver'] = ph_ver_le.transform(test['ph_ver'])
train['label'] = train['sex'].astype(str) + '-' + train['age'].astype(str)
label_le = preprocessing.LabelEncoder()
train['label'] = label_le.fit_transform(train['label'])

test['sex'] = -1
test['age'] = -1
test['label'] = -1
data = pd.concat([train, test], ignore_index=True)
# data.shape

ph_ver_dummy = pd.get_dummies(data['ph_ver'])
ph_ver_dummy.columns = ['ph_ver_' + str(i)
                        for i in range(ph_ver_dummy.shape[1])]

data = pd.concat([data, ph_ver_dummy], axis=1)

del data['ph_ver']

train = data[data.sex != -1]
test = data[data.sex == -1]
# train.shape, test.shape

# 每个app的总使用次数统计
app_num = packtime['app'].value_counts().reset_index()
app_num.columns = ['app', 'app_num']
packtime = pd.merge(left=packtime, right=app_num, on='app')
# 同样的，针对长尾分布做些处理（尝试过不做处理，或换其他阈值，这个100的阈值最高）
packtime.loc[packtime.app_num < 100, 'app'] = 'other'

# 统计每台设备的app数量
df_app = packtime[['device_id', 'app']]
apps = df_app.drop_duplicates().groupby(['device_id'])[
    'app'].apply(' '.join).reset_index()
apps['app_length'] = apps['app'].apply(lambda x: len(x.split(' ')))

train = pd.merge(train, apps, on='device_id', how='left')
test = pd.merge(test, apps, on='device_id', how='left')

# packtime['period'] = (packtime['close'] - packtime['start'])/1000
# packtime['start'] = pd.to_datetime(packtime['start'], unit='ms')
packtime['dayofweek'] = packtime['start'].dt.dayofweek
packtime['hour'] = packtime['start'].dt.hour
# packtime = packtime[(packtime['start'] < '2017-03-31 23:59:59') & (packtime['start'] > '2017-03-01 00:00:00')]

app_use_time = packtime.groupby(['device_id', 'dayofweek'])[
    'period'].agg('sum').reset_index()
week_app_use = app_use_time.pivot_table(
    values='period', columns='dayofweek', index='device_id').reset_index()
week_app_use = week_app_use.fillna(0)
week_app_use.columns = ['device_id'] + \
    ['week_day_' + str(i) for i in range(0, 7)]

week_app_use['week_max'] = week_app_use.max(axis=1)
week_app_use['week_min'] = week_app_use.min(axis=1)
week_app_use['week_sum'] = week_app_use.sum(axis=1)
week_app_use['week_std'] = week_app_use.std(axis=1)

# '''
# for i in range(0, 7):
#     week_app_use['week_day_' + str(i)] = week_app_use['week_day_' + str(i)] / week_app_use['week_sum']
# '''

user_behavior = pd.read_csv('data/user_behavior.csv')
user_behavior['app_len_max'] = user_behavior['app_len_max'].astype(np.float64)
del user_behavior['app']
train = pd.merge(train, user_behavior, on='device_id', how='left')
test = pd.merge(test, user_behavior, on='device_id', how='left')

train = pd.merge(train, week_app_use, on='device_id', how='left')
test = pd.merge(test, week_app_use, on='device_id', how='left')

top100_statis.columns = ['device_id'] + \
    ['top100_statis_' + str(i) for i in range(0, 100)]
train = pd.merge(train, top100_statis, on='device_id', how='left')
test = pd.merge(test, top100_statis, on='device_id', how='left')

train.to_csv("data/train_statistic_feat.csv", index=False)
test.to_csv("data/test_statistic_feat.csv", index=False)
