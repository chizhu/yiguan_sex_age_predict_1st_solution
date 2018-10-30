
# coding: utf-8

# In[1]:

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

# path="/Users/chizhu/data/competition_data/易观/"


# In[2]:


test = pd.read_csv(path+'deviceid_test.tsv', sep='\t', names=['device_id'])
train = pd.read_csv(path+'deviceid_train.tsv', sep='\t', names=['device_id', 'sex', 'age'])
brand = pd.read_table(path+'deviceid_brand.tsv', names=['device_id', 'vendor', 'version'])
packtime = pd.read_table(path+'deviceid_package_start_close.tsv', 
                         names=['device_id', 'app', 'start', 'close'])
packages = pd.read_csv(path+'deviceid_packages.tsv', sep='\t', names=['device_id', 'apps'])


# In[3]:


def get_str(df):
    res=""
    for i in df.split(","):
        res+=i+" "
    return res
packages["str_app"]=packages['apps'].apply(lambda x:get_str(x),1)


# In[4]:


tfidf = CountVectorizer()
train_str_app=pd.merge(train[['device_id']],packages[["device_id",'str_app']],on="device_id",how="left")
test_str_app=pd.merge(test[['device_id']],packages[["device_id",'str_app']],on="device_id",how="left")
packages['str_app'] = tfidf.fit_transform(packages['str_app'])
train_app = tfidf.transform(list(train_str_app['str_app'])).tocsr()
test_app = tfidf.transform(list(test_str_app['str_app'])).tocsr()


# In[5]:


all_id=pd.concat([train[["device_id"]],test[['device_id']]])


# In[6]:


all_id.index=range(len(all_id))


# In[7]:


# encoding:utf-8
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error
import os
if not os.path.exists("data"):
    os.mkdir("data")



############################ 切分数据集 ##########################
print('开始进行一些前期处理')
train_feature = train_app
test_feature = test_app
    # 五则交叉验证
n_folds = 5
print('处理完毕')
df_stack = pd.DataFrame()
df_stack['device_id']=all_id['device_id']
for label in ["sex"]:
    score = train[label]-1
    
    ########################### lr(LogisticRegression) ################################
    print('lr stacking')
    stack_train = np.zeros((len(train), 1))
    stack_test = np.zeros((len(test), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        clf = LogisticRegression(random_state=1017, C=8)
        clf.fit(train_feature[tr], score[tr])
        score_va = clf.predict_proba(train_feature[va])[:,1]
        
        score_te = clf.predict_proba(test_feature)[:,1]
        print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0]+= score_te
    stack_test /= n_folds
    
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_lr_classfiy_{}'.format(label)] = stack[:, 0]
    

    ########################### SGD(随机梯度下降) ################################
    print('sgd stacking')
    stack_train = np.zeros((len(train), 1))
    stack_test = np.zeros((len(test), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        sgd = SGDClassifier(random_state=1017, loss='log')
        sgd.fit(train_feature[tr], score[tr])
        score_va = sgd.predict_proba(train_feature[va])[:,1]
        score_te = sgd.predict_proba(test_feature)[:,1]
        print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0]+= score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_sgd_classfiy_{}'.format(label)] = stack[:, 0]


    ########################### pac(PassiveAggressiveClassifier) ################################
    print('PAC stacking')
    stack_train = np.zeros((len(train), 1))
    stack_test = np.zeros((len(test), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        pac = PassiveAggressiveClassifier(random_state=1017)
        pac.fit(train_feature[tr], score[tr])
        score_va = pac._predict_proba_lr(train_feature[va])[:,1]
        score_te = pac._predict_proba_lr(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_pac_classfiy_{}'.format(label)] = stack[:, 0]
    


    ########################### ridge(RidgeClassfiy) ################################
    print('RidgeClassfiy stacking')
    stack_train = np.zeros((len(train), 1))
    stack_test = np.zeros((len(test), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        ridge = RidgeClassifier(random_state=1017)
        ridge.fit(train_feature[tr], score[tr])
        score_va = ridge._predict_proba_lr(train_feature[va])[:,1]
        score_te = ridge._predict_proba_lr(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_ridge_classfiy_{}'.format(label)] = stack[:, 0]
    


    ########################### bnb(BernoulliNB) ################################
    print('BernoulliNB stacking')
    stack_train = np.zeros((len(train), 1))
    stack_test = np.zeros((len(test), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        bnb = BernoulliNB()
        bnb.fit(train_feature[tr], score[tr])
        score_va = bnb.predict_proba(train_feature[va])[:,1]
        score_te = bnb.predict_proba(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], bnb.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_bnb_classfiy_{}'.format(label)] = stack[:, 0]
    
    ########################### mnb(MultinomialNB) ################################
    print('MultinomialNB stacking')
    stack_train = np.zeros((len(train), 1))
    stack_test = np.zeros((len(test), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        mnb = MultinomialNB()
        mnb.fit(train_feature[tr], score[tr])
        score_va = mnb.predict_proba(train_feature[va])[:,1]
        score_te = mnb.predict_proba(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], mnb.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_mnb_classfiy_{}'.format(label)] = stack[:, 0]
    

    ############################ Linersvc(LinerSVC) ################################
    print('LinerSVC stacking')
    stack_train = np.zeros((len(train), 1))
    stack_test = np.zeros((len(test), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        lsvc = LinearSVC(random_state=1017)
        lsvc.fit(train_feature[tr], score[tr])
        score_va = lsvc._predict_proba_lr(train_feature[va])[:,1]
        score_te = lsvc._predict_proba_lr(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_lsvc_classfiy_{}'.format(label)] = stack[:, 0]
    
df_stack.to_csv('data/tfidf_classfiy_package.csv', index=None, encoding='utf8')
print('tfidf特征已保存\n')


# In[8]:


packtime['period'] = (packtime['close'] - packtime['start'])/1000
packtime['start'] = pd.to_datetime(packtime['start'], unit='ms')
app_use_time = packtime.groupby(['app'])['period'].agg('sum').reset_index()
app_use_top100 = app_use_time.sort_values(by='period', ascending=False)[:100]['app']
device_app_use_time = packtime.groupby(['device_id', 'app'])['period'].agg('sum').reset_index()
use_time_top100_statis = device_app_use_time.set_index('app').loc[list(app_use_top100)].reset_index()
top100_statis = use_time_top100_statis.pivot(index='device_id', columns='app', values='period').reset_index()


# In[9]:


top100_statis = top100_statis.fillna(0)


# In[10]:


# 手机品牌预处理
brand['vendor'] = brand['vendor'].astype(str).apply(lambda x : x.split(' ')[0].upper())
brand['ph_ver'] = brand['vendor'] + '_' + brand['version']

ph_ver = brand['ph_ver'].value_counts()
ph_ver_cnt = pd.DataFrame(ph_ver).reset_index()
ph_ver_cnt.columns = ['ph_ver', 'ph_ver_cnt']

brand = pd.merge(left=brand, right=ph_ver_cnt,on='ph_ver')


# In[11]:


# 针对长尾分布做的一点处理
mask = (brand.ph_ver_cnt < 100)
brand.loc[mask, 'ph_ver'] = 'other' 

train_data = pd.merge(brand[['device_id', 'ph_ver']], train, on='device_id', how='right')
test_data = pd.merge(brand[['device_id', 'ph_ver']], test, on='device_id', how='right')
train_data['ph_ver'] = train_data['ph_ver'].astype(str)
test_data['ph_ver'] = test_data['ph_ver'].astype(str)

# 将 ph_ver 进行 label encoder
ph_ver_le = preprocessing.LabelEncoder()
train_data['ph_ver'] = ph_ver_le.fit_transform(train_data['ph_ver'])
test_data['ph_ver'] = ph_ver_le.transform(test_data['ph_ver'])
train_data['label'] = train_data['sex'].astype(str) + '-' + train_data['age'].astype(str)
label_le = preprocessing.LabelEncoder()
train_data['label'] = label_le.fit_transform(train_data['label'])


# In[12]:


test_data['sex'] = -1
test_data['age'] = -1
test_data['label'] = -1
data = pd.concat([train_data, test_data], ignore_index=True)
print(data.shape)


# In[13]:


train_data = data[data.sex != -1]
test_data = data[data.sex == -1]
print(train.shape, test.shape)


# In[14]:


# 每个app的总使用次数统计
app_num = packtime['app'].value_counts().reset_index()
app_num.columns = ['app', 'app_num']
packtime = pd.merge(left=packtime, right=app_num, on='app')
# 同样的，针对长尾分布做些处理（尝试过不做处理，或换其他阈值，这个100的阈值最高）
packtime.loc[packtime.app_num < 100, 'app'] = 'other'


# In[15]:


# 统计每台设备的app数量
df_app = packtime[['device_id', 'app']]
apps = df_app.drop_duplicates().groupby(['device_id'])['app'].apply(' '.join).reset_index()
apps['app_length'] = apps['app'].apply(lambda x:len(x.split(' ')))

train_data = pd.merge(train_data, apps, on='device_id', how='left')
test_data = pd.merge(test_data, apps, on='device_id', how='left')


# In[16]:


# 获取每台设备所安装的apps的tfidf
tfidf = CountVectorizer()
apps['app'] = tfidf.fit_transform(apps['app'])

X_tr_app = tfidf.transform(list(train_data['app'])).tocsr()
X_ts_app = tfidf.transform(list(test_data['app'])).tocsr()


# In[17]:


# encoding:utf-8
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error




############################ 切分数据集 ##########################
print('开始进行一些前期处理')
train_feature = X_tr_app
test_feature = X_ts_app
    # 五则交叉验证
n_folds = 5
print('处理完毕')
df_stack = pd.DataFrame()
df_stack['device_id']=data['device_id']
for label in ["sex"]:
    score = train_data[label]-1
    
    ########################### lr(LogisticRegression) ################################
    print('lr stacking')
    stack_train = np.zeros((len(train_data), 1))
    stack_test = np.zeros((len(test_data), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        clf = LogisticRegression(random_state=1017, C=8)
        clf.fit(train_feature[tr], score[tr])
        score_va = clf.predict_proba(train_feature[va])[:,1]
        
        score_te = clf.predict_proba(test_feature)[:,1]
        print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0]+= score_te
    stack_test /= n_folds
    
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['tfidf_lr_classfiy_{}'.format(label)] = stack[:, 0]
    

    ########################### SGD(随机梯度下降) ################################
    print('sgd stacking')
    stack_train = np.zeros((len(train_data), 1))
    stack_test = np.zeros((len(test_data), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        sgd = SGDClassifier(random_state=1017, loss='log')
        sgd.fit(train_feature[tr], score[tr])
        score_va = sgd.predict_proba(train_feature[va])[:,1]
        score_te = sgd.predict_proba(test_feature)[:,1]
        print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0]+= score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['tfidf_sgd_classfiy_{}'.format(label)] = stack[:, 0]


    ########################### pac(PassiveAggressiveClassifier) ################################
    print('PAC stacking')
    stack_train = np.zeros((len(train_data), 1))
    stack_test = np.zeros((len(test_data), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        pac = PassiveAggressiveClassifier(random_state=1017)
        pac.fit(train_feature[tr], score[tr])
        score_va = pac._predict_proba_lr(train_feature[va])[:,1]
        score_te = pac._predict_proba_lr(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['tfidf_pac_classfiy_{}'.format(label)] = stack[:, 0]
    


    ########################### ridge(RidgeClassfiy) ################################
    print('RidgeClassfiy stacking')
    stack_train = np.zeros((len(train_data), 1))
    stack_test = np.zeros((len(test_data), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        ridge = RidgeClassifier(random_state=1017)
        ridge.fit(train_feature[tr], score[tr])
        score_va = ridge._predict_proba_lr(train_feature[va])[:,1]
        score_te = ridge._predict_proba_lr(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['tfidf_ridge_classfiy_{}'.format(label)] = stack[:, 0]
    


    ########################### bnb(BernoulliNB) ################################
    print('BernoulliNB stacking')
    stack_train = np.zeros((len(train_data), 1))
    stack_test = np.zeros((len(test_data), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        bnb = BernoulliNB()
        bnb.fit(train_feature[tr], score[tr])
        score_va = bnb.predict_proba(train_feature[va])[:,1]
        score_te = bnb.predict_proba(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], bnb.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['tfidf_bnb_classfiy_{}'.format(label)] = stack[:, 0]
    
    ########################### mnb(MultinomialNB) ################################
    print('MultinomialNB stacking')
    stack_train = np.zeros((len(train_data), 1))
    stack_test = np.zeros((len(test_data), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        mnb = MultinomialNB()
        mnb.fit(train_feature[tr], score[tr])
        score_va = mnb.predict_proba(train_feature[va])[:,1]
        score_te = mnb.predict_proba(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], mnb.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['tfidf_mnb_classfiy_{}'.format(label)] = stack[:, 0]
    

    ############################ Linersvc(LinerSVC) ################################
    print('LinerSVC stacking')
    stack_train = np.zeros((len(train_data), 1))
    stack_test = np.zeros((len(test_data), 1))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        lsvc = LinearSVC(random_state=1017)
        lsvc.fit(train_feature[tr], score[tr])
        score_va = lsvc._predict_proba_lr(train_feature[va])[:,1]
        score_te = lsvc._predict_proba_lr(test_feature)[:,1]
        print(score_va)
        print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
        stack_train[va,0] += score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['tfidf_lsvc_classfiy_{}'.format(label)] = stack[:, 0]
    
df_stack.to_csv('data/tfidf_classfiy.csv', index=None, encoding='utf8')
print('tfidf特征已保存\n')


# ### 利用word2vec得到每台设备所安装app的embedding表示

# In[18]:


packages['apps'] = packages['apps'].apply(lambda x:x.split(','))
packages['app_length'] = packages['apps'].apply(lambda x:len(x))


# In[19]:


embed_size = 128
fastmodel = Word2Vec(list(packages['apps']), size=embed_size, window=4, min_count=3, negative=2,
                 sg=1, sample=0.002, hs=1, workers=4)  

embedding_fast = pd.DataFrame([fastmodel[word] for word in (fastmodel.wv.vocab)])
embedding_fast['app'] = list(fastmodel.wv.vocab)
embedding_fast.columns= ["fdim_%s" % str(i) for i in range(embed_size)]+["app"]
print(embedding_fast.head())


# In[20]:


id_list = []
for i in range(packages.shape[0]):
    id_list += [list(packages['device_id'])[i]]*packages['app_length'].iloc[i]


app_list = [word for item in packages['apps'] for word in item]

app_vect = pd.DataFrame({'device_id':id_list})        
app_vect['app'] = app_list


# In[21]:


app_vect = app_vect.merge(embedding_fast, on='app', how='left')
app_vect = app_vect.drop('app', axis=1)

seqfeature = app_vect.groupby(['device_id']).agg('mean')
seqfeature.reset_index(inplace=True)


# In[22]:


print(seqfeature.head())


# ### 用户一周七天玩手机的时长情况

# In[23]:


# packtime['period'] = (packtime['close'] - packtime['start'])/1000
# packtime['start'] = pd.to_datetime(packtime['start'], unit='ms')
packtime['dayofweek'] = packtime['start'].dt.dayofweek
packtime['hour'] = packtime['start'].dt.hour
# packtime = packtime[(packtime['start'] < '2017-03-31 23:59:59') & (packtime['start'] > '2017-03-01 00:00:00')]


# In[24]:


app_use_time = packtime.groupby(['device_id', 'dayofweek'])['period'].agg('sum').reset_index()
week_app_use = app_use_time.pivot_table(values='period', columns='dayofweek', index='device_id').reset_index()
week_app_use = week_app_use.fillna(0)
week_app_use.columns = ['device_id'] + ['week_day_' + str(i) for i in range(0, 7)]

week_app_use['week_max'] = week_app_use.max(axis=1)
week_app_use['week_min'] = week_app_use.min(axis=1)
week_app_use['week_sum'] = week_app_use.sum(axis=1)
week_app_use['week_std'] = week_app_use.std(axis=1)



# ### 将各个特征整合到一块

# In[25]:


print(train_data.columns[4:])


# In[26]:


user_behavior = pd.read_csv('data/user_behavior.csv')
user_behavior['app_len_max'] = user_behavior['app_len_max'].astype(np.float64)
del user_behavior['app']
train_data = pd.merge(train_data, user_behavior, on='device_id', how='left')
test_data = pd.merge(test_data, user_behavior, on='device_id', how='left')


# In[27]:


train_data = pd.merge(train_data, seqfeature, on='device_id', how='left')
test_data = pd.merge(test_data, seqfeature, on='device_id', how='left')


# In[28]:


train_data = pd.merge(train_data, week_app_use, on='device_id', how='left')
test_data = pd.merge(test_data, week_app_use, on='device_id', how='left')


# In[29]:


top100_statis.columns = ['device_id'] + ['top100_statis_' + str(i) for i in range(0, 100)]
train_data = pd.merge(train_data, top100_statis, on='device_id', how='left')
test_data = pd.merge(test_data, top100_statis, on='device_id', how='left')


# In[30]:


train_data.to_csv("./data/train_data.csv",index=False)
test_data.to_csv("./data/test_data.csv",index=False)


# In[31]:


tfidf_feat=pd.read_csv("data/tfidf_classfiy.csv")
tf2=pd.read_csv("data/tfidf_classfiy_package.csv")
train_data=pd.read_csv("data/train_data.csv")
test_data=pd.read_csv("data/test_data.csv")
# app_w2v=pd.read_csv("./data/w2v_tfidf.csv")


# In[32]:


train = pd.merge(train_data,tfidf_feat,on="device_id",how="left")
# train = pd.merge(train_data,tf2,on="device_id",how="left")
# train = pd.merge(train_data,app_w2v,on="device_id",how="left")
test = pd.merge(test_data,tfidf_feat,on="device_id",how="left")
# test = pd.merge(test_data,tf2,on="device_id",how="left")
# test = pd.merge(test_data,app_w2v,on="device_id",how="left")


# In[85]:


train_dt = pd.merge(train_data[['device_id','ph_ver']],tfidf_feat,on="device_id",how="left")
train_dt = pd.merge(train_dt,tf2,on="device_id",how="left")
test_dt = pd.merge(test_data[['device_id',"ph_ver"]],tfidf_feat,on="device_id",how="left")
test_dt = pd.merge(test_dt,tf2,on="device_id",how="left")
feat=pd.concat([train_dt,test_dt])
feat.to_csv("data/sex_chizhu_feat.csv",index=False)


# In[33]:


features = [x for x in train.columns if x not in ['device_id', 'sex',"age","label","app"]]
Y = train['sex'] - 1


# ### 开始训练模型

# In[34]:


import lightgbm as lgb
# import xgboost as xgb
from sklearn.metrics import auc, log_loss, roc_auc_score,f1_score,recall_score,precision_score
from sklearn.cross_validation import StratifiedKFold

kf = StratifiedKFold(Y, n_folds=5, shuffle=True, random_state=1024)

params = {
            'boosting_type': 'gbdt',
            'metric': {'binary_logloss',}, 
#             'is_unbalance':'True',
            'learning_rate' : 0.01, 
             'verbose': 0,
            'num_leaves':32 ,
            # 'max_depth':8, 
            # 'max_bin':10, 
            # 'lambda_l2': 1, 
            # 'min_child_weight':50,
            'objective': 'binary', 
            'feature_fraction': 0.4,
            'bagging_fraction':0.7, # 0.9是目前最优的
            'bagging_freq':3,  # 3是目前最优的
#             'min_data': 500,
            'seed': 1024,
            'nthread': 8,
            # 'silent': True,
}
num_round = 3500
early_stopping_rounds = 100


# In[35]:


aus = []
sub1 = np.zeros((len(test), ))
pred_oob1=np.zeros((len(train),))
for i,(train_index,test_index) in enumerate(kf):
  
    tr_x = train[features].reindex(index=train_index, copy=False)
    tr_y = Y[train_index]
    te_x = train[features].reindex(index=test_index, copy=False)
    te_y = Y[test_index]

    d_tr = lgb.Dataset(tr_x, label=tr_y)
    d_te = lgb.Dataset(te_x, label=te_y)
    model = lgb.train(params, d_tr, num_boost_round=num_round, 
                      valid_sets=d_te,verbose_eval=200,
                              early_stopping_rounds=early_stopping_rounds)
    pred= model.predict(te_x, num_iteration=model.best_iteration)
    pred_oob1[test_index] =pred
    
    a = log_loss(te_y, pred)

    sub1 += model.predict(test[features], num_iteration=model.best_iteration)/5

    print ("idx: ", i) 
    print (" loss: %.5f" % a)

    print ("best tree num: ", model.best_iteration)
    aus.append(a)

print ("mean")
print ("auc:       %s" % (sum(aus) / 5.0))


# In[36]:


#####特征重要性
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
# f=dict(zip(list(train[features].keys()),model.feature_importance()))
# f=sorted(f.items(),key=lambda d:d[1], reverse = True)
# f=pd.DataFrame(f,columns=['feature','imp'])
# plt.bar(range(len(f)),f.imp)
# plt.xticks(range(len(f)),f.feature,rotation=70,fontsize=20)
# fig = plt.gcf()
# fig.set_size_inches(50, 20)


# In[37]:


# f.ix[:450,:]


# In[38]:


# features=f.ix[:434,"feature"].values


# In[39]:


pred_oob1 = pd.DataFrame(pred_oob1, columns=['sex2'])
sub1 = pd.DataFrame(sub1, columns=['sex2'])
res1=pd.concat([pred_oob1,sub1])
res1['sex1'] = 1-res1['sex2']


# In[40]:


import gc
gc.collect()


# In[41]:


train_id = pd.read_csv(path+'deviceid_train.tsv', sep='\t', names=['device_id', 'sex', 'age'])


# In[42]:


# encoding:utf-8
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error




############################ 切分数据集 ##########################
print('开始进行一些前期处理')
train_feature = train_app
test_feature = test_app
    # 五则交叉验证
n_folds = 5
print('处理完毕')
df_stack = pd.DataFrame()
df_stack['device_id']=all_id['device_id']
for label in ["age"]:
    score = train_id[label]
    
    ########################### lr(LogisticRegression) ################################
    print('lr stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        clf = LogisticRegression(random_state=1017, C=8)
        clf.fit(train_feature[tr], score[tr])
        score_va = clf.predict_proba(train_feature[va])
        
        score_te = clf.predict_proba(test_feature)
        print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
        stack_train[va] = score_va
        stack_test+= score_te
    stack_test /= n_folds
    
    stack = np.vstack([stack_train, stack_test])
    
    for i in range(stack.shape[1]):
        df_stack['pack_tfidf_lr_classfiy_{}'.format(i)] = stack[:, i]
    

    ########################### SGD(随机梯度下降) ################################
    print('sgd stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        sgd = SGDClassifier(random_state=1017, loss='log')
        sgd.fit(train_feature[tr], score[tr])
        score_va = sgd.predict_proba(train_feature[va])
        score_te = sgd.predict_proba(test_feature)
        print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
        stack_train[va] = score_va
        stack_test+= score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    for i in range(stack.shape[1]):
        df_stack['pack_tfidf_sgd_classfiy_{}'.format(i)] = stack[:, i]


    ########################### pac(PassiveAggressiveClassifier) ################################
    print('PAC stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['pack_tfidf_pac_classfiy_{}'.format(i)] = stack[:, i]
    


    ########################### ridge(RidgeClassfiy) ################################
    print('RidgeClassfiy stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['pack_tfidf_ridge_classfiy_{}'.format(i)] = stack[:, i]
    


    ########################### bnb(BernoulliNB) ################################
    print('BernoulliNB stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['pack_tfidf_bnb_classfiy_{}'.format(i)] = stack[:, i]
    
    ########################### mnb(MultinomialNB) ################################
    print('MultinomialNB stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['pack_tfidf_mnb_classfiy_{}'.format(i)] = stack[:, i]
    

    ############################ Linersvc(LinerSVC) ################################
    print('LinerSVC stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['pack_tfidf_lsvc_classfiy_{}'.format(i)] = stack[:, i]
    
df_stack.to_csv('data/pack_tfidf_age.csv', index=None, encoding='utf8')
print('tfidf特征已保存\n')


# #### tfidf

# In[43]:


# encoding:utf-8
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error




############################ 切分数据集 ##########################
print('开始进行一些前期处理')
train_feature = X_tr_app
test_feature = X_ts_app
    # 五则交叉验证
n_folds = 5
print('处理完毕')
df_stack = pd.DataFrame()
df_stack['device_id']=data['device_id']
for label in ["age"]:
    score = train[label]
    
    ########################### lr(LogisticRegression) ################################
    print('lr stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        clf = LogisticRegression(random_state=1017, C=8)
        clf.fit(train_feature[tr], score[tr])
        score_va = clf.predict_proba(train_feature[va])
        
        score_te = clf.predict_proba(test_feature)
        print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
        stack_train[va] = score_va
        stack_test+= score_te
    stack_test /= n_folds
    
    stack = np.vstack([stack_train, stack_test])
    
    for i in range(stack.shape[1]):
        df_stack['tfidf_lr_classfiy_{}'.format(i)] = stack[:, i]
    

    ########################### SGD(随机梯度下降) ################################
    print('sgd stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
    score_va = 0

    for i, (tr, va) in enumerate(StratifiedKFold(score, n_folds=n_folds, random_state=1017)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        sgd = SGDClassifier(random_state=1017, loss='log')
        sgd.fit(train_feature[tr], score[tr])
        score_va = sgd.predict_proba(train_feature[va])
        score_te = sgd.predict_proba(test_feature)
        print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
        stack_train[va] = score_va
        stack_test+= score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    for i in range(stack.shape[1]):
        df_stack['tfidf_sgd_classfiy_{}'.format(i)] = stack[:, i]


    ########################### pac(PassiveAggressiveClassifier) ################################
    print('PAC stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['tfidf_pac_classfiy_{}'.format(i)] = stack[:, i]
    


    ########################### ridge(RidgeClassfiy) ################################
    print('RidgeClassfiy stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['tfidf_ridge_classfiy_{}'.format(i)] = stack[:, i]
    


    ########################### bnb(BernoulliNB) ################################
    print('BernoulliNB stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['tfidf_bnb_classfiy_{}'.format(i)] = stack[:, i]
    
    ########################### mnb(MultinomialNB) ################################
    print('MultinomialNB stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['tfidf_mnb_classfiy_{}'.format(i)] = stack[:, i]
    

    ############################ Linersvc(LinerSVC) ################################
    print('LinerSVC stacking')
    stack_train = np.zeros((len(train), 11))
    stack_test = np.zeros((len(test), 11))
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
    
    for i in range(stack.shape[1]):
        df_stack['data/tfidf_lsvc_classfiy_{}'.format(i)] = stack[:, i]
    
df_stack.to_csv('data/tfidf_age.csv', index=None, encoding='utf8')
print('tfidf特征已保存\n')


# In[44]:


tfidf_feat=pd.read_csv("data/tfidf_age.csv")
tf2=pd.read_csv("data/pack_tfidf_age.csv")
train_data=pd.read_csv("data/train_data.csv")
test_data=pd.read_csv("data/test_data.csv")


# In[41]:


train_dt = pd.merge(train_data[['device_id','ph_ver']],tfidf_feat,on="device_id",how="left")
train_dt = pd.merge(train_dt,tf2,on="device_id",how="left")
test_dt = pd.merge(test_data[['device_id',"ph_ver"]],tfidf_feat,on="device_id",how="left")
test_dt = pd.merge(test_dt,tf2,on="device_id",how="left")
feat=pd.concat([train_dt,test_dt])
feat.to_csv("data/age_chizhu_feat.csv",index=False)


# In[40]:





# In[45]:


tfidf_feat=pd.read_csv("data/tfidf_age.csv")
tf2=pd.read_csv("data/pack_tfidf_age.csv")
train_data=pd.read_csv("data/train_data.csv")
test_data=pd.read_csv("data/test_data.csv")
train = pd.merge(train_data,tfidf_feat,on="device_id",how="left")
# train = pd.merge(train_data,tf2,on="device_id",how="left")
# train = pd.merge(train_data,app_w2v,on="device_id",how="left")
test = pd.merge(test_data,tfidf_feat,on="device_id",how="left")
# test = pd.merge(test_data,tf2,on="device_id",how="left")
# test = pd.merge(test_data,app_w2v,on="device_id",how="left")
features = [x for x in train.columns if x not in ['device_id',"age","sex","label","app"]]
Y = train['age'] 


# In[46]:


import lightgbm as lgb
# import xgboost as xgb
from sklearn.metrics import auc, log_loss, roc_auc_score,f1_score,recall_score,precision_score
from sklearn.cross_validation import StratifiedKFold

kf = StratifiedKFold(Y, n_folds=5, shuffle=True, random_state=1024)

params = {
            'boosting_type': 'gbdt',
            'metric': {'multi_logloss',}, 
#             'is_unbalance':'True',
            'learning_rate' : 0.01, 
             'verbose': 0,
            'num_leaves':32 ,
            # 'max_depth':8, 
            # 'max_bin':10, 
            # 'lambda_l2': 1, 
            # 'min_child_weight':50,
            "num_class":11,
            'objective': 'multiclass', 
            'feature_fraction': 0.4,
            'bagging_fraction':0.7, # 0.9是目前最优的
            'bagging_freq':3,  # 3是目前最优的
#             'min_data': 500,
            'seed': 1024,
            'nthread': 8,
            # 'silent': True,
}
num_round = 3500
early_stopping_rounds = 100


# In[47]:


aus = []
sub2 = np.zeros((len(test),11 ))
pred_oob2=np.zeros((len(train),11))
for i,(train_index,test_index) in enumerate(kf):
  
    tr_x = train[features].reindex(index=train_index, copy=False)
    tr_y = Y[train_index]
    te_x = train[features].reindex(index=test_index, copy=False)
    te_y = Y[test_index]

    d_tr = lgb.Dataset(tr_x, label=tr_y)
    d_te = lgb.Dataset(te_x, label=te_y)
    model = lgb.train(params, d_tr, num_boost_round=num_round, 
                      valid_sets=d_te,verbose_eval=200,
                              early_stopping_rounds=early_stopping_rounds)
    pred= model.predict(te_x, num_iteration=model.best_iteration)
    pred_oob2[test_index] =pred
    
    a = log_loss(te_y, pred)

    sub2 += model.predict(test[features], num_iteration=model.best_iteration)/5

    print ("idx: ", i) 
    print (" loss: %.5f" % a)

    print ("best tree num: ", model.best_iteration)
    aus.append(a)

print ("mean")
print ("loss:       %s" % (sum(aus) / 5.0))


# In[55]:


#####特征重要性

# import matplotlib.pyplot as plt
# f=dict(zip(list(train[features].keys()),model.feature_importance()))
# f=sorted(f.items(),key=lambda d:d[1], reverse = True)
# f=pd.DataFrame(f,columns=['feature','imp'])
# plt.bar(range(len(f)),f.imp)
# plt.xticks(range(len(f)),f.feature,rotation=70,fontsize=20)
# fig = plt.gcf()
# fig.set_size_inches(50, 20)


# In[56]:


# f.ix[:650,:]


# In[57]:


# features=f.ix[:641,"feature"].values


# In[58]:


res2_1=np.vstack((pred_oob2,sub2))
res2_1 = pd.DataFrame(res2_1)


# In[59]:


if not os.path.exists("submit"):
    os.mkdir("submit")
res1.index=range(len(res1))
res2_1.index=range(len(res2_1))
final_1=res2_1.copy()
final_2=res2_1.copy()
for i in range(11):
    final_1[i]=res1['sex1']*res2_1[i]
    final_2[i]=res1['sex2']*res2_1[i]
id_list=pd.concat([train[['device_id']],test[['device_id']]])
final=id_list
final.index=range(len(final))
final.columns= ['DeviceID']
final_pred = pd.concat([final_1,final_2],1)
final=pd.concat([final,final_pred],1)
final.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

final.to_csv('feature/lgb_feat_chizhu.csv', index=False)


# In[60]:


test['DeviceID']=test['device_id']
sub=pd.merge(test[['DeviceID']],final,on="DeviceID",how="left")
sub.to_csv("submit/lgb_chizhu.csv",index=False)


# In[61]:


# sub.sum(1)

