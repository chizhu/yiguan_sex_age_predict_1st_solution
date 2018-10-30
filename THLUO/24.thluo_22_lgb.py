
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime,timedelta  
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import gc



# In[24]:


df_train_w2v = pd.read_csv('thluo_train_best_feat.csv')
df_att_nn_feat_v6 = pd.read_csv('att_nn_feat_v6.csv')
df_att_nn_feat_v6.columns = ['device_id'] + ['att_nn_feat_' + str(i) for i in range(22)]
df_train_w2v = df_train_w2v.merge(df_att_nn_feat_v6, on='device_id', how='left')


# In[ ]:


df_train_w2v.to_csv('thluo_train_best_feat.csv', index=None)


# In[26]:


train = df_train_w2v[df_train_w2v['sex'].notnull()]
test = df_train_w2v[df_train_w2v['sex'].isnull()]

X = train.drop(['sex','age','sex_age','device_id'],axis=1)
Y = train['sex_age']
Y_CAT = pd.Categorical(Y)
Y = pd.Series(Y_CAT.codes)


# In[28]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
seed = 666
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)

sub_list = []

cate_feat = ['device_type','device_brand']

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, Y)):
    train_x, train_y = X.iloc[train_idx], Y.iloc[train_idx]
    valid_x, valid_y = X.iloc[valid_idx], Y.iloc[valid_idx] 
    
    lgb_train=lgb.Dataset(train_x,label=train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        #'learning_rate' : 0.02,
        'learning_rate' : 0.01,
        'max_depth':5,
        'num_leaves' : 2 ** 4,
        'metric': {'multi_logloss'},
        'num_class' : 22,
        'objective' : 'multiclass',
        'random_state' : 2018,
        'bagging_freq' : 5,
        'feature_fraction' : 0.7,
        'bagging_fraction' : 0.7,
        'min_split_gain' : 0.0970905919552776,
        'min_child_weight' : 9.42012323936088,  
    }  
    
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=200, verbose_eval=100)  
    
    sub = pd.DataFrame(gbm.predict(test[X.columns.values],num_iteration=gbm.best_iteration))
    sub_list.append(sub)


# In[29]:


sub = (sub_list[0] + sub_list[1] + sub_list[2] + sub_list[3] + sub_list[4]) / num_folds


# In[31]:


sub.columns=Y_CAT.categories
sub['DeviceID']=test['device_id'].values
sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]


# In[32]:


sub.to_csv('th_22_results_lgb.csv',index=False)

