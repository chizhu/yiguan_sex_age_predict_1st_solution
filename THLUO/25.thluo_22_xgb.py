
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
import xgboost as xgb
from datetime import datetime,timedelta  
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import gc
from feat_util import *


# In[2]:

print ('25.thluo_22_xgb.py')
path='input/'
data=pd.DataFrame()
#sex_age=pd.read_excel('./data/性别年龄对照表.xlsx')


# In[3]:


deviceid_packages=pd.read_csv(path+'deviceid_packages.tsv',sep='\t',names=['device_id','apps'])
deviceid_test=pd.read_csv(path+'deviceid_test.tsv',sep='\t',names=['device_id'])
deviceid_train=pd.read_csv(path+'deviceid_train.tsv',sep='\t',names=['device_id','sex','age'])


# In[4]:


df_train = pd.concat([deviceid_train, deviceid_test])


# In[5]:


df_train


# In[6]:


df_sex_prob_oof = pd.read_csv('device_sex_prob_oof.csv')
df_age_prob_oof = pd.read_csv('device_age_prob_oof.csv')
df_start_close_sex_prob_oof = pd.read_csv('start_close_sex_prob_oof.csv')
#后面两个，线上线下不对应，线下过拟合了
df_start_close_age_prob_oof = pd.read_csv('start_close_age_prob_oof.csv')
df_tfidf_lr_sex_age_prob_oof = pd.read_csv('tfidf_lr_sex_age_prob_oof.csv')
#之前的有用的
df_sex_age_bin_prob_oof = pd.read_csv('sex_age_bin_prob_oof.csv')

df_age_bin_prob_oof = pd.read_csv('age_bin_prob_oof.csv')
df_hcc_device_brand_age_sex = pd.read_csv('hcc_device_brand_age_sex.csv')
df_device_age_regression_prob_oof = pd.read_csv('device_age_regression_prob_oof.csv')
df_device_start_GRU_pred = pd.read_csv('device_start_GRU_pred.csv')
df_device_start_GRU_pred_age = pd.read_csv('device_start_GRU_pred_age.csv')
df_device_all_GRU_pred = pd.read_csv('device_all_GRU_pred.csv')
df_lgb_sex_age_prob_oof = pd.read_csv('lgb_sex_age_prob_oof.csv')
df_device_start_capsule_pred = pd.read_csv('device_start_capsule_pred.csv')
df_device_start_textcnn_pred = pd.read_csv('device_start_textcnn_pred.csv')
df_device_start_text_dpcnn_pred = pd.read_csv('device_start_text_dpcnn_pred.csv')
df_device_start_lstm_pred = pd.read_csv('device_start_lstm_pred.csv')
df_att_nn_feat_v6 = pd.read_csv('att_nn_feat_v6.csv')
df_att_nn_feat_v6.columns = ['device_id'] + ['att_nn_feat_' + str(i) for i in range(22)]

#过拟合特征
del df_start_close_age_prob_oof['device_app_groupedstart_close_age_prob_oof_4_MEAN']
del df_start_close_sex_prob_oof['device_app_groupedstart_close_sex_prob_oof_MIN']
del df_start_close_sex_prob_oof['device_app_groupedstart_close_sex_prob_oof_MAX']


# In[7]:


df_train_w2v = df_train.merge(df_sex_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_age_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_start_close_sex_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_start_close_age_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_sex_age_bin_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_age_bin_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_hcc_device_brand_age_sex, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_age_regression_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_start_GRU_pred, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_start_GRU_pred_age, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_all_GRU_pred, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_lgb_sex_age_prob_oof, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_start_capsule_pred, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_start_textcnn_pred, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_start_text_dpcnn_pred, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_start_lstm_pred, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_att_nn_feat_v6, on='device_id', how='left')


# In[9]:


df_train_w2v['sex'] = df_train_w2v['sex'].apply(lambda x:str(x))
df_train_w2v['age'] = df_train_w2v['age'].apply(lambda x:str(x))
def tool(x):
    if x=='nan':
        return x
    else:
        return str(int(float(x)))
df_train_w2v['sex']=df_train_w2v['sex'].apply(tool)
df_train_w2v['age']=df_train_w2v['age'].apply(tool)
df_train_w2v['sex_age']=df_train_w2v['sex']+'-'+df_train_w2v['age']
df_train_w2v = df_train_w2v.replace({'nan':np.NaN,'nan-nan':np.NaN})


# In[11]:


train = df_train_w2v[df_train_w2v['sex'].notnull()]
test = df_train_w2v[df_train_w2v['sex'].isnull()]

X = train.drop(['sex','age','sex_age','device_id'],axis=1)
Y = train['sex_age']
Y_CAT = pd.Categorical(Y)
Y = pd.Series(Y_CAT.codes)


# In[14]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
#seed = 2048
seed = 666
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)

sub_list = []

cate_feat = ['device_type','device_brand']

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, Y)):
    train_x, train_y = X.iloc[train_idx], Y.iloc[train_idx]
    valid_x, valid_y = X.iloc[valid_idx], Y.iloc[valid_idx] 
    
    xg_train = xgb.DMatrix(train_x, label=train_y)
    xg_val = xgb.DMatrix(valid_x, label=valid_y)    

    param = {
        'objective' : 'multi:softprob',
        'eta' : 0.03,
        'max_depth' : 3, 
        'num_class' : 22,
        'eval_metric' : 'mlogloss',
        'min_child_weight' : 3,
        'subsample' : 0.7,
        'colsample_bytree' : 0.7,
        'seed' : 2006,
        'nthread' : 5
    } 
    
    num_rounds = 1000

    watchlist = [ (xg_train,'train'), (xg_val, 'val') ]
    model = xgb.train(param, xg_train, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=50)    
    
    test_matrix = xgb.DMatrix(test[X.columns.values])
    sub = pd.DataFrame(model.predict(test_matrix))
    sub_list.append(sub)


# In[15]:


sub = (sub_list[0] + sub_list[1] + sub_list[2] + sub_list[3] + sub_list[4]) / num_folds
sub


# In[16]:


sub.columns=Y_CAT.categories
sub['DeviceID']=test['device_id'].values
sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
sub.to_csv('th_22_results_xgb.csv',index=False)

