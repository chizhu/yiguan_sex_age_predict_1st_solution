
# coding: utf-8

# In[1]:


# coding: utf-8

# In[1]:

from sklearn.metrics import log_loss
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime,timedelta  
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
get_ipython().run_line_magic('matplotlib', 'inline')

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
import os
import xgboost as xgb
path="./"
os.listdir(path)


# In[2]:
print ('27.thluo_nb_xgb.py')

train_id=pd.read_csv("input/deviceid_train.tsv",sep="\t",names=['device_id','sex','age'])
test_id=pd.read_csv("input/deviceid_test.tsv",sep="\t",names=['device_id'])

all_id=pd.concat([train_id[['device_id']],test_id[['device_id']]])
df_sex_prob_oof = pd.read_csv('device_sex_prob_oof.csv')
df_age_prob_oof = pd.read_csv('device_age_prob_oof.csv')
df_start_close_sex_prob_oof = pd.read_csv('start_close_sex_prob_oof.csv')
#后面两个，线上线下不对应，线下过拟合了
df_start_close_age_prob_oof = pd.read_csv('start_close_age_prob_oof.csv')
#df_start_close_sex_age_prob_oof = pd.read_csv('start_close_sex_age_prob_oof.csv')
df_tfidf_lr_sex_age_prob_oof = pd.read_csv('tfidf_lr_sex_age_prob_oof.csv')
#之前的有用的
df_sex_age_bin_prob_oof = pd.read_csv('sex_age_bin_prob_oof.csv')

df_age_bin_prob_oof = pd.read_csv('age_bin_prob_oof.csv')
df_hcc_device_brand_age_sex = pd.read_csv('hcc_device_brand_age_sex.csv')
df_device_age_regression_prob_oof = pd.read_csv('device_age_regression_prob_oof.csv')
df_device_start_GRU_pred = pd.read_csv('device_start_GRU_pred.csv')
df_device_start_GRU_pred_age = pd.read_csv('device_start_GRU_pred_age.csv')
df_device_all_GRU_pred = pd.read_csv('device_all_GRU_pred.csv')
#df_boost_sex_age_prob_oof = pd.read_csv('boost_sex_age_prob_oof.csv')
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


# In[3]:


df_train_w2v = all_id.merge(df_sex_prob_oof, on='device_id', how='left')
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


# In[5]:


feat = df_train_w2v.copy()


# In[6]:


train=pd.merge(train_id,feat,on="device_id",how="left")
test=pd.merge(test_id,feat,on="device_id",how="left")


# In[8]:


features = [x for x in train.columns if x not in ['device_id', 'sex',"age",]]
Y = train['sex'] - 1


# In[9]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
seed = 1024
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)


# In[10]:


params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
#      'is_unbalance':'True',
# 'scale_pos_weight': 1500.0/13458.0,
        'eval_metric': "logloss",
    
    'gamma':0.2,#0.2 is ok
    'max_depth':6,
# 'lambda':20,
    # "alpha":5,
        'subsample':0.7,
        'colsample_bytree':0.4 ,
#         'min_child_weight':2.5, 
        'eta': 0.01,
    # 'learning_rate':0.01,
    "silent":1,
    'seed':1024,
'nthread':5,
   
    }

num_round = 3500
early_stopping_rounds = 100


# In[11]:


#预测性别
aus = []
sub1 = np.zeros((len(test), ))
pred_oob1=np.zeros((len(train),))
for i,(train_index,test_index) in enumerate(folds.split(train[features], Y)):
  
    tr_x = train[features].reindex(index=train_index, copy=False)
    tr_y = Y[train_index]
    te_x = train[features].reindex(index=test_index, copy=False)
    te_y = Y[test_index]

    d_tr = xgb.DMatrix(tr_x, label=tr_y)
    d_te = xgb.DMatrix(te_x, label=te_y)
    watchlist  = [(d_tr,'train'),
    (d_te,'val')
             ]
    model = xgb.train(params, d_tr, num_boost_round=530, 
                      evals=watchlist,verbose_eval=100)
    pred = model.predict(d_te)
    pred_oob1[test_index] =pred
    # te_y=te_y.apply(lambda x:1 if x>0 else 0)
    a = log_loss(te_y, pred)


    print ("idx: ", i) 
    print (" loss: %.5f" % a)
#     print " gini: %.5f" % g
    aus.append(a)

print ("mean")
print ("auc:       %s" % (sum(aus) / 5.0))


# In[12]:


#用全部数据训练一个lgb
#用全部的train来预测test
xgb_train = xgb.DMatrix(train[features], label=Y)
watchlist  = [(xgb_train,'train')]

gbm = xgb.train(params, xgb_train, num_boost_round=530, evals=watchlist, verbose_eval=100)  

sub1 = gbm.predict(xgb.DMatrix(test[features]))


# In[13]:


pred_oob1 = pd.DataFrame(pred_oob1, columns=['sex2'])
sub1 = pd.DataFrame(sub1, columns=['sex2'])
res1=pd.concat([pred_oob1,sub1])
res1['sex1'] = 1-res1['sex2']


# In[15]:


# In[50]:


features = [x for x in train.columns if x not in ['device_id',"age"]]
Y = train['age'] 


# In[51]:


from sklearn.metrics import auc, log_loss, roc_auc_score,f1_score,recall_score,precision_score


# In[16]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
seed = 1024
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)


# In[17]:


params={
    'booster':'gbtree',
    'objective': 'multi:softprob',
    'eval_metric': "mlogloss",
    'num_class':11,
    'gamma':0.1,#0.2 is ok
    'max_depth':5,
    'subsample':0.7,
    'colsample_bytree':0.4 ,
    # 'min_child_weight':2.5, 
    'eta': 0.02,
    # 'learning_rate':0.01,
    "silent":1,
    'seed':1024,
    'nthread':5,
   
    }


# In[19]:


#预测性别
aus = []
sub2 = np.zeros((len(test),11 ))
pred_oob2=np.zeros((len(train),11))
models=[]
iters=[]
for i,(train_index,test_index) in enumerate(folds.split(train[features], Y)):
  
    tr_x = train[features].reindex(index=train_index, copy=False)
    tr_y = Y[train_index]
    te_x = train[features].reindex(index=test_index, copy=False)
    te_y = Y[test_index]

    d_tr = xgb.DMatrix(tr_x, label=tr_y)
    d_te = xgb.DMatrix(te_x, label=te_y)
    watchlist  = [(d_tr,'train'),
                  (d_te,'val')]
    model = xgb.train(params, d_tr, num_boost_round=550, 
                      evals=watchlist,verbose_eval=100)

    pred = model.predict(d_te)
    pred_oob2[test_index] = pred
    # te_y=te_y.apply(lambda x:1 if x>0 else 0)
    a = log_loss(te_y, pred)

    #sub2 += gbm.predict(test[features], num_iteration=gbm.best_iteration) / 5
    
    print ("idx: ", i) 
    print (" loss: %.5f" % a)
#     print " gini: %.5f" % g
    aus.append(a)

print ("mean")
print ("auc:       %s" % (sum(aus) / 5.0))


# In[20]:


#预测条件概率
####sex1
test['sex']=1
#用全部数据训练一个lgb
#用全部的train来预测test
xgb_train = xgb.DMatrix(train[features], label=Y)
watchlist  = [(xgb_train,'train')]

gbm = xgb.train(params, xgb_train, num_boost_round=550, evals=watchlist, verbose_eval=100)   
sub2 = gbm.predict(xgb.DMatrix(test[features]))

res2_1=np.vstack((pred_oob2,sub2))
res2_1 = pd.DataFrame(res2_1)


# In[21]:


###sex2
#预测条件概率
test['sex']=2

sub2 = np.zeros((len(test),11))
sub2 = gbm.predict(xgb.DMatrix(test[features]))
res2_2=np.vstack((pred_oob2,sub2))
res2_2 = pd.DataFrame(res2_2) 


# In[24]:


res1.index=range(len(res1))
res2_1.index=range(len(res2_1))
res2_2.index=range(len(res2_2))
final_1=res2_1.copy()
final_2=res2_2.copy()


# In[25]:


for i in range(11):
    final_1[i]=res1['sex1'] * res2_1[i]
    final_2[i]=res1['sex2'] * res2_2[i]
id_list = pd.concat([train[['device_id']],test[['device_id']]])
final = id_list
final.index = range(len(final))
final.columns = ['DeviceID']
final_pred = pd.concat([final_1,final_2], 1)
final = pd.concat([final,final_pred],1)
final.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']


# In[27]:


test['DeviceID']=test['device_id']
sub=pd.merge(test[['DeviceID']],final,on="DeviceID",how="left")
sub.to_csv("th_xgb_nb.csv",index=False)

