
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
path="./"
os.listdir(path)


# In[2]:
print ('26.thluo_nb_lgb.py')

train_id=pd.read_csv("input/deviceid_train.tsv",sep="\t",names=['device_id','sex','age'])
test_id=pd.read_csv("input/deviceid_test.tsv",sep="\t",names=['device_id'])
all_id=pd.concat([train_id[['device_id']],test_id[['device_id']]])
#nurbs=pd.read_csv("nurbs_feature_all.csv")
#nurbs.columns=["nurbs_"+str(i) for i in nurbs.columns]
thluo = pd.read_csv("thluo_train_best_feat.csv")
del thluo['age']
del thluo['sex']
del thluo['sex_age']


# In[7]:


feat = thluo.copy()


# In[8]:


train=pd.merge(train_id,feat,on="device_id",how="left")
test=pd.merge(test_id,feat,on="device_id",how="left")


# In[11]:


features = [x for x in train.columns if x not in ['device_id', 'sex',"age",]]
Y = train['sex'] - 1


# In[12]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
seed = 1024
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)


# In[13]:


params = {
    'boosting_type': 'gbdt',
    'learning_rate' : 0.02,
    #'max_depth':5,
    'num_leaves' : 2 ** 5,
    'metric': {'binary_logloss'},
    #'num_class' : 22,
    'objective' : 'binary',
    'random_state' : 6666,
    'bagging_freq' : 5,
    'feature_fraction' : 0.7,
    'bagging_fraction' : 0.7,
    'min_split_gain' : 0.0970905919552776,
    'min_child_weight' : 9.42012323936088,  
}


# In[14]:


#预测性别
aus = []
sub1 = np.zeros((len(test), ))
pred_oob1=np.zeros((len(train),))
for i,(train_index,test_index) in enumerate(folds.split(train[features], Y)):
  
    tr_x = train[features].reindex(index=train_index, copy=False)
    tr_y = Y[train_index]
    te_x = train[features].reindex(index=test_index, copy=False)
    te_y = Y[test_index]

    lgb_train=lgb.Dataset(tr_x,label=tr_y)
    lgb_eval = lgb.Dataset(te_x, te_y, reference=lgb_train)

    gbm = lgb.train(params, lgb_train, num_boost_round=300, 
                    valid_sets=[lgb_train, lgb_eval], verbose_eval=100)         

    pred = gbm.predict(te_x[tr_x.columns.values])
    pred_oob1[test_index] =pred
    # te_y=te_y.apply(lambda x:1 if x>0 else 0)
    a = log_loss(te_y, pred)
    

    print ("idx: ", i) 
    print (" loss: %.5f" % a)
#     print " gini: %.5f" % g
    aus.append(a)

print ("mean")
print ("auc:       %s" % (sum(aus) / 5.0))


# In[15]:


#用全部数据训练一个lgb
#用全部的train来预测test
lgb_train = lgb.Dataset(train[features],label=Y)

gbm = lgb.train(params, lgb_train, num_boost_round=300, valid_sets=lgb_train, verbose_eval=100)  

sub1 = gbm.predict(test[features])


# In[16]:


pred_oob1 = pd.DataFrame(pred_oob1, columns=['sex2'])
sub1 = pd.DataFrame(sub1, columns=['sex2'])
res1=pd.concat([pred_oob1,sub1])
res1['sex1'] = 1-res1['sex2']


# In[18]:



# In[50]:


features = [x for x in train.columns if x not in ['device_id',"age"]]
Y = train['age'] 


# In[51]:


import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import auc, log_loss, roc_auc_score,f1_score,recall_score,precision_score


# In[19]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
seed = 1024
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)


# In[20]:


params = {
    'boosting_type': 'gbdt',
    'learning_rate' : 0.02,
    #'max_depth':5,
    'num_leaves' : 2 ** 5,
    'metric': {'multi_logloss'},
    'num_class' : 11,
    'objective' : 'multiclass',
    'random_state' : 6666,
    'bagging_freq' : 5,
    'feature_fraction' : 0.7,
    'bagging_fraction' : 0.7,
    'min_split_gain' : 0.0970905919552776,
    'min_child_weight' : 9.42012323936088,  
}


# In[22]:


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

    lgb_train=lgb.Dataset(tr_x,label=tr_y)
    lgb_eval = lgb.Dataset(te_x, te_y, reference=lgb_train)

    gbm = lgb.train(params, lgb_train, num_boost_round=430, 
                    valid_sets=[lgb_train, lgb_eval], verbose_eval=100)         

    pred = gbm.predict(te_x[tr_x.columns.values])
    pred_oob2[test_index] = pred
    # te_y=te_y.apply(lambda x:1 if x>0 else 0)
    a = log_loss(te_y, pred)

    #sub2 += gbm.predict(test[features], num_iteration=gbm.best_iteration) / 5
    
    models.append(gbm)
    iters.append(gbm.best_iteration)    

    print ("idx: ", i) 
    print (" loss: %.5f" % a)
#     print " gini: %.5f" % g
    aus.append(a)

print ("mean")
print ("auc:       %s" % (sum(aus) / 5.0))


# In[23]:


#预测条件概率
####sex1
test['sex']=1
#用全部数据训练一个lgb
#用全部的train来预测test
lgb_train = lgb.Dataset(train[features],label=Y)

gbm = lgb.train(params, lgb_train, num_boost_round=430, valid_sets=lgb_train, verbose_eval=100)  
sub2 = gbm.predict(test[features])

res2_1=np.vstack((pred_oob2,sub2))
res2_1 = pd.DataFrame(res2_1)


# In[24]:


###sex2
#预测条件概率
test['sex']=2

sub2 = np.zeros((len(test),11))
sub2 = gbm.predict(test[features], num_iteration = gbm.best_iteration)
res2_2=np.vstack((pred_oob2,sub2))
res2_2 = pd.DataFrame(res2_2) 


# In[27]:


res1.index=range(len(res1))
res2_1.index=range(len(res2_1))
res2_2.index=range(len(res2_2))
final_1=res2_1.copy()
final_2=res2_2.copy()


# In[28]:


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


# In[30]:


test['DeviceID']=test['device_id']
sub=pd.merge(test[['DeviceID']],final,on="DeviceID",how="left")
sub.to_csv("th_lgb_nb.csv",index=False)

