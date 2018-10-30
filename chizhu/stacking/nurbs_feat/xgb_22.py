
# coding: utf-8

# In[2]:


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
# get_ipython().run_line_magic('matplotlib', 'inline')

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
path="./feature/"###nurbs概率文件路径
o_path="/dev/shm/chizhu_data/data/"###原始文件路径
os.listdir(path)


# In[4]:



all_feat=pd.read_csv(path+"feature_22_all.csv")
train_id=pd.read_csv(o_path+"deviceid_train.tsv",sep="\t",names=['device_id','sex','age'])
test_id=pd.read_csv(o_path+"deviceid_test.tsv",sep="\t",names=['device_id'])
all_id=pd.concat([train_id[['device_id']],test_id[['device_id']]])
all_id.index=range(len(all_id))
all_feat['device_id']=all_id
# deepnn_feat=pd.read_csv(path+"deepnn_fix.csv")
# deepnn_feat['device_id']=deepnn_feat['DeviceID']
# del deepnn_feat['DeviceID']


# In[9]:


train=pd.merge(train_id,all_feat,on="device_id",how="left")
# train=pd.merge(train,deepnn_feat,on="device_id",how="left")
test=pd.merge(test_id,all_feat,on="device_id",how="left")
# test=pd.merge(test,deepnn_feat,on="device_id",how="left")


# In[10]:


train['sex-age']=train.apply(lambda x:str(x['sex'])+"-"+str(x['age']),1)


# In[11]:


features = [x for x in train.columns if x not in ['device_id',"sex",'age','sex-age']]
label="sex-age"


# In[12]:


Y_CAT=pd.Categorical(train[label])


# In[13]:


import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import auc, log_loss, roc_auc_score,f1_score,recall_score,precision_score
from sklearn.cross_validation import StratifiedKFold

kf = StratifiedKFold(Y_CAT, n_folds=5, shuffle=True, random_state=1024)
params={
	'booster':'gbtree',
     "tree_method":"gpu_hist",
    "gpu_id":"1",
	'objective': 'multi:softprob',
#      'is_unbalance':'True',
# 	'scale_pos_weight': 1500.0/13458.0,
        'eval_metric': "mlogloss",
    'num_class':22,
	'gamma':0.1,#0.2 is ok
	'max_depth':6,
# 	'lambda':20,
    # "alpha":5,
        'subsample':0.7,
        'colsample_bytree':0.4 ,
        # 'min_child_weight':2.5, 
        'eta': 0.01,
    # 'learning_rate':0.01,
    "silent":1,
	'seed':1024,
	'nthread':12,
   
    }
num_round = 3500
early_stopping_rounds = 100


# In[14]:


aus = []
sub2 = np.zeros((len(test),22 ))
pred_oob2=np.zeros((len(train),22))
models=[]
iters=[]
for i,(train_index,test_index) in enumerate(kf):
  
    tr_x = train[features].reindex(index=train_index, copy=False)
    tr_y = Y_CAT.codes[train_index]
    te_x = train[features].reindex(index=test_index, copy=False)
    te_y = Y_CAT.codes[test_index]

    # tr_y=tr_y.apply(lambda x:1 if x>0 else 0)
    # te_y=te_y.apply(lambda x:1 if x>0 else 0)
    d_tr = xgb.DMatrix(tr_x, label=tr_y)
    d_te = xgb.DMatrix(te_x, label=te_y)
    watchlist  = [(d_tr,'train'),
    (d_te,'val')
             ]
    model = xgb.train(params, d_tr, num_boost_round=5500, 
                      evals=watchlist,verbose_eval=200,
                              early_stopping_rounds=100)
    models.append(model)
    iters.append(model.best_iteration)
    pred = model.predict(d_te,ntree_limit=model.best_iteration)
    pred_oob2[test_index] =pred
    # te_y=te_y.apply(lambda x:1 if x>0 else 0)
    a = log_loss(te_y, pred)

    sub2 += model.predict(xgb.DMatrix(test[features]),ntree_limit=model.best_iteration)/5
    

    print ("idx: ", i) 
    print (" loss: %.5f" % a)
#     print " gini: %.5f" % g
    aus.append(a)

print ("mean")
print ("loss:       %s" % (sum(aus) / 5.0))


# In[15]:


res=np.vstack((pred_oob2,sub2))
res = pd.DataFrame(res,columns=Y_CAT.categories)
res['DeviceID']=all_id
res=res[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

res.to_csv("xgb_nurbs_22_feat.csv",index=False)


# In[16]:


test['DeviceID']=test['device_id']
sub=pd.merge(test[['DeviceID']],res,on="DeviceID",how="left")
sub.to_csv("xgb_nurbs_22.csv",index=False)

