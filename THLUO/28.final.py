
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


th_22_results_lgb = pd.read_csv('th_22_results_lgb.csv')
th_22_results_xgb = pd.read_csv('th_22_results_xgb.csv')
th_lgb_nb = pd.read_csv('th_lgb_nb.csv')
th_xgb_nb = pd.read_csv('th_xgb_nb.csv')


# In[5]:


#直接22分类 lgb与xgb进行55 45加权融合
results_22 = pd.DataFrame(th_22_results_lgb.values[:,1:] * 0.55 + th_22_results_xgb.values[:,1:] * 0.45)
results_22.columns = th_22_results_lgb.columns[1:]
results_22['DeviceID'] = th_22_results_lgb['DeviceID']


# In[6]:


#条件概率分类, xgb与lgb进行65 35加权融合
results_nb = pd.DataFrame(th_xgb_nb.values[:,1:] * 0.65 + th_lgb_nb.values[:,1:] * 0.35)
results_nb.columns = th_xgb_nb.columns[1:]
results_nb['DeviceID'] = th_xgb_nb['DeviceID']


# In[ ]:


#两份结果继续进行加权融合
results_final = pd.DataFrame(results_22.values[:,1:] * 0.65 + results_nb.values[:,1:] * 0.35)
results_final.columns = results_22.columns[1:]
results_final['DeviceID'] = results_22['DeviceID']


# In[ ]:


results_final.to_csv('result/thluo_final.csv', index=None)

