
# coding: utf-8

# In[1]:


# coding: utf-8
import feather
import os
import re
import sys  
import gc
import random
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from scipy import stats
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.engine.topology import Layer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.utils.training_utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from TextModel import *
import warnings
warnings.filterwarnings('ignore')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# In[2]:
print('14.device_start_GRU_pred_age.py')

df_doc = pd.read_csv('01.device_click_app_sorted_by_start.csv')
deviceid_test=pd.read_csv('input/deviceid_test.tsv',sep='\t',names=['device_id'])
deviceid_train=pd.read_csv('input/deviceid_train.tsv',sep='\t',names=['device_id','sex','age'])
df_total = pd.concat([deviceid_train, deviceid_test])
df_doc = df_doc.merge(df_total, on='device_id', how='left')


df_wv2_all = pd.read_csv('w2c_all_emb.csv')

dic_w2c_all = {}
for row in df_wv2_all.values :
    app_id = row[0]
    vector = row[1:]
    dic_w2c_all[app_id] = vector


# In[3]:


train = df_doc[df_doc['age'].notnull()]
test = df_doc[df_doc['age'].isnull()]
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

lb = LabelEncoder()
train_label = lb.fit_transform(train['age'].values)
train['class'] = train_label


# In[5]:


column_name="app_list"
word_seq_len = 900
victor_size = 200
num_words = 35000
batch_size = 64
classification = 11
kfold=10


# In[6]:


from sklearn.metrics import log_loss

def get_mut_label(y_label) :
    results = []
    for ele in y_label :
        results.append(ele.argmax())
    return  results  

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            val_y = get_mut_label(self.y_val)
            score = log_loss(val_y, y_pred)
            print("\n mlogloss - epoch: %d - score: %.6f \n" % (epoch+1, score))


# In[7]:


#词向量
def w2v_pad(df_train,df_test,col, maxlen_,victor_size, num_words):

    tokenizer = text.Tokenizer(num_words=num_words, lower=False,filters="")
    tokenizer.fit_on_texts(list(df_train[col].values)+list(df_test[col].values))

    train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train[col].values), maxlen=maxlen_)
    test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test[col].values), maxlen=maxlen_)
    
    word_index = tokenizer.word_index
    
    count = 0
    nb_words = len(word_index)
    print(nb_words)
    all_data=pd.concat([df_train[col],df_test[col]])
    file_name = 'embedding/' + 'Word2Vec_start_' + col  +"_"+ str(victor_size) + '.model'
    if not os.path.exists(file_name):
        model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                         size=victor_size, window=5, iter=10, workers=11, seed=2018, min_count=2)
        model.save(file_name)
    else:
        model = Word2Vec.load(file_name)
    print("add word2vec finished....")    


                 
    embedding_word2vec_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            count += 1
            embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_word2vec_matrix[i] = unk_vec

    embedding_w2c_all = np.zeros((nb_words + 1, victor_size))  
    for word, i in word_index.items():
        embedding_vector = dic_w2c_all[word] 
        embedding_w2c_all[i] = embedding_vector
                    

    #embedding_matrix = np.concatenate((embedding_word2vec_matrix,embedding_w2c_all),axis=1)
    embedding_matrix = embedding_word2vec_matrix
    
    return train_, test_, word_index, embedding_matrix


# In[8]:


train_, test_,word2idx, word_embedding = w2v_pad(train,test,column_name, word_seq_len,victor_size, num_words)


# In[11]:


my_opt="bi_gru_model"
#参数
Y = train['class'].values

if not os.path.exists("cache/"+my_opt):
    os.mkdir("cache/"+my_opt)


# In[17]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
seed = 2006
num_folds = 10
kf = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed).split(train_, Y)

epochs = 4
my_opt=eval(my_opt)
train_model_pred = np.zeros((train_.shape[0], classification))
test_model_pred = np.zeros((test_.shape[0], classification))
for i, (train_fold, val_fold) in enumerate(kf):
    X_train, X_valid, = train_[train_fold, :], train_[val_fold, :]
    y_train, y_valid = Y[train_fold], Y[val_fold]

    y_tra = to_categorical(y_train)
    y_val = to_categorical(y_valid)
    
    #模型
    name = str(my_opt.__name__)    

    model = my_opt(word_seq_len, word_embedding, classification)    
    
    
    RocAuc = RocAucEvaluation(validation_data=(X_valid, y_val), interval=1)

    hist = model.fit(X_train, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_val),
                     callbacks=[RocAuc])   
    
    
    train_model_pred[val_fold, :] =  model.predict(X_valid)


# In[21]:


#模型
#用全部的数据预测
train_label = to_categorical(Y)
name = str(my_opt.__name__)    

model = my_opt(word_seq_len, word_embedding, classification)    


RocAuc = RocAucEvaluation(validation_data=(train_, train_label), interval=1)

hist = model.fit(train_, train_label, batch_size=batch_size, epochs=epochs, validation_data=(train_, train_label),
                 callbacks=[RocAuc])   


test_model_pred =  model.predict(test_)


# In[22]:


df_train_pred = pd.DataFrame(train_model_pred)
df_test_pred = pd.DataFrame(test_model_pred)
df_train_pred.columns = ['device_start_GRU_pred_age_' + str(i) for i in range(11)]
df_test_pred.columns = ['device_start_GRU_pred_age_' + str(i) for i in range(11)]


# In[23]:


df_train_pred = pd.concat([train[['device_id']], df_train_pred], axis=1)
df_test_pred = pd.concat([test[['device_id']], df_test_pred], axis=1)


# In[24]:


df_results = pd.concat([df_train_pred, df_test_pred])
df_results.to_csv('device_start_GRU_pred_age.csv', index=None)

