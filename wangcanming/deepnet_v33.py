import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime,timedelta  
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
%matplotlib inline

#add
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from gensim.models import FastText, Word2Vec
import re
from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
import keras.backend as K
from keras.optimizers import *
from keras.utils import to_categorical

packages = pd.read_csv('../input/yiguan/demo/Demo/deviceid_packages.tsv', sep='\t', names=['device_id', 'apps'])
test = pd.read_csv('../input/yiguan/demo/Demo/deviceid_test.tsv', sep='\t', names=['device_id'])
train = pd.read_csv('../input/yiguan/demo/Demo/deviceid_train.tsv', sep='\t', names=['device_id', 'sex', 'age'])

brand = pd.read_table('../input/yiguan/demo/Demo/deviceid_brand.tsv', names=['device_id', 'vendor', 'version'])


packages['app_lenghth'] = packages['apps'].apply(lambda x:x.split(',')).apply(lambda x:len(x))
packages['app_list'] = packages['apps'].apply(lambda x:x.split(','))
train = pd.merge(train, packages, on='device_id', how='left')
test = pd.merge(test, packages, on='device_id', how='left')

embed_size = 128
fastmodel = Word2Vec(list(packages['app_list']), size=embed_size, window=4, min_count=1, negative=2,
                 sg=1, sample=0.001, hs=1, workers=4)  

embedding_fast = pd.DataFrame([fastmodel[word] for word in (fastmodel.wv.vocab)])
embedding_fast['app'] = list(fastmodel.wv.vocab)
embedding_fast.columns= ["fdim_%s" % str(i) for i in range(embed_size)]+["app"]

tokenizer = Tokenizer(lower=False, char_level=False, split=',')

tokenizer.fit_on_texts(list(packages['apps']))

X_seq = tokenizer.texts_to_sequences(train['apps'])
X_test_seq = tokenizer.texts_to_sequences(test['apps'])

maxlen = 50
X = pad_sequences(X_seq, maxlen=maxlen, value=0)
X_test = pad_sequences(X_test_seq, maxlen=maxlen, value=0)
Y_sex = train['sex']-1

max_feaures=35001
embedding_matrix = np.zeros((max_feaures, embed_size))
for word in tokenizer.word_index:
    if word not in fastmodel.wv.vocab:
        continue
    embedding_matrix[tokenizer.word_index[word]] = fastmodel[word]

class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (2/4)
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p # decoupled weight decay (4/4)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def model_conv1D_sex(embedding_matrix):
    
    K.clear_session()
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False)
    
    # Define inputs
    seq = Input(shape=(maxlen,))
    
    # Run inputs through embedding
    emb = emb_layer(seq)
    
    lstm_layer = Bidirectional(GRU(128, recurrent_dropout=0.15, dropout=0.15,))
    lstm = lstm_layer(emb)
    
    translate = TimeDistributed(Dense(128, activation='relu'))
    t1 = translate(emb)
    t1 = TimeDistributed(Dropout(0.15))(t1)
    sum_op = Lambda(lambda x: K.sum(x, axis=1), output_shape=(128,))
    t1 = sum_op(t1)

    merge1 = concatenate([lstm, t1])
    
    # The MLP that determines the outcome
    x = Dropout(0.24)(merge1)
    #x = BatchNormalization()(x)
    #x = Dense(200, activation='relu',)(x)
    #x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=seq, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.1,))###

    return model


kfold = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
sub1 = np.zeros((X_test.shape[0], ))
oof_pref1 = np.zeros((X.shape[0], 1))
score = []
for i, (train_index, test_index) in enumerate(kfold.split(X, Y_sex)):
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr, earlystopping]
    model_sex = model_conv1D_sex(embedding_matrix)
    X_tr, X_vl, y_tr, y_vl = X[train_index], X[test_index], Y_sex[train_index], Y_sex[test_index]
    hist = model_sex.fit(X_tr, y_tr, batch_size=512, epochs=50, validation_data=(X_vl, y_vl),
                 callbacks=callbacks, verbose=2, shuffle=True)
    model_sex.load_weights(filepath)
    sub1 += np.squeeze(model_sex.predict(X_test))/kfold.n_splits
    oof_pref1[test_index] = model_sex.predict(X_vl)
    score.append(np.min(hist.history['val_loss']))
print('log loss:',np.mean(score))

def model_age_conv(embedding_matrix):
    
    K.clear_session()
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False)
    
    # Define inputs
    seq = Input(shape=(maxlen,))
    
    # Run inputs through embedding
    emb = emb_layer(seq)
    
    lstm_layer = Bidirectional(GRU(128, recurrent_dropout=0.15, dropout=0.15,))
    lstm = lstm_layer(emb)
    
    translate = TimeDistributed(Dense(128, activation='relu'))
    t1 = translate(emb)
    t1 = TimeDistributed(Dropout(0.15))(t1)
    sum_op = Lambda(lambda x: K.sum(x, axis=1), output_shape=(128,))
    t1 = sum_op(t1)
    
    merge1 = concatenate([lstm, t1])
    
    # The MLP that determines the outcome
    x = Dropout(0.24)(merge1)
    #x = BatchNormalization()(x)
    #x = Dense(200, activation='relu',)(x)
    #x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    pred = Dense(11, activation='softmax')(x)

    model = Model(inputs=seq, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer=AdamW(weight_decay=0.1,))

    return model

Y_age = to_categorical(train['age'])

sub2 = np.zeros((X_test.shape[0], 11))
oof_pref2 = np.zeros((X.shape[0], 11))
score = []


for i, (train_index, test_index) in enumerate(kfold.split(X, train['age'])):
    filepath2="weights_best2.h5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=2)
    earlystopping2 = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8, verbose=2, mode='auto')
    callbacks2 = [checkpoint2, reduce_lr2, earlystopping2]
    model_age = model_age_conv(embedding_matrix)
    X_tr, X_vl, y_tr, y_vl = X[train_index], X[test_index], Y_age[train_index], Y_age[test_index]
    hist = model_age.fit(X_tr, y_tr, batch_size=512, epochs=50, validation_data=(X_vl, y_vl),
                 callbacks=callbacks2, verbose=2, shuffle=True)
    
    model_age.load_weights(filepath2)
    sub2 += model_age.predict(X_test)/kfold.n_splits
    oof_pref2[test_index] = model_age.predict(X_vl)
    score.append(np.min(hist.history['val_loss']))

print('log loss:',np.mean(score))

sub1 = pd.DataFrame(sub1, columns=['sex2'])
oof_pref1 = pd.DataFrame(oof_pref1, columns=['sex2'])
sub1['sex1'] = 1-sub1['sex2']
oof_pref1['sex1'] = 1-oof_pref1['sex2']
sub2 = pd.DataFrame(sub2, columns=['age%s'%i for i in range(11)])
oof_pref2 = pd.DataFrame(oof_pref2, columns=['age%s'%i for i in range(11)])
sub = test[['device_id']]
sub.columns = ['DeviceID']
oof = train[['device_id']]
oof.columns = ['DeviceID']
for i in ['sex1', 'sex2']:
    for j in ['age%s'%i for i in range(11)]:
        sub[i+'_'+j] = sub1[i]*sub2[j]
        oof[i+'_'+j] = oof_pref1[i]*oof_pref2[j]
sub.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']
oof.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', 
         '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', 
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

sub.to_csv('deepnet_v33.csv', index=False)
oof.to_csv('deepnet_oof_v33.csv', index=False)

df_stack = pd.concat([oof, sub])
df_stack.to_csv('feature_wcm.csv')

