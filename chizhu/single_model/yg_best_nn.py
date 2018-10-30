import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# %matplotlib inline

#add
from category_encoders import OrdinalEncoder
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
from config import path
packages = pd.read_csv(path+'deviceid_packages.tsv',
                       sep='\t', names=['device_id', 'apps'])
test = pd.read_csv(path+'deviceid_test.tsv',
                   sep='\t', names=['device_id'])
train = pd.read_csv(path+'deviceid_train.tsv',
                    sep='\t', names=['device_id', 'sex', 'age'])

brand = pd.read_table(path+'deviceid_brand.tsv',
                      names=['device_id', 'vendor', 'version'])
behave = pd.read_csv('data/user_behavior.csv')

brand['phone_version'] = brand['vendor'] + ' ' + brand['version']
train = pd.merge(brand[['device_id', 'phone_version']],
                 train, on='device_id', how='right')
test = pd.merge(brand[['device_id', 'phone_version']],
                test, on='device_id', how='right')

train = pd.merge(train, behave, on='device_id', how='left')
test = pd.merge(test, behave, on='device_id', how='left')

packages['app_lenghth'] = packages['apps'].apply(
    lambda x: x.split(',')).apply(lambda x: len(x))
packages['app_list'] = packages['apps'].apply(lambda x: x.split(','))
train = pd.merge(train, packages, on='device_id', how='left')
test = pd.merge(test, packages, on='device_id', how='left')

embed_size = 128
fastmodel = Word2Vec(list(packages['app_list']), size=embed_size, window=4, min_count=3, negative=2,
                     sg=1, sample=0.002, hs=1, workers=4)

embedding_fast = pd.DataFrame([fastmodel[word]
                               for word in (fastmodel.wv.vocab)])
embedding_fast['app'] = list(fastmodel.wv.vocab)
embedding_fast.columns = ["fdim_%s" %
                          str(i) for i in range(embed_size)]+["app"]


tokenizer = Tokenizer(lower=False, char_level=False, split=',')

tokenizer.fit_on_texts(list(packages['apps']))

X_seq = tokenizer.texts_to_sequences(train['apps'])
X_test_seq = tokenizer.texts_to_sequences(test['apps'])

maxlen = 50
X = pad_sequences(X_seq, maxlen=maxlen, value=0)
X_test = pad_sequences(X_test_seq, maxlen=maxlen, value=0)
Y_sex = train['sex']-1

max_feaures = 35001
embedding_matrix = np.zeros((max_feaures, embed_size))
for word in tokenizer.word_index:
    if word not in fastmodel.wv.vocab:
        continue
    embedding_matrix[tokenizer.word_index[word]] = fastmodel[word]


X_h = train[['h%s' % i for i in range(24)]].values
X_h_test = test[['h%s' % i for i in range(24)]].values


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
            # decoupled weight decay (2/4)
            self.wd = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (3/4)

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
            # decoupled weight decay (4/4)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

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


def model_conv1D(embedding_matrix):

    K.clear_session()
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False
    )
    lstm_layer = Bidirectional(
        GRU(128, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))

    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)
    conv2 = Conv1D(filters=64, kernel_size=2,
                   padding='same', activation='relu', )
    conv3 = Conv1D(filters=64, kernel_size=3,
                   padding='same', activation='relu',)
    conv5 = Conv1D(filters=32, kernel_size=5,
                   padding='same', activation='relu',)

    # Define inputs
    seq = Input(shape=(maxlen,))

    # Run inputs through embedding
    emb = emb_layer(seq)

    lstm = lstm_layer(emb)
    # Run through CONV + GAP layers
    conv1a = conv1(lstm)
    gap1a = GlobalAveragePooling1D()(conv1a)
    gmp1a = GlobalMaxPool1D()(conv1a)

    conv2a = conv2(lstm)
    gap2a = GlobalAveragePooling1D()(conv2a)
    gmp2a = GlobalMaxPool1D()(conv2a)

    conv3a = conv3(lstm)
    gap3a = GlobalAveragePooling1D()(conv3a)
    gmp3a = GlobalMaxPooling1D()(conv3a)

    conv5a = conv5(lstm)
    gap5a = GlobalAveragePooling1D()(conv5a)
    gmp5a = GlobalMaxPooling1D()(conv5a)

    hin = Input(shape=(24, ))
    htime = Dense(6, activation='relu')(hin)
    merge1 = concatenate([gmp1a, gmp1a, gmp1a, gmp1a, htime])

    # The MLP that determines the outcome
    x = Dropout(0.3)(merge1)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq, hin], outputs=pred)
    model.compile(loss='binary_crossentropy',
                  optimizer=AdamW(weight_decay=0.08,))

    return model


kfold = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
sub1 = np.zeros((X_test.shape[0], ))
oof_pref1 = np.zeros((X.shape[0], 1))
score = []
count = 0
for i, (train_index, test_index) in enumerate(kfold.split(X, Y_sex)):
    print("FOLD | ", count+1)
    filepath = "sex_weights_best_%d.h5" % count
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=6, verbose=1, mode='auto')
    callbacks = [checkpoint, reduce_lr, earlystopping]

    model_sex = model_conv1D(embedding_matrix)
    X_tr, X_vl, X_tr2, X_vl2, y_tr, y_vl = X[train_index], X[test_index], X_h[
        train_index], X_h[test_index], Y_sex[train_index], Y_sex[test_index]
    hist = model_sex.fit([X_tr, X_tr2], y_tr, batch_size=256, epochs=50, validation_data=([X_vl, X_vl2], y_vl),
                         callbacks=callbacks, verbose=1, shuffle=True)
    model_sex.load_weights(filepath)
    sub1 += np.squeeze(model_sex.predict([X_test, X_h_test]))/kfold.n_splits
    oof_pref1[test_index] = model_sex.predict([X_vl, X_vl2])
    score.append(np.min(hist.history['val_loss']))
    count += 1
print('log loss:', np.mean(score))


oof_pref1 = pd.DataFrame(oof_pref1, columns=['sex2'])
sub1 = pd.DataFrame(sub1, columns=['sex2'])
res1 = pd.concat([oof_pref1, sub1])
res1['sex1'] = 1-res1['sex2']
# res1.to_csv("res1.csv", index=False)


def model_age_conv(embedding_matrix):

    # The embedding layer containing the word vectors
    K.clear_session()
    emb_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False
    )
    lstm_layer = Bidirectional(
        GRU(128, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))

    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu',)
    conv2 = Conv1D(filters=64, kernel_size=2,
                   padding='same', activation='relu', )
    conv3 = Conv1D(filters=64, kernel_size=3,
                   padding='same', activation='relu',)
    conv5 = Conv1D(filters=32, kernel_size=5,
                   padding='same', activation='relu',)

    # Define inputs
    seq = Input(shape=(maxlen,))

    # Run inputs through embedding
    emb = emb_layer(seq)

    lstm = lstm_layer(emb)
    # Run through CONV + GAP layers
    conv1a = conv1(lstm)
    gap1a = GlobalAveragePooling1D()(conv1a)
    gmp1a = GlobalMaxPool1D()(conv1a)

    conv2a = conv2(lstm)
    gap2a = GlobalAveragePooling1D()(conv2a)
    gmp2a = GlobalMaxPool1D()(conv2a)

    conv3a = conv3(lstm)
    gap3a = GlobalAveragePooling1D()(conv3a)
    gmp3a = GlobalMaxPooling1D()(conv3a)

    conv5a = conv5(lstm)
    gap5a = GlobalAveragePooling1D()(conv5a)
    gmp5a = GlobalMaxPooling1D()(conv5a)

    merge1 = concatenate([gap1a, gap2a, gap3a, gap5a])

    # The MLP that determines the outcome
    x = Dropout(0.3)(merge1)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu',)(x)
    x = Dropout(0.22)(x)
    x = BatchNormalization()(x)
    pred = Dense(11, activation='softmax')(x)

    model = Model(inputs=seq, outputs=pred)
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamW(weight_decay=0.08,))

    return model


Y_age = to_categorical(train['age'])

sub2 = np.zeros((X_test.shape[0], 11))
oof_pref2 = np.zeros((X.shape[0], 11))
score = []
count = 0
for i, (train_index, test_index) in enumerate(kfold.split(X, train['age'])):

    print("FOLD | ", count+1)

    filepath2 = "age_weights_best_%d.h5" % count
    checkpoint2 = ModelCheckpoint(
        filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr2 = ReduceLROnPlateau(
        monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
    earlystopping2 = EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
    callbacks2 = [checkpoint2, reduce_lr2, earlystopping2]

    X_tr, X_vl, y_tr, y_vl = X[train_index], X[test_index], Y_age[train_index], Y_age[test_index]

    model_age = model_age_conv(embedding_matrix)
    hist = model_age.fit(X_tr, y_tr, batch_size=256, epochs=50, validation_data=(X_vl, y_vl),
                         callbacks=callbacks2, verbose=2, shuffle=True)

    model_age.load_weights(filepath2)
    oof_pref2[test_index] = model_age.predict(X_vl)
    sub2 += model_age.predict(X_test)/kfold.n_splits
    score.append(np.min(hist.history['val_loss']))
    count += 1
print('log loss:', np.mean(score))


res2_1 = np.vstack((oof_pref2, sub2))
res2_1 = pd.DataFrame(res2_1)
# res2_1.to_csv("res2.csv", index=False)

res1.index = range(len(res1))
res2_1.index = range(len(res2_1))
final_1 = res2_1.copy()
final_2 = res2_1.copy()
for i in range(11):
    final_1[i] = res1['sex1']*res2_1[i]
    final_2[i] = res1['sex2']*res2_1[i]
id_list = pd.concat([train[['device_id']], test[['device_id']]])
final = id_list
final.index = range(len(final))
final.columns = ['DeviceID']
final_pred = pd.concat([final_1, final_2], 1)
final = pd.concat([final, final_pred], 1)
final.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6',
                 '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4',
                 '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

final.to_csv('submit/yg_best_nn.csv', index=False)
