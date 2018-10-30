import os
import re
import sys
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import random
from keras.engine.topology import Layer
from util import *

def capsule_lstm(sent_length, embeddings_weight,class_num):
    print("get_text_capsule")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    embed = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNLSTM(200, return_sequences=True))(embed)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    capsule = Flatten()(capsule)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(capsule))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model







def get_text_capsule(sent_length, embeddings_weight,class_num):
    print("get_text_capsule")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    embed = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    capsule = Flatten()(capsule)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(capsule))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_cnn1(sent_length, embeddings_weight,class_num):
    print("get_text_cnn1")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    embed = embedding(content)

    embed = SpatialDropout1D(0.2)(embed)

    conv2 = Activation('relu')(BatchNormalization()(Conv1D(128, 2, padding='same')(embed)))
    conv2 = Activation('relu')(BatchNormalization()(Conv1D(64, 2, padding='same')(conv2)))
    conv2 = MaxPool1D(pool_size=50)(conv2)

    conv3 = Activation('relu')(BatchNormalization()(Conv1D(128, 3, padding='same')(embed)))
    conv3 = Activation('relu')(BatchNormalization()(Conv1D(64, 3, padding='same')(conv3)))
    conv3 = MaxPool1D(pool_size=50)(conv3)

    conv4 = Activation('relu')(BatchNormalization()(Conv1D(128, 4, padding='same')(embed)))
    conv4 = Activation('relu')(BatchNormalization()(Conv1D(64, 4, padding='same')(conv4)))
    conv4 = MaxPool1D(pool_size=50)(conv4)

    conv5 = Activation('relu')(BatchNormalization()(Conv1D(128, 5, padding='same')(embed)))
    conv5 = Activation('relu')(BatchNormalization()(Conv1D(64, 5, padding='same')(conv5)))
    conv5 = MaxPool1D(pool_size=50)(conv5)

    cnn = concatenate([conv2, conv3, conv4, conv5], axis=-1)
    flat = Flatten()(cnn)

    drop = Dropout(0.2)(flat)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(drop))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def get_text_cnn2(sent_length, embeddings_weight,class_num):
    print("get_text_cnn2")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    embed = embedding(content)
    filter_sizes = [2, 3, 4,5]
    num_filters = 128
    embed_size = embeddings_weight.shape[1]

    x = SpatialDropout1D(0.2)(embed)
    x = Reshape((sent_length, embed_size, 1))(x)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                    activation='relu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                    activation='relu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                    activation='relu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                    activation='relu')(x)

    maxpool_0 = MaxPool2D(pool_size=(sent_length - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sent_length - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sent_length - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(sent_length - filter_sizes[3] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(z))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def get_text_cnn3(sent_length, embeddings_weight,class_num):
    print("get_text_cnn3")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)(content)

    embedding = SpatialDropout1D(0.2)(embedding)

    cnn1 = Conv1D(128, 2, padding='same', strides=1, activation='relu')(embedding)
    cnn2 = Conv1D(128, 3, padding='same', strides=1, activation='relu')(embedding)
    cnn3 = Conv1D(128, 4, padding='same', strides=1, activation='relu')(embedding)
    cnn4 = Conv1D(128, 5, padding='same', strides=1, activation='relu')(embedding)
    cnn = concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)

    cnn1 = Conv1D(64, 2, padding='same', strides=1, activation='relu')(cnn)
    cnn1 = MaxPooling1D(pool_size=100)(cnn1)
    cnn2 = Conv1D(64, 3, padding='same', strides=1, activation='relu')(cnn)
    cnn2 = MaxPooling1D(pool_size=100)(cnn2)
    cnn3 = Conv1D(64, 4, padding='same', strides=1, activation='relu')(cnn)
    cnn3 = MaxPooling1D(pool_size=100)(cnn3)
    cnn4 = Conv1D(64, 5, padding='same', strides=1, activation='relu')(cnn)
    cnn4 = MaxPooling1D(pool_size=100)(cnn4)

    cnn = concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)

    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(drop))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_text_gru1(sent_length, embeddings_weight,class_num):
    print("get_text_gru1")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_gru2(sent_length, embeddings_weight,class_num):
    print("get_text_gru2")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    x = Conv1D(100, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_gru4(sent_length, embeddings_weight,class_num):
    print("get_text_gru4")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNLSTM(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_gru5(sent_length, embeddings_weight,class_num):
    print("get_text_gru5")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    x = Dropout(0.35)(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    last = Lambda(lambda t: t[:, -1])(x)
    maxpool = GlobalMaxPooling1D()(x)
    average = GlobalAveragePooling1D()(x)
    x = concatenate([last, maxpool, average])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_gru6(sent_length, embeddings_weight,class_num):
    print("get_text_gru6")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    x = Conv1D(60, kernel_size=3, padding='valid', activation='relu', strides=1)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    embed = SpatialDropout1D(0.2)(embedding(content))
    y = Bidirectional(CuDNNGRU(100, return_sequences=True))(embed)
    y = Conv1D(40, kernel_size=3, padding='valid', activation='relu', strides=1)(y)
    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)

    x = concatenate([avg_pool, max_pool, avg_pool2, max_pool2], -1)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_rcnn1(sent_length, embeddings_weight,class_num):
    print("get_text_rcnn1")
    document = Input(shape=(None,), dtype="int32")

    embedder = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    doc_embedding = SpatialDropout1D(0.2)(embedder(document))
    forward = Bidirectional(CuDNNLSTM(200, return_sequences=True))(doc_embedding)
    together = concatenate([forward, doc_embedding], axis=2)

    semantic = Conv1D(100, 2, padding='same', strides=1, activation='relu')(together)
    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(100,))(semantic)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(pool_rnn))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=document, outputs=output)
   # model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_rcnn2(sent_length, embeddings_weight,class_num):
    print("get_text_rcnn2")
    content = Input(shape=(None,), dtype="int32")

    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Convolution1D(filters=256, kernel_size=3, padding='same', strides=1, activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.2)(CuDNNGRU(units=200, return_sequences=True)(x))
    x = Dropout(0.2)(CuDNNGRU(units=100)(x))

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_rcnn3(sent_length, embeddings_weight,class_num):
    print("get_text_rcnn3")
    content = Input(shape=(None,), dtype="int32")

    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    cnn = Convolution1D(filters=200, kernel_size=3, padding="same", strides=1, activation="relu")(x)
    cnn_avg_pool = GlobalAveragePooling1D()(cnn)
    cnn_max_pool = GlobalMaxPooling1D()(cnn)

    rnn = Dropout(0.2)(CuDNNGRU(200, return_sequences=True)(x))
    rnn_avg_pool = GlobalAveragePooling1D()(rnn)
    rnn_max_pool = GlobalMaxPooling1D()(rnn)

    con = concatenate([cnn_avg_pool, cnn_max_pool, rnn_avg_pool, rnn_max_pool], axis=-1)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(con))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def get_text_rcnn4(sent_length, embeddings_weight,class_num):
    print("get_text_rcnn4")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))

    rnn_1 = Bidirectional(CuDNNGRU(128, return_sequences=True))(embed)
    conv_2 = Conv1D(128, 2, kernel_initializer="normal", padding="valid", activation="relu", strides=1)(rnn_1)

    maxpool = GlobalMaxPooling1D()(conv_2)
    attn = AttentionWeightedAverage()(conv_2)
    average = GlobalAveragePooling1D()(conv_2)

    x = concatenate([maxpool, attn, average])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_rcnn5(sent_length, embeddings_weight,class_num):
    print("get_text_rcnn5")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))

    rnn_1 = Bidirectional(CuDNNGRU(200, return_sequences=True))(embed)
    rnn_2 = Bidirectional(CuDNNGRU(200, return_sequences=True))(rnn_1)
    x = concatenate([rnn_1, rnn_2], axis=2)

    last = Lambda(lambda t: t[:, -1], name='last')(x)
    maxpool = GlobalMaxPooling1D()(x)
    attn = AttentionWeightedAverage()(x)
    average = GlobalAveragePooling1D()(x)

    x = concatenate([last, maxpool, average, attn])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_lstm1(sent_length, embeddings_weight,class_num):
    print("get_text_lstm1")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(200, return_sequences=True))(embed))
    semantic = TimeDistributed(Dense(100, activation="tanh"))(x)
    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(100,))(semantic)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(pool_rnn))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_lstm2(sent_length, embeddings_weight,class_num):
    print("get_text_lstm2")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(200, return_sequences=True))(embed))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(100, return_sequences=True))(x))
    semantic = TimeDistributed(Dense(100, activation="tanh"))(x)
    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(100,))(semantic)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(pool_rnn))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_lstm3(sent_length, embeddings_weight,class_num):
    print("get_text_lstm3")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.2)(Bidirectional(CuDNNLSTM(200, return_sequences=True))(embed))
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_lstm_attention(sent_length, embeddings_weight,class_num):
    print("get_text_lstm_attention")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embedded_sequences = SpatialDropout1D(0.2)(embedding(content))
    x = Dropout(0.25)(CuDNNLSTM(200, return_sequences=True)(embedded_sequences))
    merged = Attention(sent_length)(x)
    merged = Dense(100, activation='relu')(merged)
    merged = Dropout(0.25)(merged)

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(merged))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_text_dpcnn(sent_length, embeddings_weight,class_num):
    print("get_text_dpcnn")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))

    block1 = Conv1D(128, kernel_size=3, padding='same', activation='linear')(embed)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(128, kernel_size=3, padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    resize_emb = Conv1D(128, kernel_size=3, padding='same', activation='linear')(embed)
    resize_emb = PReLU()(resize_emb)

    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=10)(block1_output)

    block2 = Conv1D(128, kernel_size=4, padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(128, kernel_size=4, padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)

    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=10)(block2_output)

    block3 = Conv1D(128, kernel_size=5, padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(128, kernel_size=5, padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)

    output = add([block3, block2_output])
    maxpool = GlobalMaxPooling1D()(output)
    average = GlobalAveragePooling1D()(output)

    x = concatenate([maxpool, average])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model







def bi_gru_model(sent_length, embeddings_weight,class_num):
    print("get_text_gru3")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def bi_gru_model_binary(sent_length, embeddings_weight,class_num):
    print("bi_gru_model_binary")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


