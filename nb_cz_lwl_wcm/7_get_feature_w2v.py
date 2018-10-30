from gensim.models import Word2Vec
import pandas as pd
path="Demo/"
packages = pd.read_csv(path+"deviceid_packages.tsv",
                       sep="\t", names=['id', 'app_list'])
packages['app_count'] = packages['app_list'].apply(
    lambda x: len(x.split(",")), 1)
documents = packages['app_list'].values.tolist()
texts = [[word for word in str(document).split(',')] for document in documents]
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
# texts = [[token for token in text if frequency[token] >= 5] for text in texts]
w2v = Word2Vec(texts, size=128, window=10, iter=45,
               workers=12, seed=1017, min_count=5)
w2v.wv.save_word2vec_format('./w2v_128.txt')

import gensim
import numpy as np


def get_w2v_avg(text, w2v_out_path, word2vec_Path):
    texts = []
    w2v_dim = 128
    data = text
#     data = pd.read_csv(text_path)
    data['app_list'] = data['app_list'].apply(
        lambda x: x.strip().split(","), 1)
    texts = data['app_list'].values.tolist()

    model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_Path, binary=False)
    vacab = model.vocab.keys()

    w2v_feature = np.zeros((len(texts), w2v_dim))
    w2v_feature_avg = np.zeros((len(texts), w2v_dim))

    for i, line in enumerate(texts):
        num = 0
        if line == '':
            w2v_feature_avg[i, :] = np.zeros(w2v_dim)
        else:
            for word in line:
                num += 1
                vec = model[word] if word in vacab else np.zeros(w2v_dim)
                w2v_feature[i, :] += vec
            w2v_feature_avg[i, :] = w2v_feature[i, :] / num
    w2v_avg = pd.DataFrame(w2v_feature_avg)
    w2v_avg.columns = ['w2v_avg_' + str(i) for i in w2v_avg.columns]
    w2v_avg['id'] = data['id']
    w2v_avg.to_csv(w2v_out_path, encoding='utf-8', index=None)
    return w2v_avg


w2v_feat = get_w2v_avg(packages, "feature/w2v_avg.csv", "w2v_128.txt")


