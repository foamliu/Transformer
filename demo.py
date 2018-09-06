# import the necessary packages
import os
import pickle
import random

import keras.backend as K
import nltk
import numpy as np
from gensim.models import KeyedVectors

from config import stop_word, unknown_word, Tx, Ty, embedding_size, hidden_size, unknown_embedding, stop_embedding, \
    vocab_size_zh
from config import valid_translation_folder, valid_translation_en_filename, valid_translation_zh_filename
from model import build_model

if __name__ == '__main__':
    channel = 3

    model_weights_path = 'models/model.01-10.3439.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    print('loading fasttext word embedding(en)')
    word_vectors_en = KeyedVectors.load_word2vec_format('data/wiki.en.vec')

    vocab_zh = pickle.load(open('data/vocab_train_zh.p', 'rb'))
    idx2word_zh = vocab_zh
    print('len(idx2word_zh): ' + str(len(idx2word_zh)))
    word2idx_zh = dict(zip(idx2word_zh, range(len(vocab_zh))))
    print('vocab_size_zh: ' + str(vocab_size_zh))

    print(model.summary())

    translation_path_en = os.path.join(valid_translation_folder, valid_translation_en_filename)
    translation_path_zh = os.path.join(valid_translation_folder, valid_translation_zh_filename)
    filename = 'data/samples_valid.p'

    print('loading valid texts and vocab')
    with open(translation_path_en, 'r') as f:
        data_en = f.readlines()

    with open(translation_path_zh, 'r') as f:
        data_zh = f.readlines()

    indices = range(len(data_en))

    length = 10
    samples = random.sample(indices, length)

    for i in range(length):
        idx = samples[i]
        sentence_en = data_en[idx]
        print(sentence_en)
        tokens = nltk.word_tokenize(sentence_en)
        x = np.zeros((1, Tx, embedding_size), np.float32)
        for j, word in enumerate(tokens):
            try:
                x[0, j] = word_vectors_en[word]
            except KeyError:
                word = unknown_word
                x[0, j] = unknown_embedding

        x[0, len(tokens)] = stop_embedding

        preds = model.predict(x)
        print('preds.shape: ' + str(preds.shape))

        output_zh = []
        for t in range(Ty):
            idx = np.argmax(preds[0][t])
            word_pred = idx2word_zh[idx]
            output_zh.append(word_pred)
            if word_pred == stop_word:
                break
        print(' '.join(output_zh))

    K.clear_session()
