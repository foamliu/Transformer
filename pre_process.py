import os
import pickle
import xml.etree.ElementTree
import zipfile
from collections import Counter

import jieba
import nltk
from gensim.models import KeyedVectors
from tqdm import tqdm

from config import start_word, stop_word, unknown_word, Tx, Ty, vocab_size_zh
from config import train_folder, valid_folder, test_a_folder, test_b_folder
from config import train_translation_folder, train_translation_zh_filename, train_translation_en_filename
from config import valid_translation_folder, valid_translation_zh_filename, valid_translation_en_filename
from utils import ensure_folder


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def build_train_vocab_zh():
    translation_path = os.path.join(train_translation_folder, train_translation_zh_filename)

    with open(translation_path, 'r') as f:
        data = f.readlines()

    vocab = []
    print('scanning train data (zh)')
    for sentence in tqdm(data):
        seg_list = jieba.cut(sentence.strip())
        for word in seg_list:
            vocab.append(word)

    counter_vocab = Counter(vocab)
    common = counter_vocab.most_common(vocab_size_zh - 3)
    vocab = [item[0] for item in common]
    vocab.append(start_word)
    vocab.append(stop_word)
    vocab.append(unknown_word)
    print('vocab size (zh): ' + str(len(vocab)))

    filename = 'data/vocab_train_zh.p'
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def extract_valid_data():
    valid_translation_path = os.path.join(valid_translation_folder, 'valid.en-zh.en.sgm')
    with open(valid_translation_path, 'r') as f:
        data_en = f.readlines()
    data_en = [line.replace(' & ', ' &amp; ') for line in data_en]
    with open(valid_translation_path, 'w') as f:
        f.writelines(data_en)

    root = xml.etree.ElementTree.parse(valid_translation_path).getroot()
    data_en = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_translation_folder, 'valid.en'), 'w') as out_file:
        out_file.write('\n'.join(data_en) + '\n')

    root = xml.etree.ElementTree.parse(os.path.join(valid_translation_folder, 'valid.en-zh.zh.sgm')).getroot()
    data_zh = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_translation_folder, 'valid.zh'), 'w') as out_file:
        out_file.write('\n'.join(data_zh) + '\n')


def build_samples():
    print('loading fasttext en word embedding')
    word_vectors = KeyedVectors.load_word2vec_format('data/wiki.en.vec')

    vocab_zh = pickle.load(open('data/vocab_train_zh.p', 'rb'))
    idx2word_zh = vocab_zh
    word2idx_zh = dict(zip(idx2word_zh, range(len(vocab_zh))))

    for usage in ['train', 'valid']:
        if usage == 'train':
            translation_path_en = os.path.join(train_translation_folder, train_translation_en_filename)
            translation_path_zh = os.path.join(train_translation_folder, train_translation_zh_filename)
            filename = 'data/samples_train.p'
        else:
            translation_path_en = os.path.join(valid_translation_folder, valid_translation_en_filename)
            translation_path_zh = os.path.join(valid_translation_folder, valid_translation_zh_filename)
            filename = 'data/samples_valid.p'

        print('loading {} texts and vocab'.format(usage))
        with open(translation_path_en, 'r') as f:
            data_en = f.readlines()

        with open(translation_path_zh, 'r') as f:
            data_zh = f.readlines()

        print('building {} samples'.format(usage))
        samples = []
        for idx in tqdm(range(len(data_en))):
            sentence_en = data_en[idx].strip().lower()
            input_en = []
            tokens = nltk.word_tokenize(sentence_en)
            for token in tokens:
                try:
                    temp = word_vectors[token]
                    word = token
                except KeyError:
                    word = unknown_word

                input_en.append(word)
            input_en.append(stop_word)

            sentence_zh = data_zh[idx].strip()
            seg_list = jieba.cut(sentence_zh)
            output_zh = []
            for word in seg_list:
                try:
                    idx = word2idx_zh[word]
                except KeyError:
                    idx = word2idx_zh[unknown_word]
                output_zh.append(idx)
            output_zh.append(word2idx_zh[stop_word])

            if len(input_en) <= Tx and len(output_zh) <= Ty:
                samples.append({'input': list(input_en), 'output': list(output_zh)})
        with open(filename, 'wb') as f:
            pickle.dump(samples, f)
        print('{} {} samples created at: {}.'.format(len(samples), usage, filename))


if __name__ == '__main__':
    ensure_folder('data')

    if not os.path.isdir(train_folder):
        extract(train_folder)

    if not os.path.isdir(valid_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_folder):
        extract(test_a_folder)

    if not os.path.isdir(test_b_folder):
        extract(test_b_folder)

    if not os.path.isfile('data/vocab_train_zh.p'):
        build_train_vocab_zh()

    extract_valid_data()

    if not os.path.isfile('data/samples_train.p') or not os.path.isfile('data/samples_valid.p'):
        build_samples()
