import os
import xml
from collections import Counter

import nltk
from gensim.models import KeyedVectors
from tqdm import tqdm

from config import train_translation_path, valid_translation_en_filename, valid_translation_zh_filename


def count_train_samples():
    print('count train samples')

    with open(train_translation_path, 'r') as f:
        lines = f.readlines()

    eng_sen_list = []
    chn_sen_list = []
    print('scanning train data (zh)')
    for line in tqdm(lines):
        tokens = line.split('\t')
        eng_sen = tokens[2].strip()
        chn_sen = tokens[3].strip()
        eng_sen_list.append(eng_sen)
        chn_sen_list.append(chn_sen)
    print('len(eng_sen_list): ' + str(len(eng_sen_list)))
    print(eng_sen_list[0])
    print(chn_sen_list[0])


def count_valid_samples():
    print('counting valid samples')

    with open(valid_translation_en_filename, 'r') as f:
        data_en = f.readlines()
    data_en = [line.replace(' & ', ' &amp; ') for line in data_en]
    with open(valid_translation_en_filename, 'w') as f:
        f.writelines(data_en)

    root = xml.etree.ElementTree.parse(valid_translation_en_filename).getroot()
    data_en = [elem.text.strip().split('\t')[2] for elem in root.iter() if elem.tag == 'seg']
    root = xml.etree.ElementTree.parse(valid_translation_zh_filename).getroot()
    data_zh = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    print('len(data_en): ' + str(len(data_en)))
    print('len(data_zh): ' + str(len(data_zh)))
    print(data_en[0])
    print(data_zh[0])


def train_length_en():
    print('train_length_en')
    translation_path = os.path.join(train_translation_folder, train_translation_en_filename)
    with open(translation_path, 'r') as f:
        data = f.readlines()

    max_len = 0
    lengthes = []
    print('scanning train data (en)')
    for sentence in tqdm(data):
        tokens = nltk.word_tokenize(sentence.strip().lower())
        length = len(tokens)
        lengthes.append(length)
        if length > max_len:
            max_len = length

    print('max_len: ' + str(max_len))

    counter_length = Counter(lengthes)
    total_count = 0
    common = counter_length.most_common()
    covered_count = 0
    for i in range(1, max_len + 1):
        count = [item[1] for item in common if item[0] == i]
        if count:
            covered_count += count[0]
        print('{} -> {}'.format(i, covered_count / total_count))


def train_vocab_en():
    print('train_vocab_en')
    print('loading fasttext en word embedding')
    word_vectors = KeyedVectors.load_word2vec_format('data/wiki.en.vec')
    translation_path = os.path.join(train_translation_folder, train_translation_en_filename)

    with open(translation_path, 'r') as f:
        data = f.readlines()

    vocab = []

    print('building {} train vocab (en)')
    for sentence in tqdm(data):
        tokens = nltk.word_tokenize(sentence.strip().lower())
        for token in tokens:
            vocab.append(token)

    counter_vocab = Counter(vocab)

    total_count = 0
    covered_count = 0
    for word in tqdm(counter_vocab.keys()):
        total_count += counter_vocab[word]
        try:
            temp = word_vectors[word]
            covered_count += counter_vocab[word]
        except (NameError, KeyError):
            pass

    print('count of words in text (en): ' + str(len(vocab)))
    print('fasttext vocab size (en): ' + str(len(word_vectors.vocab)))
    print('coverage: ' + str(covered_count / total_count))


if __name__ == '__main__':
    count_train_samples()
    count_valid_samples()
    # train_length_en()
    # train_vocab_en()
