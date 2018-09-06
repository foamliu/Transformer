import random
import xml
import xml.etree.ElementTree

import jieba
import numpy as np
from tqdm import tqdm

from config import train_translation_path, valid_translation_en_filename, valid_translation_zh_filename


def count_train_samples():
    print('counting train samples')

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

    length = len(eng_sen_list)
    sample_id = random.randint(0, length - 1)
    print(eng_sen_list[sample_id])
    print(chn_sen_list[sample_id])


def count_valid_samples():
    print('counting valid samples')

    ## fix parsing issues
    with open(valid_translation_en_filename, 'r') as f:
        data_en = f.readlines()
    data_en = [line.replace('&', '&amp;') for line in data_en]
    with open(valid_translation_en_filename, 'w') as f:
        f.writelines(data_en)

    with open(valid_translation_zh_filename, 'r') as f:
        data_zh = f.readlines()
    data_zh = [line.replace('<了不起的盖茨比>', '《了不起的盖茨比》') for line in data_zh]
    with open(valid_translation_zh_filename, 'w') as f:
        f.writelines(data_zh)

    root = xml.etree.ElementTree.parse(valid_translation_en_filename).getroot()
    data_en = [elem.text.strip().split('\t')[2] for elem in root.iter() if elem.tag == 'seg']
    root = xml.etree.ElementTree.parse(valid_translation_zh_filename).getroot()
    data_zh = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    print('len(data_en): ' + str(len(data_en)))
    print('len(data_zh): ' + str(len(data_zh)))

    length = len(data_en)
    sample_id = random.randint(0, length - 1)
    print(data_en[sample_id])
    print(data_zh[sample_id])


def count_train_length_en():
    import nltk
    nltk.download('punkt')
    print('counting train length en')
    with open(train_translation_path, 'r') as f:
        lines = f.readlines()

    lengthes = []
    print('scanning train data (en)')
    for line in tqdm(lines):
        tokens = line.split('\t')
        eng_sen = tokens[2].strip().lower()
        tokens = nltk.word_tokenize(eng_sen)
        length = len(tokens)
        lengthes.append(length)

    print('max_len: ' + str(np.max(lengthes)))


def count_train_length_zh():
    print('counting train length en')
    with open(train_translation_path, 'r') as f:
        lines = f.readlines()

    lengthes = []
    print('scanning train data (en)')
    for line in tqdm(lines):
        tokens = line.split('\t')
        chn_sen = tokens[3].strip().lower()
        seg_list = jieba.cut(chn_sen)
        length = len([w for w in seg_list])
        lengthes.append(length)

    print('max_len: ' + str(np.max(lengthes)))


if __name__ == '__main__':
    count_train_samples()
    count_valid_samples()
    count_train_length_en()
    count_train_length_zh()
