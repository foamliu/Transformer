import os
import pickle
import xml.etree.ElementTree
from collections import Counter

import jieba
import nltk
from gensim.models import KeyedVectors
from tqdm import tqdm

from config import start_word, stop_word, unknown_word, Tx, Ty, vocab_size_zh
from config import train_translation_folder, train_translation_path
from config import valid_translation_folder, valid_translation_zh_filename, valid_translation_en_filename
from config import train_translation_en_filename, train_translation_zh_filename


def build_train_vocab_zh():
    translation_path = os.path.join(train_translation_folder, train_translation_path)

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


''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants


def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts


def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2  # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')


if __name__ == '__main__':
    main()
