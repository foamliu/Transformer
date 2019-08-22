import pickle

from tqdm import tqdm

from config import train_translation_en_filename, train_translation_zh_filename, valid_translation_en_filename, \
    valid_translation_zh_filename, vocab_file


def build_vocab(token, word2idx, idx2char):
    if not token in word2idx:
        next_index = len(word2idx)
        word2idx[token] = next_index
        idx2char[next_index] = token


def process(file, word2idx, idx2char):
    print(file)
    with open(file, 'r', encoding='utf-8') as file:
        data = file.readlines()

    max_length = 0

    for line in tqdm(data):
        for token in line.strip():
            build_vocab(token, word2idx, idx2char)

        if len(line) > max_length:
            max_length = len(line)
    print('max_length: ' + str(max_length))


if __name__ == '__main__':
    src_char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    src_idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>'}
    tgt_char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    tgt_idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>'}

    process(train_translation_en_filename, src_char2idx, src_idx2char)
    process(train_translation_zh_filename, tgt_char2idx, tgt_idx2char)
    process(valid_translation_en_filename, src_char2idx, src_idx2char)
    process(valid_translation_zh_filename, tgt_char2idx, tgt_idx2char)

    print(len(src_char2idx))
    print(len(tgt_char2idx))

    print(src_char2idx)
    print(tgt_char2idx)

    data = {
        'dict': {
            'src_char2idx': src_char2idx,
            'src_idx2char': src_idx2char,
            'tgt_char2idx': tgt_char2idx,
            'tgt_idx2char': tgt_idx2char
        }
    }
    with open(vocab_file, 'wb') as file:
        pickle.dump(data, file)
