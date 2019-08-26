import pickle

from tqdm import tqdm

from config import train_translation_en_filename, train_translation_zh_filename, valid_translation_en_filename, \
    valid_translation_zh_filename, vocab_file, maxlen_in, maxlen_out, data_file


def build_vocab(token, word2idx, idx2char):
    if not token in word2idx:
        next_index = len(word2idx)
        word2idx[token] = next_index
        idx2char[next_index] = token


def process(file, word2idx, idx2char):
    print('processing {}...'.format(file))
    with open(file, 'r', encoding='utf-8') as file:
        data = file.readlines()

    # lengths = []

    for line in tqdm(data):
        for token in line.strip():
            build_vocab(token, word2idx, idx2char)

    #     lengths.append(len(line.strip()))
    #
    # n, bins, patches = plt.hist(lengths, 50, density=True, facecolor='g', alpha=0.75)
    #
    # plt.xlabel('Lengths')
    # plt.ylabel('Probability')
    # plt.title('Histogram of Lengths')
    # plt.grid(True)
    # plt.show()


def get_data(in_file, out_file):
    print('getting data {}->{}...'.format(in_file, out_file))
    with open(in_file, 'r', encoding='utf-8') as file:
        in_lines = file.readlines()
    with open(out_file, 'r', encoding='utf-8') as file:
        out_lines = file.readlines()

    samples = []

    for i in tqdm(range(len(in_lines))):
        in_line = in_lines[i].strip()
        in_data = [src_char2idx[token] for token in in_line]

        out_line = out_lines[i].strip()
        out_data = [src_char2idx[token] for token in out_line]

        if len(in_data) < maxlen_in and len(out_data) < maxlen_out:
            samples.append({'in': in_data, 'out': out_data})
    return samples


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

    # print(src_char2idx)
    # print(tgt_char2idx)

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

    train = get_data(train_translation_en_filename, train_translation_zh_filename)
    valid = get_data(valid_translation_en_filename, valid_translation_zh_filename)

    data = {
        'train': train,
        'valid': valid
    }

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)
