import pickle

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import train_translation_en_filename, train_translation_zh_filename, valid_translation_en_filename, \
    valid_translation_zh_filename
from config import vocab_file, IGNORE_ID, eos_id
from utils import text_to_sequence


def get_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    data = [line.strip() for line in data]
    return data


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        src, tgt = elem
        max_input_len = max_input_len if max_input_len > len(src) else len(src)
        max_target_len = max_target_len if max_target_len > len(tgt) else len(tgt)

    for i, elem in enumerate(batch):
        src, tgt = elem
        input_length = len(src)
        padded_input = np.pad(src, (0, max_input_len - len(src)), 'constant', constant_values=0)
        padded_target = np.pad(tgt, (0, max_target_len - len(tgt)), 'constant', constant_values=IGNORE_ID)
        batch[i] = (padded_input, padded_target, input_length)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


class AiChallenger2017Dataset(Dataset):
    def __init__(self, split):
        with open(vocab_file, 'rb') as file:
            data = pickle.load(file)

        self.src_char2idx = data['dict']['src_char2idx']
        self.src_idx2char = data['dict']['src_idx2char']
        self.tgt_char2idx = data['dict']['tgt_char2idx']
        self.tgt_idx2char = data['dict']['tgt_idx2char']

        if split == 'train':
            self.src = get_data(train_translation_en_filename)
            self.dst = get_data(train_translation_zh_filename)
        else:
            self.src = get_data(valid_translation_en_filename)
            self.dst = get_data(valid_translation_zh_filename)

        print('loading {} pairs {} samples...'.format(len(self.src), split))

    def __getitem__(self, i):
        src_text = self.src[i]
        src_text = text_to_sequence(src_text, self.src_char2idx)
        tgt_text = self.dst[i]
        tgt_text = text_to_sequence(tgt_text, self.tgt_char2idx) + [eos_id]

        return src_text, tgt_text

    def __len__(self):
        return len(self.src)


def main():
    from utils import sequence_to_text

    valid_dataset = AiChallenger2017Dataset('valid')
    print(valid_dataset[0])

    with open(vocab_file, 'rb') as file:
        data = pickle.load(file)

    src_idx2char = data['dict']['src_idx2char']
    tgt_idx2char = data['dict']['tgt_idx2char']

    src_text, tgt_text = valid_dataset[0]
    src_text = sequence_to_text(src_text, src_idx2char)
    src_text = ''.join(src_text)
    print('src_text: ' + src_text)

    tgt_text = sequence_to_text(tgt_text, tgt_idx2char)
    tgt_text = ''.join(tgt_text)
    print('tgt_text: ' + tgt_text)


if __name__ == "__main__":
    main()
