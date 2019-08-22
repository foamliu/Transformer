import pickle

from torch.utils.data import Dataset

from config import train_translation_en_filename, train_translation_zh_filename, valid_translation_en_filename, \
    valid_translation_zh_filename
from config import vocab_file
from utils import text_to_sequence


def get_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    data = [line.strip() for line in data]
    return data


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

        self.samples = data[split]
        print('loading {} {} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        src_text = self.src[i]
        src_text = text_to_sequence(src_text, self.src_char2idx)
        tgt_text = self.dst[i]
        tgt_text = text_to_sequence(tgt_text, self.tgt_char2idx)

        return src_text, tgt_text

    def __len__(self):
        return len(self.src)


if __name__ == "__main__":
    from utils import sequence_to_text

    valid_dataset = AiChallenger2017Dataset('valid')
    print(valid_dataset[0])

    with open(vocab_file, 'rb') as file:
        data = pickle.load(file)

    src_char2idx = data['dict']['src_char2idx']
    src_idx2char = data['dict']['src_idx2char']
    tgt_char2idx = data['dict']['tgt_char2idx']
    tgt_idx2char = data['dict']['tgt_idx2char']

    src_text, tgt_text = valid_dataset[0]
    src_text = sequence_to_text(src_text, src_idx2char)
    src_text = ''.join(src_text)
    print('src_text: ' + src_text)

    tgt_text = sequence_to_text(tgt_text, tgt_idx2char)
    tgt_text = ''.join(src_text)
    print('tgt_text: ' + tgt_text)
