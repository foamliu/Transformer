import os
import zipfile

from config import train_folder, valid_folder, test_a_folder
from utils import ensure_folder


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    ensure_folder('data')

    if not os.path.isdir(train_folder):
        extract(train_folder)

    if not os.path.isdir(valid_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_folder):
        extract(test_a_folder)

    # if not os.path.isdir(test_b_folder):
    #     extract(test_b_folder)
    #
    # if not os.path.isfile('data/vocab_train_zh.p'):
    #     build_train_vocab_zh()
    #
    # extract_valid_data()
    #
    # if not os.path.isfile('data/samples_train.p') or not os.path.isfile('data/samples_valid.p'):
    #     build_samples()
