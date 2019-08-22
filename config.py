import numpy as np

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

batch_size = 128
epochs = 10000
patience = 50
embedding_size = 300
vocab_size_zh = 50000
max_token_length_en = Tx = 20 + 1  # 1 is for tailing stop word
max_token_length_zh = Ty = 20 + 1  # 1 is for tailing stop word

hidden_size = 1024

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
sos_id = 0
eos_id = 1
num_train = 8852422
num_valid = 7613

train_folder = 'data/ai_challenger_translation_train_20170904.zip'
valid_folder = 'data/ai_challenger_translation_validation_20170912.zip'
train_translation_en_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.en'
train_translation_zh_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh'
valid_translation_en_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en'
valid_translation_zh_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.zh'

start_word = '<start>'
stop_word = '<end>'
unknown_word = '<unk>'
start_embedding = np.zeros((embedding_size,))
stop_embedding = np.ones((embedding_size,))
unknown_embedding = np.ones((embedding_size,)) / 2
