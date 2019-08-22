import os
import numpy as np
import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

batch_size = 128
epochs = 10000
patience = 50
embedding_size = 300
vocab_size_zh = 50000
max_token_length_en = Tx = 20 + 1   # 1 is for tailing stop word
max_token_length_zh = Ty = 20 + 1   # 1 is for tailing stop word

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
train_translation_folder = os.path.join(train_folder, 'translation_train_20170912')
valid_translation_folder = os.path.join(valid_folder, 'translation_validation_20170912')
train_translation_path = 'data/ai_challenger_MTEnglishtoChinese_trainingset_20180827/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt'
valid_translation_en_filename = 'data/ai_challenger_MTEnglishtoChinese_validationset_20180823/ai_challenger_MTEnglishtoChinese_validationset_20180823_en.sgm'
valid_translation_zh_filename = 'data/ai_challenger_MTEnglishtoChinese_validationset_20180823/ai_challenger_MTEnglishtoChinese_validationset_20180823_zh.sgm'

start_word = '<start>'
stop_word = '<end>'
unknown_word = '<unk>'
start_embedding = np.zeros((embedding_size,))
stop_embedding = np.ones((embedding_size,))
unknown_embedding = np.ones((embedding_size,)) / 2
