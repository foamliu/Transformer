import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
d_model = 512
epochs = 10000
embedding_size = 300
hidden_size = 1024
data_file = 'data.pkl'
vocab_file = 'vocab.pkl'
n_src_vocab = 192
n_tgt_vocab = 8280  # target
maxlen_in = 50
maxlen_out = 25
vocab_size_in = 5000
vocab_size_out = 5000
# Training parameters
grad_clip = 1.0  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
num_train = 9008709
num_valid = 7878

train_translation_en_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.en'
train_translation_zh_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh'
valid_translation_en_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en'
valid_translation_zh_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.zh'
