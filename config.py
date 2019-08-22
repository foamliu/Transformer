import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
d_model = 512
epochs = 10000
embedding_size = 300
hidden_size = 1024
vocab_file = 'vocab.pkl'
vocab_size = 8279  # target

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
sos_id = 0
eos_id = 1
num_train = 10000000
num_valid = 8000

train_translation_en_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.en'
train_translation_zh_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh'
valid_translation_en_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en'
valid_translation_zh_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.zh'
