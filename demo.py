# import the necessary packages
import pickle
import random
import time

import numpy as np
import torch

from config import device, logger, data_file, vocab_file
from transformer.transformer import Transformer

if __name__ == '__main__':
    filename = 'transformer.pt'
    print('loading {}...'.format(filename))
    start = time.time()
    model = Transformer()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
    model = model.to(device)
    model.eval()

    logger.info('loading samples...')
    start = time.time()
    with open(data_file, 'rb') as file:
        data = pickle.load(file)
        samples = data['valid']
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    logger.info('loading vocab...')
    start = time.time()
    with open(vocab_file, 'rb') as file:
        data = pickle.load(file)
        src_idx2char = data['dict']['src_idx2char']
        tgt_idx2char = data['dict']['tgt_idx2char']
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    samples = random.sample(samples, 10)

    for sample in samples:
        sentence_in = sample['in']
        sentence_out = sample['out']

        input = torch.from_numpy(np.array(sentence_in, dtype=np.long)).to(device)
        input_length = [input[0].shape[0]]
        input_length = torch.LongTensor(input_length).to(device)

        nbest_hyps = model.recognize(input=input, input_length=len(sentence_in), char_list=tgt_idx2char)
        print(nbest_hyps)

        out_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [tgt_idx2char[idx] for idx in out]
            out = ''.join(out)
            out_list.append(out)
        out = out_list[0].replace('<sos>', '').replace('<eos>', '')
        print('out: {}'.format(out))

        sentence_in = [src_idx2char[idx] for idx in sentence_in]
        sentence_in = ' '.join(sentence_in)
        sentence_out = [tgt_idx2char[idx] for idx in sentence_out]
        sentence_out = ''.join(sentence_out)
        print(sentence_in)
        print(sentence_out)
