# import the necessary packages
import pickle
import time

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

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

    # samples = random.sample(samples, 10)
    bleu_scores = []

    for sample in tqdm(samples):
        sentence_in = sample['in']
        sentence_out = sample['out']

        input = torch.from_numpy(np.array(sentence_in, dtype=np.long)).to(device)
        input_length = torch.LongTensor([len(sentence_in)]).to(device)

        sentence_in = ' '.join([src_idx2char[idx] for idx in sentence_in])
        sentence_out = ''.join([tgt_idx2char[idx] for idx in sentence_out])
        sentence_out = sentence_out.replace('<sos>', '').replace('<eos>', '')
        # print('< ' + sentence_in)
        # print('= ' + sentence_out)

        try:
            with torch.no_grad():
                nbest_hyps = model.recognize(input=input, input_length=input_length, char_list=tgt_idx2char)
                # print(nbest_hyps)
        except RuntimeError:
            print('sentence_in: ' + sentence_in)
            continue

        score_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [tgt_idx2char[idx] for idx in out]
            out = ''.join(out)
            out = out.replace('<sos>', '').replace('<eos>', '')
            reference = list(sentence_out)
            hypothesis = list(out)
            score = sentence_bleu([reference], hypothesis)
            score_list.append(score)

        bleu_scores.append(max(score_list))

    print('len(bleu_scores): ' + str(len(bleu_scores)))
    print('np.max(bleu_scores): ' + str(np.max(bleu_scores)))
    print('np.min(bleu_scores): ' + str(np.min(bleu_scores)))
    print('np.mean(bleu_scores): ' + str(np.mean(bleu_scores)))

    import numpy as np
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # the histogram of the data
    n, bins, patches = plt.hist(bleu_scores, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Scores')
    plt.ylabel('Probability')
    plt.title('BLEU Scores')
    plt.grid(True)
    plt.show()
