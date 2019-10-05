# import the necessary packages
import pickle
import random
import time
import numpy as np
import torch

from config import n_src_vocab, n_tgt_vocab, sos_id, eos_id, logger, data_file, vocab_file
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.transformer import Transformer
from utils import parse_args

if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    encoder_fn = 'encoder.pt'
    decoder_fn = 'decoder.pt'
    logger.info('loading {} & {}...'.format(encoder_fn, decoder_fn))
    encoder = Encoder(n_src_vocab, args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id, eos_id, n_tgt_vocab,
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)
    encoder.load_state_dict(torch.load(encoder_fn))
    decoder.load_state_dict(torch.load(decoder_fn))
    model = Transformer(encoder, decoder)
    logger.info('elapsed {} seconds'.format(time.time() - start))

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

        input = np.array(sentence_in, dtype=np.long)
        input = torch.from_numpy(input)

        input_length = np.array(len(sentence_in), dtype=np.long)
        input_length = torch.from_numpy(input_length)

        nbest_hyps = model.recognize(input=input, input_length=len(sentence_in), char_list=tgt_idx2char, args=args)
        print(nbest_hyps)

        sentence_in = [src_idx2char[idx] for idx in sentence_in]
        sentence_in = ' '.join(sentence_in)
        sentence_out = [tgt_idx2char[idx] for idx in sentence_out]
        sentence_out = ''.join(sentence_out)
        print(sentence_in)
        print(sentence_out)
