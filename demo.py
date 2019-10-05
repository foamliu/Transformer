# import the necessary packages
import pickle
import random
import time

import torch

from config import n_src_vocab, n_tgt_vocab, sos_id, eos_id, logger, data_file
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
    logger.info('elapsed {} sec'.format(time.time() - start))

    logger.info('loading samples...')
    start = time.time()
    with open(data_file, 'rb') as file:
        data = pickle.load(file)
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f}'.format(elapsed))

    samples = data['valid']
    samples = random.sample(samples, 10)

    for sample in samples:
        print(sample)
