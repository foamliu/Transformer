import time

import torch

from config import n_src_vocab, n_tgt_vocab, sos_id, eos_id
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.transformer import Transformer
from utils import parse_args

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model']
    print(type(model))

    encoder_fn = 'encoder.pt'
    decoder_fn = 'decoder.pt'
    print('saving {} & {}...'.format(encoder_fn, decoder_fn))
    start = time.time()
    torch.save(model.encoder.state_dict(), encoder_fn)
    torch.save(model.decoder.state_dict(), decoder_fn)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(encoder_fn))
    start = time.time()

    args = parse_args()

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
    print('elapsed {} sec'.format(time.time() - start))
