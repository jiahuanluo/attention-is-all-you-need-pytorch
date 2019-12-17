''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator import Translator
from torch.utils.data import DataLoader

import utils


def prepare_mydataloaders(opt, device):
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = 140
    opt.src_pad_idx = data['dict']['src'].labelToIdx[Constants.PAD_WORD]
    opt.trg_pad_idx = data['dict']['tgt'].labelToIdx[Constants.PAD_WORD]
    opt.trg_bos_idx = data['dict']['tgt'].labelToIdx[Constants.BOS_WORD]
    opt.trg_eos_idx = data['dict']['tgt'].labelToIdx[Constants.EOS_WORD]
    opt.unk_idx = 1
    opt.src_vocab_size = len(data['dict']['src'].labelToIdx)
    opt.trg_vocab_size = len(data['dict']['tgt'].labelToIdx)
    # ========= Preparing Model =========#
    # if opt.embs_share_weight:
    #     assert data['dict']['src'].labelToIdx == data['dict']['tgt'].labelToIdx, \
    #         'To sharing word embedding the src/trg word2idx table shall be the same.'
    testset = utils.BiDataset(data['test'])
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=1,
                                             shuffle=False,
                                             collate_fn=utils.padding)
    return data['dict']['tgt'], testloader


def load_model(opt, device):
    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Translate bpe encoded files 
    # parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    # parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    # parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # data = pickle.load(open(opt.data_pkl, 'rb'))
    # SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    # opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    # opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    # opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    # opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    # test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    device = torch.device('cuda' if opt.cuda else 'cpu')
    TRG, test_loader = prepare_mydataloaders(opt, device)
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    with open(opt.output, 'w', encoding='utf-8') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            sec_seq = example[0].view(1, -1)
            pred_seq = translator.translate_sentence(sec_seq.to(device))
            pred_line = ' '.join(TRG.idxToLabel[idx] for idx in pred_seq)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            f.write(pred_line)

    print('[Info] Finished.')


if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
