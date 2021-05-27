from utils import init_param
from simple import simple_seq2seq
from knowledge import KnowledgeSeq2Seq
from cteg import CTEG_paper, CTEG_official
from magic import MagicSeq2Seq
from config import config_seq2seq
from tools import tools_get_logger
import torch
from torch import nn

def build_model(model_name, dataset_name, vocab_size, device, load_path=None, init_way=None, mask_idx=None):
    
    pretrained_wv_path = config_seq2seq.pretrained_wv_path[dataset_name]
    if model_name == 'paper':
        seq2seq = CTEG_paper(vocab_size=vocab_size,
                             embed_size=config_seq2seq.embedding_size,
                             pretrained_wv_path=pretrained_wv_path,
                             encoder_lstm_hidden=config_seq2seq.encoder_lstm_hidden_size,
                             encoder_bid=config_seq2seq.encoder_lstm_is_bid,
                             lstm_layer=config_seq2seq.lstm_layer_num,
                             attention_size=config_seq2seq.attention_size,
                             device=device)
    elif model_name == 'cteg':
        seq2seq = CTEG_official(embed_size=200,
                                hidden_size=512,
                                vocab_size=vocab_size,
                                device=device,
                                pretrained_wv_path=pretrained_wv_path)
    elif model_name == 'simple':
        seq2seq = simple_seq2seq(1, 512, vocab_size, 200, device,
                                 pretrained_wv_path=pretrained_wv_path)
    elif model_name == 'knowledge':
        seq2seq = KnowledgeSeq2Seq(vocab_size=vocab_size,
                                     embed_size=config_seq2seq.embedding_size,
                                     pretrained_wv_path=pretrained_wv_path,
                                     device=device,
                                     mask_idx=mask_idx)
    elif model_name == 'magic':
        seq2seq = MagicSeq2Seq(vocab_size=vocab_size,
                               embed_size=config_seq2seq.embedding_size,
                               pretrained_wv_path=pretrained_wv_path,
                               encoder_lstm_hidden=512,
                               encoder_bid=True,
                               lstm_layer=1,
                               device=device)
    else:
        raise NotImplementedError(f'{model_name} not supported')

    seq2seq.to(torch.device('cpu'))
    if load_path:
        loaded = torch.load(load_path, map_location=torch.device('cpu'))
        seq2seq_dict = seq2seq.state_dict()
        for name, param in loaded.items():
            if name not in seq2seq_dict:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            if param.shape != seq2seq_dict[name].shape:
                continue
            seq2seq_dict[name].copy_(param)
            tools_get_logger('model_builder').info(f'{name} loaded')
        # seq2seq.load_state_dict(torch.load(load_path, map_location=device))
    else:
        init_param(seq2seq, init_way=init_way)
    seq2seq.to(device)
    seq2seq.eval()
    param_num = sum(param.numel() for param in seq2seq.parameters())

    tools_get_logger('model_builder').info(f"loading pretrained {model_name} from {load_path}\nparams_all_num {param_num}")

    return seq2seq


def activate_dropout_in_train_mode(seq2seq:torch.nn.Module):
    seq2seq.eval()
    for m in seq2seq.modules():
        # print(m.__class__.__name__.lower())
        if m.__class__.__name__.lower().startswith('dropout'):
            m.train()
    return seq2seq
