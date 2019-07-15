import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, params):

        super(Encoder, self).__init__()

        if params.model == 'rnn':
            cell = nn.RNN(params.bme_dim, params.rnn_hidden_dim, params.num_layers, bidirectional=params.bidirectional)
        elif params.model == 'gru':
            cell = nn.GRU(params.bme_dim, params.rnn_hidden_dim, params.num_layers, bidirectional=params.bidirectional)
        elif params.model == 'lstm':
            cell = nn.LSTM(params.bme_dim, params.rnn_hidden_dim, params.num_layers, bidirectional=params.bidirectional)
        else:
            raise Exception("Model type not supported: {}".format(params.model))

        if params.bidirectional:
            fc = nn.Linear(params.rnn_hidden_dim*2, params.out_embedding_dim)
        else:
            fc = nn.Linear(params.rnn_hidden_dim, params.out_embedding_dim)

        self.rnn_cell = cell
        self.fc = fc

    def forward(self, s):       # (batch_size, seq_len, bme_embedding_dim)
        s = s.transpose(0, 1)   # (seq_len, batch_size, bme_embedding_dim)
        s, _ = self.rnn_cell(s) # (seq_len, batch_size, lstm_hidden_dim)
        s = s.contiguous()
        s = self.fc(s)          # (seq_len, batch_size, out_embedding_dim)
        return s


def cos_embedding_loss(output, target, neg_samples=False):
    '''Compute cosine similarity loss between output and target

    Args:
        output of shape (seq_len, batch_size, embedding_dim)
        target of shape (seq_len, batch_size, embedding_dim)
    '''
    if neg_samples:
        return torch.sum(torch.exp(torch.max(0, F.cosine_similarity(output, target, dim=2))))
    return torch.sum(torch.exp(1 - F.cosine_similarity(output, target, dim=2)))