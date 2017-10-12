import torch
import numpy as np
from utils.masked_cross_entropy import sequence_mask


def masked_rnn_cell(rnn_cell, embedded_seq, input_lengths, hidden, bidirectional, use_cuda):

    outputs, _ = rnn_cell(embedded_seq, hidden)
    current_batch_size = len(input_lengths)
    if bidirectional:
        hidden_size = int(outputs.size()[-1]/2)
    else:
        hidden_size = outputs.size()[-1]
    max_length = max(input_lengths)
    mask = sequence_mask(input_lengths, use_cuda=use_cuda).float()
    mask = mask.transpose(0, 1).unsqueeze(-1).expand(max_length,
                                                     current_batch_size,
                                                     hidden_size * 2 if bidirectional else hidden_size)
    # output_ret = outputs * mask
    # last_time_step_indices = torch.from_numpy(np.array(input_lengths) - 1).long()

    output_ret = outputs
    last_time_step_indices = torch.from_numpy(np.ones(current_batch_size) * (max(input_lengths) - 1)).long()
    batch_extractor_indices = torch.from_numpy(np.arange(current_batch_size)).long()
    if use_cuda:
        last_time_step_indices = last_time_step_indices.cuda()
        batch_extractor_indices = batch_extractor_indices.cuda()
    hidden_ret = output_ret[last_time_step_indices, batch_extractor_indices, :]
    if bidirectional:
        hidden_ret = hidden_ret[:, hidden_size:] + hidden_ret[:, :hidden_size]
        output_ret = output_ret[:, :, hidden_size:] + output_ret[:, :,:hidden_size]
    return output_ret, hidden_ret.unsqueeze(0)





