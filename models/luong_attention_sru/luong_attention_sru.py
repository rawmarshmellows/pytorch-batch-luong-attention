import torch
import logging
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils.masked_cross_entropy import sequence_mask

import sys


class EncoderRNN(nn.Module):
    def __init__(self, args, word_embedding_matrix):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.embedding = word_embedding_matrix
        self.use_cuda = args.use_cuda
        self.sru = SRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        """
        input_seqs : (Max input length, batch_size)
        input_lengths: (batch_size)
        """

        # Max input length, batch size, hidden_size
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)

        # Max input length, batch_size, hidden_size, we add the backward and forward
        # hidden states together
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        hidden = (hidden[-1, :, :] + hidden[-2, :, :]).unsqueeze(0)
        return outputs, hidden


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 input_size,
                 output_size,
                 n_layers,
                 word_embedding_matrix,
                 dropout,
                 use_cuda):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

        # Define layers
        self.embedding = word_embedding_matrix
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.attn = LuongAttention(hidden_size, use_cuda)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self,
                input_seq,
                input_lengths,
                encoder_hidden,
                encoder_outputs,
                encoder_outputs_length):
        """
        input_seq : batch_size
        hidden : hidden_size, batch_size
        encoder_outputs : max input length, batch_size, hidden_size
        """
        # Note: we run this one step at a time

        # logging.debug(f"input_seq:\n{input_seq}")
        # logging.debug(f"last_hidden:\n{last_hidden}")
        # logging.debug(f"encoder_outputs:\n{encoder_outputs}")

        # sort the input by descending order, but now we also need to sort the encoder
        # outputs w/ the same index
        sorted_index = np.argsort(input_lengths).tolist()[::-1]
        unsorted_index = np.argsort(sorted_index)
        sorted_input_seq = input_seq[:, sorted_index]
        sorted_input_lengths = np.array(input_lengths)[sorted_index]
        sorted_encoder_hidden = encoder_hidden[:, sorted_index, :]
        sorted_encoder_outputs = encoder_outputs[:, sorted_index, :]
        sorted_encoder_outputs_length = np.array(encoder_outputs_length)[sorted_index].tolist()

        # decoder input: (batch size, hidden_size)
        embedded = self.embedding(sorted_input_seq)

        packed = pack_padded_sequence(embedded, sorted_input_lengths)
        decoder_outputs, decoder_hidden = self.gru(packed, sorted_encoder_hidden)
        decoder_outputs, decoder_outputs_length = pad_packed_sequence(decoder_outputs)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average

        # batch size, max input length. if we use batch we would need to feed the attention
        # the sorted encoder outputs and sorted encoder output lengths
        attn_weights = self.attn(sorted_encoder_outputs,
                                 sorted_encoder_outputs_length,
                                 decoder_outputs,
                                 decoder_outputs_length)
        logging.debug(f"attn_weights:\n{attn_weights}")

        # attn_weights is in (batch_size, max_decoder_length, max_encoder_length)
        # encoder_outputs is (max_encoder_length, batch_size, hidden_size)
        context = torch.bmm(attn_weights, encoder_outputs.transpose(1, 0))
        concat_rep = torch.cat([decoder_outputs, context.transpose(0, 1)], -1)
        concat_output = self.concat(F.tanh(concat_rep))

        final_decoder_output = self.out(concat_output)
        # Why do we need to use .tolist() for `output` but not for `final_decoder_output`
        original_position_outputs = final_decoder_output[:, unsorted_index.tolist(), :]

        # Return final output, hidden state, and attention weights (for visualization)
        return original_position_outputs, decoder_hidden, attn_weights


class LuongAttention(nn.Module):

    def __init__(self, hidden_size, use_cuda):
        super().__init__()
        self.hidden_size = hidden_size
        self.general_weights = Variable(torch.randn(hidden_size, hidden_size))
        self.use_cuda = use_cuda
        if use_cuda:
            self.general_weights = self.general_weights.cuda()

    def forward(self,
                encoder_outputs,
                encoder_outputs_length,
                decoder_outputs,
                decoder_outputs_length):
        """

        :param encoder_outputs: max_encoder_length, batch_size, hidden_size
        :param encoder_outputs_length: batch_size
        :param decoder_outputs: max_decoder_length, batch_size, hidden_size
        :param decoder_outputs_length: batch_size
        :return: attention_aware_output
        """

        # (batch_size, max_decoder_length, hidden_size)
        decoder_outputs = torch.transpose(decoder_outputs, 0, 1)

        # (batch_size, hidden_size, max_encoder_length)
        encoder_outputs = encoder_outputs.permute(1, 2, 0)

        # (batch_size, max_encoder_length, max_decoder_length
        score = torch.bmm(decoder_outputs @ self.general_weights, encoder_outputs)

        (attention_mask,
         max_enc_outputs_length,
         max_dec_outputs_length) = self.attention_sequence_mask(encoder_outputs_length, decoder_outputs_length)
        masked_score = score + attention_mask
        weights_flat = F.softmax(masked_score.view(-1, max_enc_outputs_length))
        weights = weights_flat.view(-1, max_dec_outputs_length, max_enc_outputs_length)

        return weights

    def attention_sequence_mask(self, encoder_outputs_length, decoder_outputs_length):
        batch_size = len(encoder_outputs_length)
        max_encoder_outputs_length = max(encoder_outputs_length)
        max_decoder_outputs_length = max(decoder_outputs_length)

        encoder_sequence_mask = sequence_mask(encoder_outputs_length, use_cuda=self.use_cuda)
        encoder_sequence_mask_expand = (encoder_sequence_mask
                                        .unsqueeze(1)
                                        .expand(batch_size,
                                                max_decoder_outputs_length,
                                                max_encoder_outputs_length))

        decoder_sequence_mask = sequence_mask(decoder_outputs_length, use_cuda=self.use_cuda)
        decoder_sequence_mask_expand = (decoder_sequence_mask
                                        .unsqueeze(2)
                                        .expand(batch_size,
                                                max_decoder_outputs_length,
                                                max_encoder_outputs_length))
        attention_mask = (encoder_sequence_mask_expand *
                          decoder_sequence_mask_expand).float()
        attention_mask = (attention_mask - 1) * sys.maxsize
        return (attention_mask,
                max_encoder_outputs_length,
                max_decoder_outputs_length)
