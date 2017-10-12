import torch
import logging
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size, n_layers, dropout, word_embedding_matrix, rnn_cell, use_cuda):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = word_embedding_matrix
        self.use_cuda = use_cuda

        if rnn_cell == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, hidden, input_lengths):
        """
        input_seqs : (Max input length, batch_size)
        input_lengths: (batch_size)
        """

        # Max input length, batch size, hidden_size
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)

        # Max input length, batch_size, hidden_size, we add the backward and forward
        # hidden states together
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # Get the forwards and backwards hidden states
        hidden_layers = []
        for i in range(self.n_layers):
            hidden_layers.append((hidden[i * 2, :, :] + hidden[(i * 2) + 1, :, :]).unsqueeze(0))
        return outputs, torch.cat(hidden_layers, 0)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
        if self.use_cuda: hidden = hidden.cuda()
        return hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size, use_cuda=False):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        """
        hidden : 1, batch_size, hidden_size
        encoder_outputs : max input length, batch_size, hidden_size
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(this_batch_size, max_len))

        if self.use_cuda:
            attn_energies = attn_energies.cuda()

        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b, :], encoder_outputs[i, b].unsqueeze(0))

        # Batch size, 1, max input length
        return F.softmax(attn_energies.unsqueeze(1))

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output).view(-1)
            energy = hidden.view(-1).dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, score_function, hidden_size, input_size, output_size,
                 n_layers, dropout , word_embedding_matrix, use_cuda):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.score_function = score_function
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # Define layers
        self.embedding = word_embedding_matrix
        self.gru = nn.GRU(self.input_size, self.hidden_size, n_layers, dropout=self.dropout)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # Choose attention models
        if score_function != 'none':
            self.attn = Attn(score_function, hidden_size, use_cuda=use_cuda)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        """
        input_seq : batch_size
        hidden : hidden_size, batch_size
        encoder_outputs : max input length, batch_size, hidden_size
        """
        # Note: we run this one step at a time

        # logging.debug(f"input_seq:\n{input_seq}")
        # logging.debug(f"last_hidden:\n{last_hidden}")
        # logging.debug(f"encoder_outputs:\n{encoder_outputs}")

        batch_size = input_seq.size(0)

        # (batch size, hidden_size)
        embedded = self.embedding(input_seq)

        # (1, batch size, hidden_size) add another dimension so that it works with
        # the GRU
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        logging.debug(f"embedded:\n{embedded}")
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)
        logging.debug(f"rnn_output:\n{rnn_output}")
        logging.debug(f"hidden:\n{hidden}")


        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average

        # batch size, max input length
        attn_weights = self.attn(rnn_output, encoder_outputs)
        logging.debug(f"attn_weights:\n{attn_weights}")

        # (batch_size, 1, max input length) @ batch_size, max input length, hidden size
        # note that we use this convention here to take advantage of the bmm function
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without fftmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights