import argparse
import logging
import time

import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from models.luong_attention_sru import luong_attention_sru
from utils.batches import luong_attention_batches
from utils.embeddings import create_embedding_maps
from utils.load_and_preprocessing.translation import load_train_and_val
from utils.masked_cross_entropy import masked_cross_entropy
from utils.tokens import Tokens

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--train_steps", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--input_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--eval_every", default=500, type=int)
    parser.add_argument("--eval_batch_size", default=10, type=int)
    parser.add_argument("--n_layers", default=1, type=int)
    parser.add_argument("--n_epochs", default=25, type=int)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--score_function", default="general")
    parser.add_argument("--teacher_forcing_ratio", default=0.5)
    parser.add_argument("--decoder_learning_ratio", default=5.)
    parser.add_argument("--clip_norm", default=50.0)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--debug_restrict_data", type=int)
    parser.add_argument("--different_vocab", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(level=log_level)
    return args


def main():
    args = parse_args()

    train, val = load_train_and_val(args.train_dir, args.debug_restrict_data)

    logging.info("Creating embedding maps")
    encoder_embedding_map, \
    decoder_embedding_map, \
    encoder_embedding_matrix, \
    decoder_embedding_matrix = create_embedding_maps(train, val, args.hidden_size, args.different_vocab)

    encoder = luong_attention_sru.EncoderRNN(args, encoder_embedding_matrix)
    decoder = luong_attention_sru.LuongAttnDecoderRNN(args.hidden_size,
                                                        args.input_size,
                                                        decoder_embedding_map.n_words,
                                                        args.n_layers,
                                                        decoder_embedding_matrix,
                                                        args.dropout,
                                                        args.use_cuda)

    if args.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate * args.decoder_learning_ratio)

    logging.info("Starting training")
    run_train(args.n_epochs,
              args.batch_size,
              args.eval_batch_size,
              args.eval_every,
              train,
              val,
              encoder_embedding_map,
              decoder_embedding_map,
              encoder,
              decoder,
              encoder_optimizer,
              decoder_optimizer,
              args.clip_norm,
              args.use_cuda,
              )


def run_train(n_epochs,
              batch_size,
              eval_batch_size,
              eval_every,
              train,
              val,
              encoder_embedding_map,
              decoder_embedding_map,
              encoder,
              decoder,
              encoder_optimizer,
              decoder_optimizer,
              clip_norm,
              use_cuda,
              ):
    for i in range(n_epochs):
        logging.info(f"EPOCH: {i+1}")
        for j, batch in enumerate(tqdm(luong_attention_batches(train,
                                                               encoder_embedding_map,
                                                               decoder_embedding_map,
                                                               batch_size,
                                                               use_cuda=use_cuda,
                                                               ))):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # max input length, batch size
            source_var = batch["source_var"]

            # batch size
            source_lengths = batch["source_lengths"]

            # max target length, batch size
            target_var = batch["target_var"]

            # batch size
            target_lengths = batch["target_lengths"]

            batch_size = len(target_lengths)

            logging.debug("Encoding")
            tic = time.time()

            # encoder_outputs: max input length, batch size, hidden size
            # encoder_hidden: num_layers, batch size, hidden size
            encoder_outputs, encoder_hidden = encoder(source_var,
                                                      source_lengths)
            toc = time.time()
            logging.debug(f"Seconds take to encode: {round(toc-tic,2)}")

            logging.debug("Decoding")
            tic = time.time()

            decoder_input = Variable(torch.LongTensor([Tokens.SOS_token] * batch_size))
            decoder_hidden = encoder_hidden
            if use_cuda:
                decoder_input = decoder_input.cuda()
                decoder_hidden = decoder_hidden.cuda()


            decoder_inputs = torch.cat([decoder_input.view(1, -1), target_var[:-1]], 0)
            all_decoder_outputs, decoder_hidden, attn_weights = decoder(decoder_inputs,
                                                                        target_lengths,
                                                                        decoder_hidden,
                                                                        encoder_outputs,
                                                                        source_lengths)

            toc = time.time()
            loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(),
                                        target_var.transpose(0, 1).contiguous(),
                                        target_lengths,
                                        use_cuda=use_cuda)
            logging.debug(f"Time taken for decode: {round(toc-tic, 2)}")
            logging.debug("Backpropagating")

            tic = time.time()
            loss.backward()
            toc = time.time()
            logging.debug(f"Seconds taken for backpropagation: {round(toc-tic, 2)}")

            # Clip gradients
            logging.debug("Clipping Gradients")
            _ = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip_norm)
            _ = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip_norm)

            logging.debug("Updating Weights")
            encoder_optimizer.step()
            decoder_optimizer.step()

            if j % eval_every == 0:
                logging.info("TRAINING SET!")
                run_eval(encoder_embedding_map,
                         decoder_embedding_map,
                         encoder,
                         decoder,
                         train,
                         eval_batch_size,
                         use_cuda=use_cuda)

                logging.info("EVALUATION SET!")
                run_eval(encoder_embedding_map,
                         decoder_embedding_map,
                         encoder,
                         decoder,
                         val,
                         eval_batch_size,
                         use_cuda=use_cuda)


            logging.info(f"LOSS: {loss.data[0]}")


def run_eval(encoder_embedding_map,
             decoder_embedding_map,
             encoder,
             decoder,
             val,
             eval_batch_size,
             use_cuda=False):

    batch = next(luong_attention_batches(val,
                                         encoder_embedding_map,
                                         decoder_embedding_map,
                                         bucket=False,
                                         batch_size=eval_batch_size,
                                         use_cuda=use_cuda))

    # Disable to avoid dropout
    encoder.train(False)
    decoder.train(False)

    source_var = batch["source_var"]  # max input length, batch size
    source_lengths = batch["source_lengths"]  # batch size
    target_var = batch["target_var"]  # max target length, batch size
    target_lengths = batch["target_lengths"]  # batch size

    encoder_outputs, encoder_hidden = encoder(source_var,
                                              source_lengths)

    # (1, eval_batch_size)
    decoder_inputs = Variable(torch.LongTensor([Tokens.SOS_token] * eval_batch_size)).unsqueeze(0)
    decoder_hidden = encoder_hidden
    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, eval_batch_size, decoder.output_size))

    if use_cuda:
        decoder_inputs = decoder_inputs.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # decoder_attentions = torch.zeros(eval_batch_size, max_target_length, max_target_length)

    logging.info("Decoding")

    for t in tqdm(range(max_target_length)):
        decoder_output, decoder_hidden, attn_weights = decoder(decoder_inputs,
                                                               target_lengths,
                                                               decoder_hidden,
                                                               encoder_outputs,
                                                               source_lengths)
        # num_time_step (will be 1), batch_size, output_vocab_size
        all_decoder_outputs[t] = decoder_output

        # num_time_step (will be 1), batch_size, 1
        _, top_i = decoder_output.data.topk(1)
        decoder_inputs = Variable(top_i).squeeze(0).transpose(1, 0)

    format_eval_output(encoder_embedding_map,
                       decoder_embedding_map,
                       source_var,
                       target_var,
                       all_decoder_outputs)


def format_eval_output(encoder_embedding_map,
                       decoder_embedding_map,
                       inp,
                       target,
                       all_decoder_outputs):
    inp = inp.cpu()
    target = target.cpu()
    _, batch_top_i = all_decoder_outputs.topk(1)
    batch_top_i = batch_top_i.cpu().squeeze(-1).transpose(1, 0).data.numpy()

    input_sentences = []
    target_sentences = []
    decoded_sentences = []

    for var in inp.transpose(1, 0).data.numpy():
        input_sentences.append(encoder_embedding_map.get_sentence_from_indexes(var))
    for var in target.transpose(1, 0).data.numpy():
        target_sentences.append(decoder_embedding_map.get_sentence_from_indexes(var))
    for indexes in batch_top_i:
        sentence = decoder_embedding_map.get_sentence_from_indexes(indexes)
        trunc_sentence = []
        for word in sentence:
            if word != "<EOS>":
                trunc_sentence.append(word)
            else:
                break
        decoded_sentences.append(trunc_sentence)

    for source, target, decode in zip(input_sentences, target_sentences, decoded_sentences):
        logging.info(f"\nSource:{' '.join(source)}\nTarget:{' '.join(target)}\nDecode:{' '.join(decode)}")


if __name__ == "__main__":
    main()
