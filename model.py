from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import pdb
import numpy as np
from helper import *
import csv

class LSTM(nn.Module):
    def __init__(self, hidden_dim, bs, otu_handler,
                 use_gpu=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.otu_handler = otu_handler
        self.lstm = nn.LSTM(self.otu_handler.num_strains, hidden_dim, 1)
        # The linear layer maps from hidden state space to target space
        # target space = vocab size, or number of unique characters in daa
        self.linear0 = nn.Linear(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, self.otu_handler.num_strains)
        # self.linear2 = nn.Linear()

        # Non-torch inits.
        self.batch_size = bs
        self.use_gpu = use_gpu
        self.hidden = self.__init_hidden()

    def __forward(self, input):
        # input sentence is shape: sequence_size x batch x num_strains
        output, self.hidden = self.lstm(input, self.hidden)
        output = self.linear0(output)
        output = self.linear1(output)
        return output

    def __init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))
        else:
            self.hidden =  (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def train(self, slice_len, batch_size, epochs, lr, samples_per_epoch,
              slice_incr_perc=None, save_params=None):
        np.random.seed(1)

        self.batch_size = batch_size

        if self.use_gpu:
            self.cuda()

        loss_function = nn.MSELoss()
        # Try Adagrad & RMSProp
        optimizer = optim.SGD(self.parameters(), lr=lr)

        # For logging the data for plotting
        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(epochs):
            iterate = 0

            '''
            Visit each possible example once. Can maybe tweak this to be more
            stochastic.
            '''
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                      slice_len)
                data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                # print(data.size())
                targets = add_cuda_to_variable(targets, self.use_gpu).transpose(1, 2).transpose(0, 1)
                # print(targets.size())
                # Pytorch accumulates gradients. We need to clear them out before each instance
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                outputs = self.__forward(data)
                # print(outputs.size())
                loss = 0
                for bat in range(batch_size):
                    loss += loss_function(outputs[:, bat, :], targets[:, bat, :].squeeze(1))
                loss.backward()
                optimizer.step()

                if iterate % 1000 == 0:
                    print('Loss ' + str(loss.data[0] / self.batch_size))
                    data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                          slice_len, train=False)
                    data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                    targets = add_cuda_to_variable(targets, self.use_gpu).transpose(1, 2).transpose(0, 1)

                    self.__init_hidden()
                    outputs_val = self.__forward(data)
                    val_loss = 0
                    for bat in range(self.batch_size):
                        val_loss += loss_function(outputs_val[:, bat, :], targets[:, bat, :].squeeze(1))
                    val_loss_vec.append(val_loss.data[0] / self.batch_size)
                    train_loss_vec.append(loss.data[0] / self.batch_size)
                    print('Validataion Loss ' + str(val_loss.data[0]/batch_size))
                iterate += 1
            print('Completed Epoch ' + str(epoch))

            if slice_incr_perc is not None:
                slice_len += slice_len * slice_incr_perc
                slice_len = min(self.otu_handler.min_len - 1, int(slice_len))
                print('Increased slice length to: {}'.format(slice_len))

            if save_params is not None:
                torch.save(self.state_dict(), save_params[0])
                with open(save_params[1], 'w+') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(train_loss_vec)
                    writer.writerow(val_loss_vec)
                print('Saved model state to: {}'.format(save_params[0]))

        return train_loss_vec, val_loss_vec

    # def daydream(self, primer, T, predict_len=None):
    #     vocab_size = len(self.vocab)
    #     # Have we detected an end character?
    #     end_found = False
    #     self.batch_size = 1
    #
    #     self.__init_hidden()
    #     primer_input = [self.vocab[char] for char in primer]
    #
    #     self.slice_len = len(primer_input)
    #     # build hidden layer
    #     _ = self.__forward(add_cuda_to_variable(primer_input[:-1], self.use_gpu))
    #
    #     inp = add_cuda_to_variable([primer_input[-1]], self.use_gpu)
    #     self.seq_len = 1
    #     predicted = list(primer_input)
    #     if predict_len is not None:
    #         for p in range(predict_len):
    #             output = self.__forward(inp)
    #             soft_out = custom_softmax(output.data.squeeze(), T)
    #             predicted.append(flip_coin(soft_out, self.use_gpu))
    #             inp = add_cuda_to_variable([predicted[-1]], self.use_gpu)
    #
    #     else:
    #         while end_found == False:
    #             output = self.__forward(inp)
    #             soft_out = custom_softmax(output.data.squeeze(), T)
    #             found_char = flip_coin(soft_out, self.use_gpu)
    #             predicted.append(found_char)
    #             # print(found_char)
    #             if found_char == self.vocab[self.end_char]:
    #                 end_found = True
    #             inp = add_cuda_to_variable([predicted[-1]], self.use_gpu)
    #
    #     strlist = [self.vocab.keys()[self.vocab.values().index(pred)] for pred in predicted]
    #     return (''.join(strlist).replace(self.pad_char, '')).replace(self.start_char, '').replace(self.end_char, '')
