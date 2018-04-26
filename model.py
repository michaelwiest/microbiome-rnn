from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
import numpy as np
from helpers.model_helper import *
import csv


'''
Model for predicting OTU counts of microbes given historical data.
Uses fully connected layers and an LSTM (could use 1d convolutions in the
future for better accuracy).

As of now does not have a "dream" function for generating predictions from a
seeded example.
'''
class LSTM(nn.Module):
    def __init__(self, hidden_dim, bs, otu_handler,
                 use_gpu=False,
                 LSTM_in_size=None):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.otu_handler = otu_handler
        if LSTM_in_size is None:
            LSTM_in_size = self.otu_handler.num_strains
        self.lstm = nn.LSTM(LSTM_in_size, hidden_dim, 1)

        # Compression layers from raw number of inputs to reduced number
        self.before_lstm = nn.Sequential(
            nn.Linear(self.otu_handler.num_strains, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, LSTM_in_size),
            # nn.BatchNorm1d(self.otu_handler.num_strains)
            nn.Tanh()
        )
        # Expansion layers from reduced number to raw number of strains
        self.after_lstm = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.otu_handler.num_strains),
            # nn.BatchNorm1d(self.otu_handler.num_strains)
            nn.Tanh()
        )


        # Non-torch inits.
        self.batch_size = bs
        self.use_gpu = use_gpu
        self.hidden = self.__init_hidden()

    def __forward(self, input_data):
        # input_data is shape: sequence_size x batch x num_strains
        input_data = self.before_lstm(input_data)
        output, self.hidden = self.lstm(input_data, self.hidden)
        # output = output.transpose(1, 2)
        output = self.after_lstm(output)
        # print(output.size())
        return output

    def __init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(1,
                                                self.batch_size,
                                                self.hidden_dim).cuda()),
                           Variable(torch.zeros(1,
                                                self.batch_size,
                                                self.hidden_dim).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(1,
                                                self.batch_size,
                                                self.hidden_dim)),
                           Variable(torch.zeros(1,
                                                self.batch_size,
                                                self.hidden_dim))
                           )

    def train(self, slice_len, batch_size, epochs, lr, samples_per_epoch,
              slice_incr=None, save_params=None):
        np.random.seed(1)

        self.batch_size = batch_size

        if self.use_gpu:
            self.cuda()

        loss_function = nn.MSELoss()
        # Try Adagrad & RMSProp
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # For logging the data for plotting
        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(epochs):
            iterate = 0

            '''
            For a specified number of examples per epoch.
            '''
            for iterate in range(int(samples_per_epoch / self.batch_size)):

                # Select a random sample from the data handler.
                data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                      slice_len)
                data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                print(data.size())
                targets = add_cuda_to_variable(targets, self.use_gpu)#.transpose(1, 2).transpose(0, 1)
                # Pytorch accumulates gradients. We need to clear them out before each instance
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                outputs = self.__forward(data)
                # print(outputs.size())
                outputs = outputs[-1, :, :]
                # print(outputs.size())
                loss = 0
                for bat in range(batch_size):
                    # loss += loss_function(outputs[:, bat, :], targets[:, bat, :].squeeze(1))
                    loss += loss_function(outputs[bat, :], targets[bat, :])
                    # print(loss.size())
                loss.backward()
                optimizer.step()

                if iterate % 1000 == 0:
                    print('Loss ' + str(loss.data[0] / self.batch_size))
                    data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                          slice_len, train=False)
                    data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                    targets = add_cuda_to_variable(targets, self.use_gpu)#.transpose(1, 2).transpose(0, 1)

                    self.__init_hidden()
                    outputs_val = self.__forward(data)
                    outputs_val = outputs_val[-1, :, :]
                    val_loss = 0
                    for bat in range(self.batch_size):
                        # val_loss += loss_function(outputs_val[:, bat, :], targets[:, bat, :].squeeze(1))
                        val_loss += loss_function(outputs_val[bat, :], targets[bat, :])
                    val_loss_vec.append(val_loss.data[0] / self.batch_size)
                    train_loss_vec.append(loss.data[0] / self.batch_size)
                    print('Validataion Loss ' + str(val_loss.data[0]/batch_size))
                iterate += 1
            print('Completed Epoch ' + str(epoch))

            if slice_incr is not None:
                if slice_incr >= 1.0:
                    slice_len += slice_incr
                else:
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

    def daydream(self, primer, T, predict_len=100, window_size=10,
                 init_hidden=True):
        self.batch_size = 1
        self.__init_hidden()

        predicted = primer
        for p in range(predict_len):
            inp = add_cuda_to_variable(predicted, self.use_gpu) \
                    .unsqueeze(-1) \
                    .transpose(0, 2) \
                    .transpose(0, 1)[-window_size:, :, :]
            output = self.__forward(inp)[-1].transpose(0,1).data.numpy()
            predicted = np.concatenate((predicted, output), axis=1)
            if init_hidden:
                self.__init_hidden()

        return predicted
