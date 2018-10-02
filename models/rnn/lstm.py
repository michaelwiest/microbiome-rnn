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
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
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
    def __init__(self, hidden_dim, otu_handler,
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
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, LSTM_in_size),
            # nn.BatchNorm1d(self.otu_handler.num_strains)
            nn.Tanh()
        )
        # Expansion layers from reduced number to raw number of strains
        self.after_lstm = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, self.otu_handler.num_strains),
            # nn.BatchNorm1d(self.otu_handler.num_strains)
            # nn.Tanh()
        )


        # Non-torch inits.
        self.use_gpu = use_gpu
        self.hidden = None

    def __forward(self, input_data):
        # input_data is shape: sequence_size x batch x num_strains
        input_data = self.before_lstm(input_data)
        output, self.hidden = self.lstm(input_data, self.hidden)
        output = self.after_lstm(output)

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

    def do_training(self, slice_len, batch_size, epochs, lr, samples_per_epoch,
              slice_incr=None, save_params=None):
        np.random.seed(1)

        self.batch_size = batch_size
        self.__init_hidden()

        if self.use_gpu:
            self.cuda()

        loss_function = nn.MSELoss()
        # TODO: Try Adagrad & RMSProp
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # For logging the data for plotting
        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(epochs):
            iterate = 0

            # For a specified number of examples per epoch. This basically
            # decides how many examples to do before increasing the length
            # of the slice of data fed to the LSTM.
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                self.train() # Put the network in training mode.

                # Select a random sample from the data handler.
                data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                           slice_len)

                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains
                data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                targets = add_cuda_to_variable(targets, self.use_gpu)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                # Do a forward pass of the model.
                outputs = self.__forward(data).transpose(0, 1).transpose(1, 2)

                # For this round set our loss to zero and then compare
                # accumulated losses for all of the batch examples.
                # Finally step with the optimizer.
                loss = 0
                for bat in range(batch_size):
                    loss += loss_function(outputs[bat, :], targets[bat, :])
                loss.backward()
                optimizer.step()
                iterate += 1

            # Basically do the same as above, but with validation data.
            # Also don't have the optimizer step at all.
            self.eval()
            print('Loss ' + str(loss.data[0] / self.batch_size))
            data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                  slice_len, train=False)

            data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
            targets = add_cuda_to_variable(targets, self.use_gpu)

            self.__init_hidden()
            outputs_val = self.__forward(data)
            outputs_val = outputs_val.transpose(0, 1).transpose(1, 2)

            # Get the loss associated with this validation data.
            val_loss = 0
            for bat in range(self.batch_size):
                val_loss += loss_function(outputs_val[bat, :], targets[bat, :])
            val_loss_vec.append(val_loss.data[0] / self.batch_size)
            train_loss_vec.append(loss.data[0] / self.batch_size)


            print('Completed Epoch ' + str(epoch))

            # If we want to increase the slice of the data that we are
            # training on then do so.
            if slice_incr is not None:
                # Handle percentage increase or integer increase.
                if slice_incr >= 1.0:
                    slice_len += slice_incr
                else:
                    slice_len += slice_len * slice_incr
                # Make sure that the slice doesn't get longer than the
                # amount of data we can feed to it. Could handle this with
                # padding characters.
                slice_len = min(self.otu_handler.min_len - 1, int(slice_len))
                print('Increased slice length to: {}'.format(slice_len))

            # Save the model and logging information.
            if save_params is not None:
                torch.save(self.state_dict(), save_params[0])
                with open(save_params[1], 'w+') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(train_loss_vec)
                    writer.writerow(val_loss_vec)
                print('Saved model state to: {}'.format(save_params[0]))

        return train_loss_vec, val_loss_vec

    '''
    Function for letting the LSTM "dream" up new data. Given a primer it will
    generate examples for as long as specified.

    The "serial" argument determines wheter or not examples are fed one
    at a time to the LSTM with no gradient zeroing, or fed as a batch
    and then zeroed everytime. serial=True has been giving better results.
    '''
    def daydream(self, primer, predict_len=100, window_size=20,
                 serial=True):
        self.batch_size = 1
        self.__init_hidden()

        predicted = primer
        # If we do it the serial way, then prime the model with all examples
        # up to the most recent one.
        if serial:
            inp = add_cuda_to_variable(predicted[:, :-1], self.use_gpu) \
                .unsqueeze(-1) \
                .transpose(0, 2) \
                .transpose(0, 1)[-window_size:, :, :]
            _ = self.__forward(inp)
        for p in range(predict_len):
            if serial:
                inp = add_cuda_to_variable(predicted[:, -1], self.use_gpu)
            else:
                inp = add_cuda_to_variable(predicted, self.use_gpu)
            inp = inp.transpose(0, 2).transpose(0, 1)[-window_size:, :, :]
            # Only keep the last predicted value.
            output = self.__forward(inp)[-1].transpose(0, 1).data.numpy()
            # Add the new value to the values to be passed to the LSTM.
            predicted = np.concatenate((predicted, output), axis=1)

            if not serial:
                self.__init_hidden()

        return predicted
