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

class FFN(nn.Module):
    def __init__(self, hidden_dim, bs, otu_handler, slice_len, conv_filters,
                 kernel_size=3,
                 use_gpu=False):
        super(FFN, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.otu_handler = otu_handler
        self.slice_len = slice_len
        self.batch_size = bs
        self.use_gpu = use_gpu
        # self.hidden = self.__init_hidden()

        # Compression layers from raw number of inputs to reduced number
        self.__set_layers()
        self.final_layer = nn.Linear(self.hidden_dim *
                                     int(self.slice_len / 2),
                                     self.otu_handler.num_strains)

    def __forward(self, input_data):
        # input_data is shape: sequence_size x batch x num_strains
        after_conv = self.conv_layers(input_data.transpose(0, 1).transpose(1, 2))
        after_linear = self.linear_layers(after_conv.transpose(1, 2).transpose(1, 0))
        return self.final_layer(after_linear.view(self.batch_size, -1))


    '''
    Batch norm depends on the batch size which can be changed when doing
    daydreaming. so need to set it like this.
    '''
    def __set_layers(self):
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.otu_handler.num_strains,
                      self.conv_filters,
                      self.kernel_size, padding=1)
            , nn.ReLU()
            , nn.Conv1d(self.conv_filters,
                        self.conv_filters * 2,
                        self.kernel_size, padding=1)
            , nn.ReLU()
            , nn.Conv1d(self.conv_filters * 2,
                        self.conv_filters * 4,
                        self.kernel_size, padding=1)
            , nn.ReLU()
            , nn.Conv1d(self.conv_filters * 4,
                        self.otu_handler.num_strains,
                        1)
            , nn.ReLU()
            , nn.MaxPool1d(2, 2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.otu_handler.num_strains, self.hidden_dim),
            nn.BatchNorm1d(self.batch_size),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.batch_size),
            nn.ReLU()
            )

    def do_training(self, batch_size, epochs, lr, samples_per_epoch,
              slice_incr=None, save_params=None):

        np.random.seed(1)
        self.train()

        self.batch_size = batch_size

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

                # Select a random sample from the data handler.
                data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                                   self.slice_len)

                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains
                data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                targets = add_cuda_to_variable(targets, self.use_gpu)
                self.zero_grad()

                # Do a forward pass of the model.
                # Only keep the last prediction as that is what we are
                # comparing against. Essentially treating everything up to
                # that as a primer.
                outputs = self.__forward(data)

                # For this round set our loss to zero and then compare
                # accumulate losses for all of the batch examples.
                # Finally step with the optimizer.
                loss = 0
                for bat in range(batch_size):
                    loss += loss_function(outputs[bat, :], targets[bat, :])
                loss.backward()
                optimizer.step()

                # Basiaclly do the same as above, but with validation data.
                # Also don't have the optimizer step at all.
                if iterate % 1000 == 0:
                    self.eval()
                    print('Loss ' + str(loss.data[0] / self.batch_size))
                    data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                          self.slice_len, train=False)
                    data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                    targets = add_cuda_to_variable(targets, self.use_gpu)

                    outputs_val = self.__forward(data)

                    # Get the loss associated with this validation data.
                    val_loss = 0
                    for bat in range(self.batch_size):
                        val_loss += loss_function(outputs_val[bat, :], targets[bat, :])
                    val_loss_vec.append(val_loss.data[0] / self.batch_size)
                    train_loss_vec.append(loss.data[0] / self.batch_size)
                iterate += 1

            print('Completed Epoch ' + str(epoch))

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
    Function for letting the model "dream" up new data. Given a primer it will
    generate examples for as long as specified.
    '''
    def daydream(self, primer, predict_len=100):
        self.batch_size = 1
        self.__set_layers()
        # self.eval()

        predicted = primer
        for p in range(predict_len):
            inp = add_cuda_to_variable(predicted[:, -self.slice_len:], self.use_gpu)
            inp = inp.unsqueeze(-1).transpose(0, 2).transpose(0, 1)
            output = self.__forward(inp).transpose(0, 1).data.numpy()
            # Add the new value to the values to be passed to the LSTM.
            predicted = np.concatenate((predicted, output), axis=1)

        return predicted
