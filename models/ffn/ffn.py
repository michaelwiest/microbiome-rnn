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

class FFN(nn.Module):
    def __init__(self, hidden_dim, bs, otu_handler, slice_len,
                 kernel_size=3,
                 use_gpu=False):
        super(FFN, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.otu_handler = otu_handler
        self.slice_len = slice_len
        self.batch_size = bs
        self.use_gpu = use_gpu
        self.__set_layers()
        self.final_layer = nn.Linear(self.hidden_dim *
                                     int(self.slice_len),
                                     self.otu_handler.num_strains)

    def forward(self, input_data):
        print(input_data.size())
        # input_data is shape: sequence_size x batch x num_strains
        # after_conv = self.conv_layers(input_data.transpose(1, 2))
        print(after_conv.size())
        after_linear = self.linear_layers(after_conv.transpose(1, 2).transpose(1, 0))
        return self.final_layer(after_linear.view(self.batch_size, -1))



    '''
    Batch norm depends on the batch size which can be changed when doing
    daydreaming. so need to set it like this.
    '''
    def __set_layers(self):
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.otu_handler.num_strains,
                      64,
                      self.kernel_size, padding=1)
            , nn.ReLU()
            , nn.Conv1d(64,
                        64 * 2,
                        self.kernel_size, padding=1)
            , nn.ReLU()
            , nn.Conv1d(64 * 2,
                        64 * 4,
                        self.kernel_size, padding=1)
            , nn.ReLU()
            , nn.Conv1d(64 * 4,
                        self.otu_handler.num_strains,
                        1)
            , nn.ReLU()
            # , nn.MaxPool1d(2, 2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.otu_handler.num_strains, self.hidden_dim),
            nn.BatchNorm1d(self.batch_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.batch_size),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.batch_size),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.batch_size),
            nn.ReLU(),
            # nn.Dropout(0.5)
            )

    def get_intermediate_losses(self, loss_function,
                                num_batches=10):
        '''
        This generates some scores
        '''
        self.eval()

        scores_to_return = []

        # First get some training loss and then a validation loss.
        if self.otu_handler.test_data is not None:
            samples = ['train', 'validation', 'test']
        else:
            samples = ['train', 'validation']

        for which_sample in samples:
            loss = 0
            for b in range(num_batches):
                data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                           self.slice_len,
                                                                           which_data=which_sample)
                data = add_cuda_to_variable(data, self.use_gpu,
                                            requires_grad=False).transpose(1, 2).transpose(0, 1)
                targets = add_cuda_to_variable(targets, self.use_gpu,
                                               requires_grad=False)

                outputs = self.forward(data)

                # Get the loss associated with this validation data.

                loss += loss_function(outputs, targets)

            # Store a normalized loss.
            if self.use_gpu:
                scores_to_return.append(loss.data.cpu().numpy().item()
                                        / (num_batches))
            else:
                scores_to_return.append(loss.data.numpy().item()
                                        / (num_batches))
        return scores_to_return

    def __print_and_log_losses(self, new_losses, save_params):
        train_l = new_losses[0]
        val_l = new_losses[1]
        self.train_loss_vec.append(train_l)
        self.val_loss_vec.append(val_l)
        print('Train loss: {}'.format(train_l))
        print('  Val loss: {}'.format(val_l))

        if len(new_losses) == 3:
            test_l = new_losses[2]
            self.test_loss_vec.append(test_l)
            print(' Test loss: {}'.format(test_l))

        if save_params is not None:
            with open(save_params[1], 'w+') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(self.train_loss_vec)
                writer.writerow(self.val_loss_vec)
                if len(new_losses) == 3:
                    writer.writerow(self.test_loss_vec)


    def do_training(self, batch_size, epochs, lr, samples_per_epoch,
                    save_params=None):

        np.random.seed(1)
        # self.train()

        self.batch_size = batch_size

        if self.use_gpu:
            self.cuda()

        loss_function = nn.MSELoss()
        # TODO: Try Adagrad & RMSProp
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # For logging the data for plotting
        self.train_loss_vec = []
        self.val_loss_vec = []
        self.test_loss_vec = []

        # Get some initial losses.
        losses = self.get_intermediate_losses(loss_function)
        self.__print_and_log_losses(losses, save_params)

        for epoch in range(epochs):
            iterate = 0

            # For a specified number of examples per epoch. Ideally this
            # would be a function of how much data there is. Ie, go through
            # all the data once.
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                self.train()
                # Select a random sample from the data handler.
                data, targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                           self.slice_len,
                                                                           target_slice=False)

                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains
                data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                targets = add_cuda_to_variable(targets, self.use_gpu)
                self.zero_grad()

                # Do a forward pass of the model.
                outputs = self.__forward(data)

                # For this round set our loss to zero and then compare
                # accumulate losses for all of the batch examples.
                # Finally step with the optimizer.

                loss = loss_function(outputs[bat, :], targets[bat, :])
                loss.backward()
                optimizer.step()

                # Basiaclly do the same as above, but with validation data.
                # Also don't have the optimizer step at all.

                iterate += 1

            print('Completed Epoch ' + str(epoch))


            # Get some train and val losses. These can be used for early
            # stopping later on.
            losses = self.get_intermediate_losses(loss_function)
            self.__print_and_log_losses(losses, save_params)

            # Save the model and logging information.
            if save_params is not None:
                torch.save(self.state_dict(), save_params[0])
                print('Saved model state to: {}'.format(save_params[0]))

        return train_loss_vec, val_loss_vec


    '''
    Function for letting the model "dream" up new data. Given a primer it will
    generate examples for as long as specified.
    '''
    def daydream(self, primer, predict_len=100):
        self.batch_size = 1
        # self.__set_layers()
        # self.eval()

        predicted = primer
        for p in range(predict_len):
            inp = add_cuda_to_variable(predicted[:, -self.slice_len:], self.use_gpu)
            inp = inp.unsqueeze(-1).transpose(0, 2).transpose(0, 1)
            output = self.__forward(inp).transpose(0, 1).data.numpy()
            # Add the new value to the values to be passed to the LSTM.
            predicted = np.concatenate((predicted, output), axis=1)

        return predicted
