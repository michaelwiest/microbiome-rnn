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



class LSTM(nn.Module):
    '''
    Model for predicting OTU counts of microbes given historical data.
    Uses fully connected layers and an LSTM (could use 1d convolutions in the
    future for better accuracy).

    As of now does not have a "dream" function for generating predictions from a
    seeded example.
    '''
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

    def forward(self, data):
        # data is shape: sequence_size x batch x num_strains
        data = self.before_lstm(data)
        data, self.hidden = self.lstm(data, self.hidden)
        data = self.after_lstm(data)

        return data.transpose(0, 1).transpose(1, 2)

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

    def get_intermediate_losses(self, loss_function, slice_len,
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
                                                                           slice_len,
                                                                           which_data=which_sample)
                data = add_cuda_to_variable(data, self.use_gpu,
                                            requires_grad=False).transpose(1, 2).transpose(0, 1)
                targets = add_cuda_to_variable(targets, self.use_gpu,
                                               requires_grad=False)

                self.__init_hidden()
                outputs = self.forward(data)
                output_len = outputs.size(2)
                targets = targets[:, :, -output_len:]

                # Get the loss associated with this validation data.

                loss += loss_function(outputs[bat, :], targets[bat, :])

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


    def do_training(self, slice_len, batch_size, epochs, lr, samples_per_epoch,
              slice_incr_frequency=None, save_params=None):
        np.random.seed(1)

        self.batch_size = batch_size
        self.__init_hidden()

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
        losses = self.get_intermediate_losses(loss_function, slice_len)
        self.__print_and_log_losses(losses, save_params)

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
                targets = add_cuda_to_variable(targets, self.use_gpu,
                                               requires_grad=False)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                # Do a forward pass of the model.
                outputs = self.forward(data)

                # Beacuse of potential convolution losing the size of the
                # input we only select as many output examples to compare
                # against as the model can generate.
                output_len = outputs.size(2)
                targets = targets[:, :, -output_len:]

                # For this round set our loss to zero and then compare
                # accumulated losses for all of the batch examples.
                # Finally step with the optimizer.


                loss = loss_function(outputs[bat, :], targets[bat, :])
                loss.backward()
                optimizer.step()
                iterate += 1

            print('Completed Epoch ' + str(epoch))

            # Get some train and val losses. These can be used for early
            # stopping later on.
            losses = self.get_intermediate_losses(loss_function, slice_len)
            self.__print_and_log_losses(losses, save_params)

            # If we want to increase the slice of the data that we are
            # training on then do so.
            if slice_incr_frequency is not None or slice_incr_frequency > 0:
                if epoch != 0 and epoch % slice_incr_frequency == 0:
                    slice_len += 1
                    # Make sure that the slice doesn't get longer than the
                    # amount of data we can feed to it. Could handle this with
                    # padding characters.
                    slice_len = min(self.otu_handler.min_len - 1, int(slice_len))
                    print('Increased slice length to: {}'.format(slice_len))

            # Save the model and logging information.
            if save_params is not None:
                torch.save(self.state_dict(), save_params[0])
                print('Saved model state to: {}'.format(save_params[0]))

    def daydream(self, primer, predict_len=100, window_size=20,
                 serial=True):
        '''
        Function for letting the LSTM "dream" up new data. Given a primer it will
        generate examples for as long as specified.

        The "serial" argument determines wheter or not examples are fed one
        at a time to the LSTM with no gradient zeroing, or fed as a batch
        and then zeroed everytime. serial=True has been giving better results.
        '''
        if len(primer.shape) != 3:
            raise ValueError('Please provide a 3d array of shape: '
                             '(num_strains, slice_length, batch_size)')
        self.batch_size = primer.shape[-1]
        self.__init_hidden()

        predicted = primer
        # If we do it the serial way, then prime the model with all examples
        # up to the most recent one.
        if serial:
            inp = add_cuda_to_variable(predicted[:, :-1],
                                       self.use_gpu,
                                       requires_grad=False) \
                .transpose(0, 2) \
                .transpose(0, 1)[-window_size:, :, :]
            _ = self.forward(inp)
        for p in range(predict_len):
            if serial:
                inp = add_cuda_to_variable(predicted[:, -1, :], self.use_gpu).unsqueeze(1)
            else:
                inp = add_cuda_to_variable(predicted, self.use_gpu)
            inp = inp.transpose(0, 2).transpose(0, 1)[-window_size:, :, :]
            # Only keep the last predicted value.
            if self.use_gpu:
                output = self.forward(inp)[:, :, -1].transpose(0, 1).data.cpu().numpy()
            else:
                output = self.forward(inp)[:, :, -1].transpose(0, 1).data.numpy()

            # Need to reshape the tensor so it can be concatenated.
            output = np.expand_dims(output, 1)
            # Add the new value to the values to be passed to the LSTM.
            predicted = np.concatenate((predicted, output), axis=1)

            if not serial:
                self.__init_hidden()

        return predicted
