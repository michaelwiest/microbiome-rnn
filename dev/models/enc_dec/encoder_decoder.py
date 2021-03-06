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


class Encoder(nn.Module):
    '''
    Encoder object. It returns its own output to be used in the attention
    mechanism.
    '''
    def __init__(self, input_size, hidden_dim, num_lstms, use_gpu):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_dim
        self.input_size = input_size
        self.use_gpu = use_gpu

        self.lstm = nn.LSTM(input_size, hidden_dim, num_lstms)
        self.linear = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, self.input_size),
            nn.ReLU()
        )

    def forward(self, input, hidden):
        input = self.linear(input)
        output, hidden = self.lstm(input, hidden)
        return output, hidden

class Decoder(nn.Module):
    '''
    Decoder object. It can variably use attention tools.
    Attention mechanism taken from the pytorch tutorial.
    '''
    def __init__(self,
                 input_size,
                 hidden_dim,
                 num_lstms,
                 max_len,
                 use_gpu,
                 use_attention=False):
        super(Decoder, self).__init__()
        self.max_len = max_len
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_dim, num_lstms)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_size),
        )

        if use_attention:
            self.attn = nn.Linear(self.hidden_dim * 2,
                                  self.max_len)
            self.attn_combine = nn.Linear(self.hidden_dim * 2,
                                          self.input_size)
            self.embedder = nn.Linear(self.input_size, self.hidden_dim)
        self.hidden = None

    def forward(self, input, encoder_output=None, hidden=None):
        '''
        Forward pass through the decoder object.
        '''
        if hidden is None:
            hidden = self.hidden

        if self.use_attention:
            if not encoder_output.size()[0] == self.max_len:
                to_cat = add_cuda_to_variable(torch.zeros(self.max_len - encoder_output.size()[0],
                                                           encoder_output.size()[1],
                                                           encoder_output.size()[2]),
                                                           self.use_gpu)
                encoder_output = torch.cat((to_cat, encoder_output), 0)
            embedded = self.embedder(input)
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0][0]), 1)),
                dim=1).unsqueeze(1)
            attn_applied = torch.bmm(attn_weights,
                                     encoder_output.transpose(0, 1)
                                     ).transpose(0, 1)
            output = torch.cat((embedded[0], attn_applied[0]), 1)
            input = self.attn_combine(output).unsqueeze(0)

        output, self.hidden = self.lstm(input, hidden)
        output = self.linear(output)
        return output


class EncoderDecoder(nn.Module):
    '''
    Model for predicting OTU counts of microbes given historical data.
    Uses fully connected layers and an three LSTMS.
    It learns to predict forward in time and also to recapitulate the input
    sequence.
    '''
    def __init__(self, hidden_dim, otu_handler,
                 num_lstms,
                 use_gpu=False,
                 LSTM_in_size=None,
                 use_attention=True):
        super(EncoderDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.otu_handler = otu_handler
        self.use_gpu = use_gpu

        # Essentially how many OTUs to pass to the LSTMs.
        if LSTM_in_size is None:
            LSTM_in_size = self.otu_handler.num_strains

        # Usually just 1.
        self.num_lstms = num_lstms
        # Make the encoder and decoders objects.
        self.encoder = Encoder(LSTM_in_size, hidden_dim, num_lstms, self.use_gpu)
        self.decoder_forward = Decoder(LSTM_in_size,
                                       hidden_dim,
                                       num_lstms,
                                       self.otu_handler.min_len,
                                       self.use_gpu,
                                       use_attention=use_attention)
        self.decoder_backward = Decoder(LSTM_in_size,
                                        hidden_dim,
                                        num_lstms,
                                        self.otu_handler.min_len,
                                        self.use_gpu,
                                        use_attention=use_attention)

        # Non-torch inits.

        self.hidden = None

        self.best_model = self.state_dict()
        self.best_model_epoch = 0
        self.best_loss = np.inf


    def forward(self, input_data, num_predictions,
                teacher_data=None, bypass_encoder=False):
        # Teacher data should be a tuple of length two where the first value
        # is the data corresponding to the future prediction and the
        # second value is the data corresponding to the reversed input.
        # data is shape: sequence_size x batch x num_strains

        if not bypass_encoder:
            encoder_output, self.hidden = self.encoder.forward(input_data, self.hidden)

            self.decoder_forward.hidden = self.hidden
            self.decoder_backward.hidden = self.hidden

        # Get the last input example.
        forward_inp = input_data[-1, :, :].unsqueeze(0)
        backward_inp = input_data[-1, :, :].unsqueeze(0)
        for i in range(num_predictions):
            forward = self.decoder_forward.forward(forward_inp,
                                                   encoder_output=encoder_output)
            backward = self.decoder_backward.forward(backward_inp,
                                                     encoder_output=encoder_output)

            # Add our prediction to the list of predictions.
            if i == 0:
                forward_pred = forward
                backward_pred = backward
            else:
                forward_pred = torch.cat((forward_pred,
                                          forward), 0)
                backward_pred = torch.cat((backward_pred,
                                          backward), 0)

            # If there is no teacher data then use the most recent prediction
            # to make the next prediction. Otherwise use the teacher data.
            if teacher_data is None:
                forward_inp = forward
                backward_inp = backward
            else:
                forward_inp = teacher_data[0][i, :, :].unsqueeze(0)
                # If we are beyond the bounds of the backward teacher data,
                # just use the new prediction. These predictions are not passed
                # to the loss function so it doesn't matter.
                if i >= (teacher_data[1].size()[0]):
                    backward_inp = backward
                else:
                    backward_inp = teacher_data[1][i, :, :].unsqueeze(0)

        return forward_pred.transpose(1, 2).transpose(0, 2), backward_pred.transpose(1, 2).transpose(0, 2)[:, :, :input_data.size()[0]]

    def __evaluate_early_stopping(self,
                                current_epoch,
                                early_stopping_patience,
                                validation_index=1):
        '''
        Check if our current state is better than the best one so far.
        if so then update it. Also if we are beyond the patience limit then
        we should stop if no model improvement.
        '''
        losses = self.loss_tensor[:, :, -1].sum(axis=1).tolist()
        val_loss = losses[validation_index]
        if val_loss < self.best_loss:
            self.best_model = self.state_dict()
            self.best_loss = val_loss
            self.best_model_epoch = current_epoch
            stop_training = False
            print('\tUpdated best model state!')
        elif current_epoch - self.best_model_epoch > early_stopping_patience:
            stop_training = True
        else:
            stop_training = False

        return stop_training



    def __init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim).cuda()),
                           Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim)),
                           Variable(torch.zeros(self.num_lstms,
                                                self.batch_size,
                                                self.hidden_dim))
                           )
    def get_intermediate_losses(self,
                                loss_function,
                                inp_slice_len,
                                target_slice_len,
                                teacher_force_frac,
                                num_batches=10,
                                which_donor=None,
                                which_sample=None # This is a string
                                ):
        '''
        This generates some scores for the accuracy of the training and
        validation and test data. It returns the losses on a per-otu basis.

        This is super redundant with the do training code and could be
        combined in the future.
        '''
        # Put the model in evaluation mode.
        self.eval()

        if which_sample is None:
            # First get some training loss and then a validation loss.
            if self.otu_handler.test_data is not None:
                samples = ['train', 'validation', 'test']
            else:
                samples = ['train', 'validation']
        elif type(which_sample) == str and which_sample in ['train', 'validation', 'test']:
            samples = [which_sample]

        strain_losses = np.zeros((len(samples), self.otu_handler.num_strains))

        for i, which_sample in enumerate(samples):

            for b in range(num_batches):
                # Select a random sample from the data handler.
                data, forward_targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                                   inp_slice_len,
                                                                                   target_slice_len,
                                                                                   slice_offset=inp_slice_len,
                                                                                   which_data=which_sample,
                                                                                   which_donor=which_donor)
                # this is the data that the backward decoder will reconstruct
                backward_targets = np.flip(data, axis=2).copy()
                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains

                data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                forward_targets = add_cuda_to_variable(forward_targets, self.use_gpu,
                                                       requires_grad=False)
                backward_targets = add_cuda_to_variable(backward_targets, self.use_gpu,
                                                        requires_grad=False)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                if np.random.rand() < teacher_force_frac:
                    tf = (forward_targets.transpose(1, 2).transpose(0, 1),
                          backward_targets.transpose(1, 2).transpose(0, 1))
                else:
                    tf = None
                # Do a forward pass of the model.
                forward_preds, backward_preds = self.forward(data,
                                                             target_slice_len,
                                                             teacher_data=tf)


                if self.use_gpu:
                    forward_preds = forward_preds.detach().cpu()
                    backward_preds = backward_preds.detach().cpu()
                    forward_targets = forward_targets.detach().cpu()
                    backward_targets = backward_targets.detach().cpu()
                # We want to get the loss on a per-strain basis.
                for strain in range(self.otu_handler.num_strains):
                    strain_losses[i, strain] += loss_function(forward_preds[:, strain, :],
                                                              forward_targets[:, strain, :])
                    strain_losses[i, strain] += loss_function(backward_preds[:, strain, :],
                                                              backward_targets[:, strain, :])

        # normalize the loss. The two is because we have two loss functions.
        strain_losses /= (2 * num_batches * self.otu_handler.num_strains)
        return strain_losses

    def __print_and_log_losses(self, new_losses, save_params,
                               instantiate=False # Overwrite tensor if first time.
                               ):
        '''
        This function joins the newest loss values to the ongoing tensor.
        It also prints out the data in a readable fashion.
        '''
        if instantiate:
            self.loss_tensor = np.expand_dims(new_losses, axis=-1)
        else:
            new_losses = np.expand_dims(new_losses, axis=-1)
            self.loss_tensor = np.concatenate((self.loss_tensor, new_losses),
                                              axis=-1)

        to_print = self.loss_tensor[:, :, -1].sum(axis=1).tolist()
        print_str = ['Train loss: {}', '  Val loss: {}',
                     ' Test loss: {}']
        for i, tp in enumerate(to_print):
            print(print_str[i].format(tp))

        if save_params is not None:
            np.save(save_params[1], self.loss_tensor)


    def do_training(self,
                    inp_slice_len,
                    target_slice_len,
                    batch_size,
                    epochs,
                    lr,
                    samples_per_epoch,
                    teacher_force_frac,
                    weight_decay,
                    inp_slice_incr_frequency=None,
                    target_slice_incr_frequency=None,
                    save_params=None,
                    use_early_stopping=True,
                    early_stopping_patience=10):
        '''
        This function is what actually trains the model.
        '''
        np.random.seed(1)

        self.batch_size = batch_size
        self.__init_hidden()

        if self.use_gpu:
            self.cuda()

        loss_function = nn.MSELoss()
        # TODO: Try Adagrad & RMSProp
        optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=weight_decay)

        # Get some initial losses.
        losses = self.get_intermediate_losses(loss_function, inp_slice_len,
                                              target_slice_len,
                                              teacher_force_frac)

        self.loss_tensor = None
        self.__print_and_log_losses(losses, save_params, instantiate=True)

        for epoch in range(epochs):
            iterate = 0

            # For a specified number of examples per epoch. This basically
            # decides how many examples to do before increasing the length
            # of the slice of data fed to the LSTM.
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                self.train() # Put the network in training mode.

                # Select a random sample from the data handler.
                data, forward_targets = self.otu_handler.get_N_samples_and_targets(self.batch_size,
                                                                           inp_slice_len,
                                                                           target_slice_len,
                                                                           slice_offset=inp_slice_len
                                                                           )

                # this is the data that the backward decoder will reconstruct
                backward_targets = np.flip(data, axis=2).copy()

                # Transpose
                #   from: batch x num_strains x sequence_size
                #   to: sequence_size x batch x num_strains

                data = add_cuda_to_variable(data, self.use_gpu).transpose(1, 2).transpose(0, 1)
                forward_targets = add_cuda_to_variable(forward_targets, self.use_gpu,
                                                       requires_grad=False)
                backward_targets = add_cuda_to_variable(backward_targets, self.use_gpu,
                                                        requires_grad=False)
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.__init_hidden()

                if np.random.rand() < teacher_force_frac:
                    tf = (forward_targets.transpose(1, 2).transpose(0, 1),
                          backward_targets.transpose(1, 2).transpose(0, 1))
                else:
                    tf = None
                # Do a forward pass of the model.
                forward_preds, backward_preds = self.forward(data,
                                                             target_slice_len,
                                                             teacher_data=tf)

                # For this round set our loss to zero and then compare
                # accumulated losses for all of the batch examples.
                # Finally step with the optimizer.
                floss = loss_function(forward_preds, forward_targets)
                bloss = loss_function(backward_preds, backward_targets)
                loss = floss + bloss
                loss.backward()
                optimizer.step()
                iterate += 1

            print('Completed Epoch ' + str(epoch))

            # Get some train and val losses. These can be used for early
            # stopping later on.
            losses = self.get_intermediate_losses(loss_function,
                                                  inp_slice_len,
                                                  target_slice_len,
                                                  teacher_force_frac)
            self.__print_and_log_losses(losses, save_params)

            # If we want to increase the slice of the data that we are
            # training on then do so.
            if inp_slice_incr_frequency is not None:
                if inp_slice_incr_frequency > 0:
                    if epoch != 0 and epoch % inp_slice_incr_frequency == 0:
                        inp_slice_len += 1
                        # Make sure that the slice doesn't get longer than the
                        # amount of data we can feed to it. Could handle this with
                        # padding characters.
                        inp_slice_len = min(self.otu_handler.min_len - 1, int(inp_slice_len))
                        print('Increased input slice length to: {}'.format(inp_slice_len))

            if target_slice_incr_frequency is not None:
                if target_slice_incr_frequency > 0:
                    if epoch != 0 and epoch % target_slice_incr_frequency == 0:
                        target_slice_len += 1
                        # Make sure that the slice doesn't get longer than the
                        # amount of data we can feed to it. Could handle this with
                        # padding characters.
                        target_slice_len = min(self.otu_handler.min_len - 1, int(target_slice_len))
                        print('Increased target slice length to: {}'.format(target_slice_len))

            # Check if our current state is better than our best thus far.
            if use_early_stopping:
                stop_early = self.__evaluate_early_stopping(epoch,
                                                            early_stopping_patience)
            else:
                self.best_model = self.state_dict()
                self.best_model_epoch = epoch
            # Save the model and logging information.
            if save_params is not None:
                torch.save(self.best_model, save_params[0])
                print('Saved model state to: {}'.format(save_params[0]))
                print('Best model from epoch: {}'.format(self.best_model_epoch))

            # If we have met our stopping criteria then stop.
            if stop_early and use_early_stopping:
                break

    def daydream2(self, primer, predict_len=100):
        '''
        This function primes the model with a certain sequence and then allows
        it to predict into the future.

        There used to be a function called daydream, hence why this is called
        what it is.
        '''
        if len(primer.shape) != 3:
            raise ValueError('Please provide a 3d array of shape: '
                             '(num_strains, slice_length, batch_size)')
        self.batch_size = primer.shape[-1]
        self.__init_hidden()

        inp = add_cuda_to_variable(primer, self.use_gpu, requires_grad=False).transpose(0, 2).transpose(0, 1)
        output, _ = self.forward(inp, predict_len)
        output = output.transpose(0, 1).transpose(1, 2)
        if self.use_gpu:
            output = output.cpu().detach().numpy()
        else:
            output = output.detach().numpy()

        output = np.concatenate((primer, output), axis=1)
        return output
