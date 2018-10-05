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
from lstm import LSTM
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helpers.model_helper import *
import csv



class ConvLSTM(LSTM):
    '''
    This subclass inherits from the LSTM class and
    adds 1d convolution over the input time.
    '''
    def __init__(self, hidden_dim, otu_handler,
                 use_gpu=False,
                 LSTM_in_size=None):
        super(ConvLSTM, self).__init__(hidden_dim, otu_handler,
                                   use_gpu=use_gpu,
                                   LSTM_in_size=LSTM_in_size)
        self.hidden_dim = hidden_dim
        self.otu_handler = otu_handler
        if LSTM_in_size is None:
            LSTM_in_size = self.otu_handler.num_strains

        self.lstm = nn.LSTM(LSTM_in_size, hidden_dim, 1)
        self.conv_element = nn.Sequential(
            nn.Conv1d(self.otu_handler.num_strains, 256,
                      kernel_size=4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 256,
                      kernel_size=2, stride=1, padding=1),

            nn.ReLU(),
            nn.Conv1d(256, 256,
                      kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256,
                      kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256,
                      kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, LSTM_in_size,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.deconv_element = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, self.otu_handler.num_strains,
                               kernel_size=4, stride=2, padding=3),
            nn.ReLU()
        )

    def forward(self, input_data):
        # input_data is shape: sequence_size x batch x num_strains
        input_data = input_data.transpose(0,1).transpose(1, 2)
        input_data = self.conv_element(input_data)
        input_data = input_data.transpose(0, 2).transpose(1, 2)
        output, self.hidden = self.lstm(input_data, self.hidden)
        output = self.deconv_element(output.transpose(0,1).transpose(1,2))
        return output
