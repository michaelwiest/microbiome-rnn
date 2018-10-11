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
from ffn import FFN
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helpers.model_helper import *
import csv



class ConvFFN(FFN):
    '''
    This subclass inherits from the LSTM class and
    adds 1d convolution over the input time.
    '''
    def __init__(self, hidden_dim, bs, otu_handler, slice_len,
                 use_gpu=False):
        super(ConvFFN, self).__init__(hidden_dim, bs, otu_handler,
                                      slice_len,
                                      use_gpu=use_gpu)
        self.conv_element = nn.Sequential(
            nn.Conv1d(self.otu_handler.num_strains, hidden_dim,
                      kernel_size=4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim,
                      kernel_size=2, stride=1, padding=1),

            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim,
                      kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim,
                      kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim,
                      kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.time_transformer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.deconv_element = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim,
                               kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.hidden_dim, self.otu_handler.num_strains,
                               kernel_size=4, stride=2, padding=3),
            nn.ReLU()
        )
        # self.lin_final = nn.Linear(self.otu_handler.num_strains,
        #                            self.otu_handler.num_strains)

    def forward(self, data):
        # data is shape: sequence_size x batch x num_strains
        data = data.transpose(0, 1).transpose(1, 2)
        # print(data.size())
        data = self.conv_element(data)
        # print(data.size())
        data = data.transpose(0, 2).transpose(1, 2)
        data = self.time_transformer(data)
        # print(data.size())
        data = self.deconv_element(data.transpose(0,1).transpose(1,2))
        # print(data.size())
        return data
