from otu_handler import *
from model import *
from helper import *
import csv
import numpy as np

batch_size = 30
hidden_dim = 100
samples_per_epoch = 5000
num_epochs = 15
learning_rate = 0.003
seq_len = 6
slice_incr_perc = 0.1
otu_handler = OTUHandler(['data/gut_A_subset_5.csv',
                          'data/gut_B_subset_5.csv'])
otu_handler.set_train_val()

use_gpu = torch.cuda.is_available()

rnn = LSTM(hidden_dim, batch_size, otu_handler, use_gpu)

model_name = 'model.pt'
log_name = 'log.csv'
train_loss, val_loss = rnn.train(seq_len, batch_size,
                                 num_epochs,
                                 learning_rate,
                                 samples_per_epoch,
                                 save_params=(model_name, log_name),
                                 slice_incr_perc=slice_incr_perc
                                 )
# print(rnn.batch_dream(5, '$M', 2012, 1, fs, 566))
