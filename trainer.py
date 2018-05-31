from otu_handler import OTUHandler
from lstm import LSTM
from ffn import FFN
import torch
import os
import sys
import pandas as pd
from params import *

# Constant stuff
model_dir = 'models'
log_dir = 'logs'

# Read in our data
input_dir = sys.argv[1]
files = []
for (dirpath, dirnames, filenames) in os.walk(input_dir):
    files.extend(filenames)
    break
files = [os.path.join(input_dir, f) for f in files if not f.endswith('_clr.csv')]

# Generate the data handler object
otu_handler = OTUHandler(files)

# Set train and validation split
otu_handler.set_train_val()
otu_handler.normalize_data()

use_gpu = torch.cuda.is_available()


model_name = 'model_lstm.pt'
log_name = 'log_lstm.csv'
save_params = (os.path.join(model_dir, model_name),
               os.path.join(log_dir, log_name))


rnn = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
           LSTM_in_size=reduced_num_strains)


train_loss, val_loss = rnn.do_training(seq_len, batch_size,
                                       num_epochs,
                                       learning_rate,
                                       samples_per_epoch,
                                       save_params=save_params,
                                       slice_incr=slice_incr_amt
                                       )

model_name = 'model_conv.pt'
log_name = 'log_conv.csv'

ffn = FFN(hidden_dim, batch_size, otu_handler, 20, 32, use_gpu=use_gpu)
train_loss, val_loss = ffn.do_training(batch_size, num_epochs, learning_rate,
                                       samples_per_epoch, save_params=save_params)
