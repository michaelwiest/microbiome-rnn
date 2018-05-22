from otu_handler import OTUHandler
from lstm import LSTM
from ffn import FFN
import torch
import os
import sys
import pandas as pd
# Constant stuff
model_dir = 'models'
log_dir = 'logs'


batch_size = 30
hidden_dim = 64
samples_per_epoch = 500000
num_epochs = 15
learning_rate = 0.0000005
seq_len = 3
reduced_num_strains = 5

# If this value is below one then it increases by that percent
# every epoch. If greater than one then it increases by that fixed
# amount after every epoch. Only used for RNN.
slice_incr_amt = 1

# Read in our data
input_dir = sys.argv[1]
files = []
for (dirpath, dirnames, filenames) in os.walk(input_dir):
    files.extend(filenames)
    break
files = [os.path.join(input_dir, f) for f in files if not f.endswith('_clr.csv')]
print(files)
# Generate the data handler object
otu_handler = OTUHandler(files)


# Set train and validation split
otu_handler.set_train_val()
otu_handler.normalize_data()

use_gpu = torch.cuda.is_available()


model_name = 'model.pt'
log_name = 'log.csv'
save_params = (os.path.join(model_dir, model_name),
               os.path.join(log_dir, log_name))


# rnn = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
#            LSTM_in_size=reduced_num_strains)
#
#
# train_loss, val_loss = rnn.do_training(seq_len, batch_size,
#                                  num_epochs,
#                                  learning_rate,
#                                  samples_per_epoch,
#                                  save_params=save_params,
#                                  slice_incr=slice_incr_amt
#                                  )

ffn = FFN(hidden_dim, batch_size, otu_handler, 20, 32, use_gpu=use_gpu)
train_loss, val_loss = ffn.do_training(batch_size, num_epochs, learning_rate,
                                 samples_per_epoch, save_params=save_params)
