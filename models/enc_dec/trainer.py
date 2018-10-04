from lstm import LSTM
import torch
import os
import sys
import pandas as pd
from params import *
# add the root directory of the project to the path..
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from otu_handler import OTUHandler

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

print('Loaded in data. Ready to train.\n')

use_gpu = torch.cuda.is_available()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

save_params = (os.path.join(output_dir, model_name),
               os.path.join(output_dir, log_name))


rnn = LSTM(hidden_dim, otu_handler, use_gpu,
           LSTM_in_size=reduced_num_strains)


train_loss, val_loss = rnn.do_training(seq_len, batch_size,
                                       num_epochs,
                                       learning_rate,
                                       samples_per_epoch,
                                       save_params=save_params,
                                       slice_incr_frequency=slice_incr_frequency
                                       )
