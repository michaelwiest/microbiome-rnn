from lstm import LSTM
from conv_lstm import ConvLSTM
import torch
import os
import sys
import pandas as pd
from params import *
import argparse

# add the root directory of the project to the path..
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from otu_handler import OTUHandler

# Read in our data
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str,
                    help="The directory of input training data.")
parser.add_argument("-t", "--test", type=str,
                    help="The directory of excluded test data.")
args = parser.parse_args()
input_dir = args.data
test_dir = args.test

files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
if test_dir is not None:
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
else:
    test_files = None
# Generate the data handler object
otu_handler = OTUHandler(files, test_files)

# Set train and validation split
otu_handler.set_train_val()
otu_handler.normalize_data()
print('Loaded in data. Ready to train.\n')

use_gpu = torch.cuda.is_available()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

save_params = (os.path.join(output_dir, model_name),
               os.path.join(output_dir, log_name))

if use_convs:
    print('Using Conv-LSTM')
    rnn = ConvLSTM(hidden_dim, otu_handler, use_gpu,
                   LSTM_in_size=reduced_num_strains)
else:
    rnn = LSTM(hidden_dim, otu_handler, use_gpu,
               LSTM_in_size=reduced_num_strains)


train_loss, val_loss = rnn.do_training(seq_len, batch_size,
                                       num_epochs,
                                       learning_rate,
                                       samples_per_epoch,
                                       save_params=save_params,
                                       slice_incr_frequency=slice_incr_frequency
                                       )
