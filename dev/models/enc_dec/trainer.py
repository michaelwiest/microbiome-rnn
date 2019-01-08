from encoder_decoder import EncoderDecoder
import torch
import os
import sys
import pandas as pd
from params import *
import argparse


# add the root directory of the project to the path..
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from otu_handler import OTUHandler


# add the root directory of the project to the path..
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from otu_handler import OTUHandler

# Read in our data
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str,
                    help="The directory of input training data.")
parser.add_argument("-t", "--test", type=str,
                    help="The directory of excluded test data.")
parser.add_argument("-g", "--gpu", type=int,
                    help="GPU index to use.")
args = parser.parse_args()
input_dir = args.data
test_dir = args.test
gpu_to_use = args.gpu

# Get all the files.
files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
if test_dir is not None:
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
    test_files.sort()
else:
    test_files = None
files.sort()

# Generate the data handler object
otu_handler = OTUHandler(files, test_files)

# Set train and validation split
otu_handler.set_train_val()

# Normalize the data.
if type(norm_method) == list:
    for nm in norm_method:
        otu_handler.normalize_data(method=nm)
else:
    otu_handler.normalize_data(method=norm_method)

print('Loaded in data. Ready to train.\n')

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(gpu_to_use)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

save_params = (os.path.join(output_dir, model_name),
               os.path.join(output_dir, log_name))


rnn = EncoderDecoder(hidden_dim,
                     otu_handler,
                     num_lstms,
                     use_gpu,
                     LSTM_in_size=reduced_num_strains)


rnn.do_training(seq_len, batch_size,
               num_epochs,
               learning_rate,
               samples_per_epoch,
               teacher_force_frac,
               weight_decay,
               save_params=save_params,
               use_early_stopping=use_early_stopping,
               early_stopping_patience=early_stopping_patience,
               slice_incr_frequency=slice_incr_frequency
               )
