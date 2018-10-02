from otu_handler import OTUHandler
from ffn import FFN
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

use_gpu = torch.cuda.is_available()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

save_params = (os.path.join(output_dir, model_name),
               os.path.join(output_dir, log_name))


ffn = FFN(hidden_dim, batch_size, otu_handler, 20, 32, use_gpu=use_gpu)
train_loss, val_loss = ffn.do_training(batch_size, num_epochs, learning_rate,
                                       samples_per_epoch, save_params=save_params)
