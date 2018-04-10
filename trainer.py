from otu_handler import OTUHandler
from model import LSTM
import torch
import os

# Constant stuff
model_dir = 'models'
log_dir = 'logs'


batch_size = 30
hidden_dim = 100
samples_per_epoch = 100000
num_epochs = 15
learning_rate = 0.0005
seq_len = 6

# If this value is below one then it increases by that percent
# every epoch. If greater than one then it increases by that fixed
# amount after every epoch.
slice_incr_amt = 2
otu_handler = OTUHandler([
                          # 'data/gut_A_subset_5.csv',
                          'subsetted_data/gut_A_subset_5_clr.csv',
                          # 'data/gut_B_subset_5.csv'
                          'subsetted_data/gut_B_subset_5_clr.csv'
                          ])
otu_handler.set_train_val()

use_gpu = torch.cuda.is_available()

rnn = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
           LSTM_in_size=10)

model_name = 'model.pt'
log_name = 'log.csv'
save_params = (os.path.join(model_dir, model_name),
               os.path.join(log_dir, log_name))

train_loss, val_loss = rnn.train(seq_len, batch_size,
                                 num_epochs,
                                 learning_rate,
                                 samples_per_epoch,
                                 save_params=save_params,
                                 slice_incr=slice_incr_amt
                                 )
