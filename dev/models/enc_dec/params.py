batch_size = 50
# Size of the hidden layers of the network.
hidden_dim = 512
# How many samples to check before one epoch. Kind of arbitrary.
samples_per_epoch = 500000
num_epochs = 50
learning_rate = 0.0000002
weight_decay = 0
# Length of the sequence being passed to the LSTM or FFN.
inp_slice_len = 5
target_slice_len = 5

# This is how many input strains there are.
num_strains = 36

# If this value is below one then it increases by that percent
# every epoch. If greater than one then it increases by that fixed
# amount after every epoch. Only used for RNN.
inp_slice_incr_frequency = None
target_slice_incr_frequency = 2

# What percentage of the time to use teacher forcing during training
teacher_force_frac = 0.5

# Whether or not to use attention when using the decoder.
use_attention = True

# How many stacked LSTMs to use?
num_lstms = 1

# How to normalize the data. Can be 'clr' or 'zscore', or a list of either.
norm_method = ['clr', 'zscore']
# Where to save the model's results.
output_dir = '2019_enc_dec_output'

# Early stopping stuff
use_early_stopping = True
early_stopping_patience = 15

additional_comments = ''

run_suffix = '_EncDec_h{}_isl{}_tsl{}_ns{}_isif{}_tsif{}_numlstm{}{}'.format(hidden_dim,
                                              inp_slice_len,
                                              target_slice_len,
                                              num_strains,
                                              inp_slice_incr_frequency,
                                              target_slice_incr_frequency,
                                              num_lstms,
                                              additional_comments)
model_name = 'model{}.pt'.format(run_suffix)
log_name = 'log{}.npy'.format(run_suffix)
