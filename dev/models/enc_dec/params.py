batch_size = 50
# Size of the hidden layers of the network.
hidden_dim = 2048
# How many samples to check before one epoch. Kind of arbitrary.
samples_per_epoch = 500000
num_epochs = 50
learning_rate = 0.0000005
weight_decay = 0
# Length of the sequence being passed to the LSTM or FFN.
seq_len = 5
# For the compression step with the LSTM how many strains to reduce it to.
reduced_num_strains = 31
# If this value is below one then it increases by that percent
# every epoch. If greater than one then it increases by that fixed
# amount after every epoch. Only used for RNN.
slice_incr_frequency = 2

# What percentage of the time to use teacher forcing during training
teacher_force_frac = 0.5

# How many stacked LSTMs to use?
num_lstms = 1

# How to normalize the data. Can be 'clr' or 'zscore', or a list of either.
norm_method = ['clr', 'zscore']
# Where to save the model's results.
output_dir = 'enc_dec_output'

# Early stopping stuff
use_early_stopping = True
early_stopping_patience = 15

additional_comments = '_swapped_test'

run_suffix = '_EncDec_h{}_sl{}_rns{}_sif{}_numlstm{}{}'.format(hidden_dim,
                                              seq_len,
                                              reduced_num_strains,
                                              slice_incr_frequency,
                                              num_lstms,
                                              additional_comments)
model_name = 'model{}.pt'.format(run_suffix)
log_name = 'log{}.npy'.format(run_suffix)
