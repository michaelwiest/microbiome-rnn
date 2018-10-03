batch_size = 50
# Size of the hidden layers of the network.
hidden_dim = 128
# How many samples to check before one epoch. Kind of arbitrary.
samples_per_epoch = 500000
num_epochs = 20
learning_rate = 0.0000005
# Length of the sequence being passed to the LSTM or FFN.
seq_len = 15
# For the compression step with the LSTM how many strains to reduce it to.
reduced_num_strains = 50
# If this value is below one then it increases by that percent
# every epoch. If greater than one then it increases by that fixed
# amount after every epoch. Only used for RNN.
slice_incr_amt = 0

additional_comments = ''
run_suffix = '_h{}_sl{}_rns{}_sia{}{}'.format(hidden_dim,
                                              seq_len,
                                              reduced_num_strains,
                                              slice_incr_amt,
                                              additional_comments)
model_name = 'model{}.pt'.format(run_suffix)
log_name = 'log{}.pt'.format(run_suffix)
output_dir = 'rnn_output'
