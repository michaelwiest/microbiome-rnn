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


run_suffix = '_h128_sl15_rns50'
model_name = 'model{}.pt'
log_name = 'log{}.pt'
output_dir = 'rnn_output'
