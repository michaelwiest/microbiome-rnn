import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from model import *
from otu_handler import OTUHandler
import pandas as pd
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt

comparison_fname = sys.argv[1]
time_point_index = int(sys.argv[2])
time_window = int(sys.argv[3])

raw_df = pd.read_csv(comparison_fname + '.csv', index_col=0)
clr_df = pd.read_csv(comparison_fname + '_clr.csv', index_col=0)
gm = np.array(gmean(raw_df, axis=1))
primer = clr_df.values[:, time_point_index - time_window: time_point_index]


# Not really used. Just for instantiating the model.
batch_size = 30
hidden_dim = 32

# Read in our data
input_dir = sys.argv[4]
files = []
for (dirpath, dirnames, filenames) in os.walk(input_dir):
    files.extend(filenames)
    break
files = [os.path.join(input_dir, f) for f in files if f.endswith('_clr.csv')]

# Generate the data handler object
# print(files)
otu_handler = OTUHandler(files)

# Set train and validation split
otu_handler.set_train_val()
use_gpu = torch.cuda.is_available()

# Get the model
rnn = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
           LSTM_in_size=10)
rnn.load_state_dict(torch.load(sys.argv[5]))
dream = rnn.daydream(primer, 1, 300)

print(dream.shape)
print(clr_df.shape)
plot_len = 30
plt.plot(clr_df.values[:3, time_point_index - time_window:time_point_index - time_window + plot_len].T)
plt.plot(dream[:3, :plot_len].T)
plt.show()
