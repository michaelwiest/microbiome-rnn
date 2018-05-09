import sys
import os.path

import pandas as pd
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from model import *
from otu_handler import OTUHandler
plt.style.use('fivethirtyeight')

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
# otu_handler.set_train_val()
use_gpu = torch.cuda.is_available()

# Get the model
rnn = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
           LSTM_in_size=5)
rnn.load_state_dict(torch.load(sys.argv[5]))
dream = rnn.daydream(primer, 1, 300, serial=False)


plot_len = 50
num_strains = 4

plt.figure(figsize=(18, 9))
plt.plot(clr_df.values[:num_strains, time_point_index - time_window:
                           time_point_index - time_window + plot_len].T,
         label='Actual', linewidth=2
         )
plt.gca().set_prop_cycle(None)
plt.plot(dream[:num_strains, :plot_len].T, linestyle='--',
         label='Predicted', linewidth=2)
plt.axvspan(0, time_window - 1, alpha=0.3, color='gray',
            label='Priming Region', hatch='/')
plt.legend(loc='lower right')
plt.xlabel('Time')
plt.ylabel('CLR(OTU)')
plt.title('Predicted vs. Actual OTU Counts')
plt.show()
