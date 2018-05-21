import sys
import os.path

import pandas as pd
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from skbio.stats.composition import clr
from scipy.stats.mstats import gmean, zscore

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lstm import *
from ffn import *
from otu_handler import OTUHandler
plt.style.use('fivethirtyeight')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_model(model_file, input_dir,
              batch_size=30,
              hidden_dim=64,
              slice_len=20,
              ffn=True):
    # Read in our data
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        files.extend(filenames)
        break
    files = [os.path.join(input_dir, f) for f in files]

    # Generate the data handler object
    otu_handler = OTUHandler(files)
    otu_handler.set_train_val()
    otu_handler.normalize_data()

    # Set train and validation split
    # otu_handler.set_train_val()
    use_gpu = torch.cuda.is_available()

    # Get the model
    if ffn:
        m = FFN(hidden_dim, batch_size, otu_handler, slice_len)
    else:
        m = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
                   LSTM_in_size=5)
    m.load_state_dict(torch.load(model_file))
    return m


def get_comparison_data(model, comparison_index, time_point_index, time_window):
    d = model.otu_handler.val_data[comparison_index]
    primer = d.values[:, time_point_index - time_window: time_point_index]
    # gm = gmean(primer.T)
    primer = zscore(primer)
    # gm = np.expand_dims(gm, 1)

    # primer = np.concatenate((gm, primer), axis=1)
    return primer


def plot_comparison(model, comparison_index,
                   time_point_index,
                   time_window,
                   num_strains=6,
                   plot_len=100,
                   raw_plot=False):

    strains = model.otu_handler.strains
    df = model.otu_handler.val_data[comparison_index]
    plt.figure(figsize=(18, 9))
    av = df.values
    av = zscore(av)
    actual_vals = av[:, time_point_index - time_window:
                     time_point_index - time_window + plot_len]

    for i in range(num_strains):
        # Plot the actual values
        plt.plot(actual_vals[i, :].T,
                 color=colors[i],
                 label='{}'.format(strains[i]), linewidth=2)

def main():
    # Plot the CLR vals or the raw OTU vals.
    time_point_index = int(sys.argv[1])
    time_window = int(sys.argv[2])
    input_dir = sys.argv[3]
    model_file = sys.argv[4]
    num_strains_to_plot = int(sys.argv[5])
    plot_len = 100

    # rnn = get_model(model_file, input_dir)
    model = get_model(model_file, input_dir, ffn=True)
    primer = get_comparison_data(model, 0, time_point_index,
                                 model.slice_len)
    dream = model.daydream(primer)
    plot_comparison(model, 0, time_point_index, time_window,
                    num_strains=num_strains_to_plot, plot_len=plot_len)

    for i in range(num_strains_to_plot):
        # Plot the predicted values.
        plt.plot(dream[i, :plot_len],
                 linestyle='--',
                 linewidth=2,
                 color=colors[i])

    plt.axvspan(0, time_window - 1, alpha=0.3, color='gray',
                label='Priming Region', hatch='/')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2, fontsize=9)
    plt.xlabel('Time')
    plt.ylabel('CLR(OTU)')
    plt.title('Predicted vs. Actual OTU Counts')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
