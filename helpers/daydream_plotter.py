import sys
import os.path

import pandas as pd
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from skbio.stats.composition import clr

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lstm import *
from otu_handler import OTUHandler
plt.style.use('fivethirtyeight')


def get_model(model_file, input_dir,
              batch_size=30,
              hidden_dim=64):
    # Read in our data
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
    rnn.load_state_dict(torch.load(model_file))
    return rnn


def get_comparison_data(comparison_fname, time_point_index, time_window):
    raw_df = pd.read_csv(comparison_fname + '.csv', index_col=0)
    gm = np.array(gmean(raw_df, axis=1))
    primer = raw_df.values[:, time_point_index - time_window: time_point_index]
    gm = gmean(primer.T)
    primer = clr(primer)
    gm = np.expand_dims(gm, 1)

    primer = np.concatenate((gm, primer), axis=1)
    return primer, gm


def dream_and_plot(rnn, primer, comparison_fname,
                   time_point_index,
                   time_window,
                   gm,  # Geometric means
                   num_strains=6,
                   plot_len=100,
                   raw_plot=False):

    dream = rnn.daydream(primer, 1, 300, serial=False)
    gm = np.repeat(gm, dream.shape[1], axis=1)
    if raw_plot:
        dream = gm * np.exp(dream)
        df = pd.read_csv(comparison_fname + '.csv', index_col=0)
    else:
        df = pd.read_csv(comparison_fname + '_clr.csv', index_col=0)

    strains = rnn.otu_handler.strains
    plt.figure(figsize=(18, 9))
    actual_vals = df.values[:, time_point_index - time_window:
                               time_point_index - time_window + plot_len]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(num_strains):
        # Plot the actual values
        plt.plot(actual_vals[i, :].T,
                 color=colors[i],
                 label='{}'.format(strains[i]), linewidth=2)
        # Plot the predicted values.
        plt.plot(dream[i, 1:plot_len],
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


def main():
    # Plot the CLR vals or the raw OTU vals.
    plot_raw_vals = True

    comparison_fname = sys.argv[1]
    time_point_index = int(sys.argv[2])
    time_window = int(sys.argv[3])
    input_dir = sys.argv[4]
    model_file = sys.argv[5]

    rnn = get_model(model_file, input_dir)
    primer, gm = get_comparison_data(comparison_fname, time_point_index,
                                     time_window)
    dream_and_plot(rnn, primer, comparison_fname,
                   time_point_index, time_window, gm, raw_plot=plot_raw_vals)



if __name__ == '__main__':
    main()
