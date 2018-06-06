'''
This file is for plotting the predictions of the model given a Priming
sequence. This file is pretty crude and could definitely be abstracted
where all of this is in each of the model objects.
'''
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
              hidden_dim=128,
              slice_len=20,
              conv_filters=32,
              ffn=True,
              lstm_in_size=80):
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
        m = FFN(hidden_dim, batch_size, otu_handler, slice_len, conv_filters)
    else:
        m = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
                   LSTM_in_size=lstm_in_size)
    m.load_state_dict(torch.load(model_file))
    return m


def get_comparison_data(model, comparison_index, time_point_index, time_window):
    d = model.otu_handler.val_data[comparison_index]
    primer = d.values[:, time_point_index - time_window: time_point_index]
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
    actual_vals = df.values[:, time_point_index - time_window:
                     time_point_index - time_window + plot_len]
    if raw_plot:
        actual_vals = model.otu_handler.un_normalize_data(actual_vals, comparison_index)
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
    comparison_file_index = int(sys.argv[6])
    plot_len = 100
    raw_plot = True
    lstm_slice_len = 20
    try:
        model = get_model(model_file, input_dir, ffn=True)
        is_ffn = True
    except KeyError:
        model = get_model(model_file, input_dir, ffn=False)
        is_ffn = False
    model.eval()
    if is_ffn:
        primer = get_comparison_data(model, comparison_file_index, time_point_index,
                                     model.slice_len)
    else:
        primer = get_comparison_data(model, comparison_file_index, time_point_index,
                                      lstm_slice_len)

    dream = model.daydream(primer, plot_len)

    if raw_plot:
        dream = model.otu_handler.un_normalize_data(dream, comparison_file_index)
    plot_comparison(model, comparison_file_index, time_point_index, time_window,
                    num_strains=num_strains_to_plot, plot_len=plot_len,
                    raw_plot=raw_plot)

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
