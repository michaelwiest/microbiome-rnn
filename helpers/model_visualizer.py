from __future__ import print_function
import os
import torch
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from otu_handler import OTUHandler
from model import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


'''
Plots the transformed weights onto a scatter plot.
'''
def plot_scatter_from_weights(weights, otu_handler, pca, taxonomy_depth=4):
    ind = list(otu_handler.samples[0].index.values)
    ind = np.array([';'.join(i.split(';')[:taxonomy_depth]) for i in ind])
    ind_sub = np.array(list(set(ind)))
    print('{} categories'.format(len(ind_sub)))
    cm = plt.get_cmap('tab20')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/len(ind_sub)) for i in range(len(ind_sub))])
    print(weights.shape)
    for i in range(len(ind_sub)):
        bools = (ind_sub[i] == ind)
        sub = weights[bools, :]
        plt.scatter(sub[:, 0], sub[:, 1], label=ind_sub[i], alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2, fontsize=8)
    plt.xlabel('PC0 ({}%)'.format(str(100.0 * pca.explained_variance_ratio_[0])[:5]))
    plt.ylabel('PC1 ({}%)'.format(str(100.0 * pca.explained_variance_ratio_[1])[:5]))
    plt.title('Reduced Dimensionality Hidden Weights\nOf Neural Network')
    plt.tight_layout()
    plt.show()

def main():
    # Not really used. Just for instantiating the model.
    batch_size = 30
    hidden_dim = 32

    input_dir = sys.argv[1]
    model_name = sys.argv[2]
    taxonomy_depth = int(sys.argv[3])

    # Read in our data
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        files.extend(filenames)
        break
    files = [os.path.join(input_dir, f) for f in files if f.endswith('_clr.csv')]

    # Generate the data handler object
    otu_handler = OTUHandler(files)

    # Set train and validation split
    otu_handler.set_train_val()
    use_gpu = torch.cuda.is_available()

    # Get the model
    rnn = LSTM(hidden_dim, batch_size, otu_handler, use_gpu,
               LSTM_in_size=5)

    # Load in the trained model data.
    rnn.load_state_dict(torch.load(model_name))

    # Perform PCA
    pca = PCA(n_components=2)

    # Plot the values from before the LSTM and also after.
    # The indices below need to be changed based on model architecture.
    trans = pca.fit_transform(np.array(rnn.before_lstm[0].weight.data).T)
    plot_scatter_from_weights(trans, otu_handler, pca, taxonomy_depth)
    trans = pca.fit_transform(np.array(rnn.after_lstm[6].weight.data))
    plot_scatter_from_weights(trans, otu_handler, pca, taxonomy_depth)

if __name__ == '__main__':
    main()
