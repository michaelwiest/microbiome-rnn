from __future__ import print_function
import os
import pdb
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
              fancybox=True, shadow=True, ncol=2, fontsize=6)
    plt.xlabel('PC0 ({}%)'.format(str(100.0 * pca.explained_variance_ratio_[0])[:5]))
    plt.ylabel('PC1 ({}%)'.format(str(100.0 * pca.explained_variance_ratio_[1])[:5]))
    plt.title('Reduced Dimensionality Hidden Weights\nOf Neural Network')
    plt.show()

def main():
    # Not really used. Just for instantiating the model.
    batch_size = 30
    hidden_dim = 32

    # Read in our data
    input_dir = sys.argv[1]
    model_name = sys.argv[2]
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
               LSTM_in_size=10)

    # Load in the trained model data.
    rnn.load_state_dict(torch.load(model_name))
    pca = PCA(n_components=2)
    trans = pca.fit_transform(np.array(rnn.before_lstm[0].weight.data).T)
    plot_scatter_from_weights(trans, otu_handler, pca, 4)
    # print(rnn.after_lstm)
    trans = pca.fit_transform(np.array(rnn.after_lstm[6].weight.data))
    plot_scatter_from_weights(trans, otu_handler, pca, 4)
    pdb.set_trace()

if __name__ == '__main__':
    main()
