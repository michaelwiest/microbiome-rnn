# microbiome_rnn
RNN for predicting OTU counts of organisms from human gut microbiome.

## Usage
`python trainer.py`

## Structure
General structure of the network is a few fully connected layers to reduce the dimensionality of the input. That is then fed to an LSTM. Following the LSTM there are more fully connected layers to expand the dimensionality back to the original dimensions.

## Data Preprocessing
Raw data from QIITA can be found under the directory: `from_qiita`.
Each of the `*.biom` files in those directories is passed through the `biom_preproccessing.py` file where it has its taxonomy added and data cleaned up.

After that, desired files are put into their own directory (`reference_data`) where they are then subsetted using the script `top_N_strains.py`.

## Notes
As of now, the rolling mean values generate better predictions as they are less "spikey" than the raw values.
