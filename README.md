# Microbiome Encoder Decoder
A suite of models for doing time-series predictions of the human gut microbiome.
At the moment the models are:
+ Feed-forward network (FFN)
+ Long short term memory (LSTM)
+ Encoder-decoder.
As of right now the Encdoer-decoder is the most successful and all reporting is done on that model.

## Data
### Data sources
The input data is from QIITA (a repository for microbiome data). The data are:
1. [__Study_id=11052__]("https://qiita.ucsd.edu/study/description/11052"). This is a different set of Rob Knight's time-series data.
2. [__Study_id=2202__]("https://qiita.ucsd.edu/study/description/2202"). This is another [__time-series study__]("http://dx.doi.org/10.1186/gb-2014-15-7-r89") with 2 other individuals.
3. [__Study_id=10283__]("https://qiita.ucsd.edu/study/description/10283"). This is Larry Smarr's pre and post-op data for a colonoscopy. There is also data from a few other women in here.
4. [__Study_id=1015__]("https://qiita.ucsd.edu/study/description/1015"). This is a dataset of Rob Knight and a few other researchers data from a trip abroad. Rob's data from here is excluded because it is present in __Study 11052__.

### Data preprocessing
After studies have been downloaded from QIITA (in the `*.biom` format) they need to be cleaned up prior to being fed to the model. The metadata for a study should also be downloaded from QIITA at the same time. Additionally, a taxonomy file is needed (this can be generated from QIIME).

The general workflow is to use the scripts in the `data_preprocessing` directory in this order:
- `metadata_taxonomy_adder`
- `biom_combiner` (if necessary)
- `host_site_separator_time_sorting`

At this point only the sampling sites or hosts of interest can be selected as they are now in their own files (as opposed to combined into one `*.biom` file).

- `sum_truncate_sort_taxonomy`
- `filtering_normalization_completion`
- `top_N_strains`

How to use each of these scripts is described in the respective file.

The output of the data preprocessing pipeline is available in `input_data`. The directory `summed_completed_no_chloro` was used to train the models reported on in this study. The `_no_chloro` refers to the fact that an organism identified as part of a chloroplast was manually removed from those datasets.

## Training a model
Given a directory of data as produced from above (let's call it `input_dir`) training is very simple. Each model type has its own directory `models/<model_type>/`; in that directory there is a file called `params.py` where pertinent training features can be altered.

To train a model do:
```
python dev/models/<some_model>/trainer.py -d input_data/summed_completed_no_chloro/all_strains_top_N
```

If you want to also have a testing dataset, remove the dataset CSV(s) from the directory listed above and put them in their own directory (ie, `test`). Now the command above becomes:
```
python dev/models/<some_model>/trainer.py -d input_data/summed_completed_no_chloro/all_strains_top_N -t input_data/summed_completed_no_chloro/test
```

## Evaluating a trained model
The ipython notebook under `notebooks/Model Evaluator.ipynb` has all of the tools necessary for assessing model quality.

## Evaluating the input data
The ipython notebook under `notebooks/Input Analysis` has all of the tools necessary for assessing trends in the data.
