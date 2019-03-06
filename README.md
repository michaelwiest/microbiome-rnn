# microbiome_rnn
A suite of models for doing time-series predictions of the human gut microbiome.
At the moment the models are:
+ Feed-forward network (FFN)
+ Long short term memory (LSTM)
+ Encoder-decoder.

## Data
### Data sources
The input data is from QIITA which is a repository for microbiome data. The data are:
<!-- 1. [__Study_id=550__]("https://qiita.ucsd.edu/study/description/550"). This is [__Rob Knight's Moving Picture of Human Microbiome__]("http://dx.doi.org/10.1186/gb-2011-12-5-r50") paper -->
1. [__Study_id=11052__]("https://qiita.ucsd.edu/study/description/11052"). This is a different set of Rob Knight's time-series data.
2. [__Study_id=2202__]("https://qiita.ucsd.edu/study/description/2202"). This is another [__time-series study__]("http://dx.doi.org/10.1186/gb-2014-15-7-r89") with 2 other individuals.
3. [__Study_id=10283__]("https://qiita.ucsd.edu/study/description/10283"). This is Larry Smarr's pre and post-op data for a colonoscopy. There is also data from a few other women in here.
4. [__Study_id=1015__]("https://qiita.ucsd.edu/study/description/1015"). This is a dataset of Rob Knight and a few other researchers data from a trip abroad. Rob's data from here is excluded because it is present in __Study 11052__.

### Data preprocessing
After studies have been downloaded from QIITA (in the `*.biom` format) they need to be cleaned up prior to being fed to the model. Additionally, a taxonomy file is needed (this can be generated from QIIME).

The general workflow is to use the scripts in the `data_preprocessing` directory in this order:
- `metadata_taxonomy_adder`
- `biom_combiner` (if necessary)
- `host_site_separator_time_sorting`

At this point only the sampling sites or hosts of interest can be selected as they are now in their own files (as opposed to combined into one `*.biom` file).

- `sum_truncate_sort_taxonomy`
- `filtering_normalization_completion`
- `top_N_strains`

How to use each of these files is described in the respective file.

<!-- First, add taxonomy to the data, sort the data in time and break out each sampling site, and sampled individual.
```bash
$ python data_preprocessing/biom_preproccessing.py <some_biom_file.biom> <corresponding_taxonomy_file.txt>
```
This will generate a file with the same name as the input but with the suffix `_sorted_tax.csv`. Many files may be created as the input is broken out by sampling site and sampled individual.

Next, we need to perform matrix completion and normalize the data to reduce sequencing bias.
```bash
$ python data_preprocessing/otu_matrix_completer.py <the_sorted_file_from_above.csv>
```

This will generate the data that is in a usable state.

Lastly, we want to perform a subsetting operation so that all the inputs to the network have the same strains in the same position in the input. Put all of the desired files (as outputted from the above step) into some directory (called `some_dir` here for example). A paramter `N` is supplied to the script that is an integer of how many of the top strains in every file to keep. See below:

```bash
$ python data_preprocessing/top_N_strains.py some_dir <some_output_dir> N
```
At this point all of the files in `<some_output_dir>` can be fed to the models. -->

## Training a model
Given a directory of data as produced from above (let's call it `input_dir`) training is very simple. Each model type has its own directory `models/<model_type>/`; in that directory there is a file called `params.py` where pertinent training features can be altered.

To train a model do:
```
python dev/models/<some_model>/trainer.py -d input_data/all_studies_reduced_62
```

## Evaluating a trained model
The ipython notebook under `notebooks/Mode Evaluator.ipynb` has all of the tools necessary for assessing model quality.
