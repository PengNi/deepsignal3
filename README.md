# DeepSignal3

## A deep learning tool for DNA methylation detection from modern Oxford Nanopore reads.

## Contents

- [Installation](#Installation)
- [Trained models](#Trained-models)
- [Quick start](#Quick-start)
- [Usage](#Usage)

## Installation

deepsignal3 is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/).

- Prerequisites:\
   [Python3.\*](https://www.python.org/) (version >=3.12) \
   [Dorado](https://github.com/nanoporetech/dorado)\
   [Guppy](https://nanoporetech.com/community)
- Dependencies: \
   [numpy](http://www.numpy.org/) \
   [h5py](https://github.com/h5py/h5py) \
   [statsmodels](https://github.com/statsmodels/statsmodels/) \
   [scikit-learn](https://scikit-learn.org/stable/) \
   [mappy](https://github.com/lh3/minimap2/tree/master/python) \
   [PyTorch](https://pytorch.org/) (version >=1.2.0, <=2.1.0)

#### 1. Create an environment

We highly recommend to use a virtual environment for the installation of deepsignal3 and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):

```bash
# create
conda create -n deepsignalpenv python=3.12
# activate
conda activate deepsignalpenv
# deactivate
conda deactivate
```

The virtual environment can also be created by using [virtualenv](https://github.com/pypa/virtualenv/).

#### 2. Install deepsignal3

- After creating and activating the environment, download deepsignal3 (**lastest version**) from github:

```bash
git clone https://github.com/PengNi/deepsignal3.git
cd deepsignal3
python setup.py install
```

- [PyTorch](https://pytorch.org/) can be automatically installed during the installation of deepsignal3. However, if the version of [PyTorch](https://pytorch.org/) installed is not appropriate for your OS, an appropriate version should be re-installed in the same environment as the [instructions](https://pytorch.org/get-started/locally/):

```bash
# install using conda
conda install pytorch==1.11.0 cudatoolkit=10.2 -c pytorch
# or install using pip
pip install torch==1.11.0
```

## Trained models

Currently, we have trained the following models:

- [human_r1041_4khz_CG_epoch7.ckpt](model/human_r1041_4khz_CG_epoch7.ckpt): model trained using human **R10.4.1(4kHz)** data with reference genome chm13v2 for detecting 5mC at CpG sites.
- [human_r1041_5khz_CG_epoch5.ckpt](model/human_r1041_5khz_CG_epoch5.ckpt): model trained using human **R10.4.1(5kHz)** data with reference genome chm13v2 for detecting 5mC at CpG sites.
- ~~[plant_r1041_4khz_C_epoch7.ckpt](model/plant_r1041_4khz_C_epoch7.ckpt): model trained using rice **R10.4.1(4kHz)** data for detecting 5mC at CG/CHG/CHH. The use of this model requires the use of parameters `--motifs C --seq_len 21 --signal_len 16`. (Not recommended)~~

- [plant_r1041_5khz_C_epoch4.ckpt](model/plant_r1041_5khz_C_epoch4.ckpt): model trained using rice **R10.4.1(5kHz)** data for detecting 5mC at CG/CHG/CHH. The use of this model requires the use of parameters `--motifs C --seq_len 13 --signal_len 15`. (recommend)

## Example data

Example data, including training data and test data, can be downloaded from ([google drive](https://drive.google.com/drive/folders/1GNkT0a8-jNdNJe1Wx2eI5hJY_Zv9bXqF)). Example data from the human genome HG002.

## Quick start

To call modifications, the raw fast5 files should be basecalled ([Guppy](https://nanoporetech.com/community)(version <=6.2.1)), and the raw pod5 files should be basecalled ([Dorado](https://github.com/nanoporetech/dorado)). Belows are commands to call 5mC in CG (you can use --motifs to change, for example --motifs CHH):

Demo commands of using Dorado and deepsignal3 to call 5mC from POD5/SloW5/BloW5 files:

```bash
# 1. dorado basecall using GPU
dorado basecaller dna_r9.4.1_e8_sup@v3.3/ --emit-moves --device cuda:all pod5/ --reference chm13v2.0.fa  > demo.bam --batchsize 64
# 2. deepsignal3 call_mods
deepsignal3 call_mods --input_path pod5/ --bam demo.bam --model_path *.ckpt --result_file pod5.CG.call_mods.tsv --nproc 32 --nproc_gpu 4  --seq_len 21 --signal_len 15 -b 8192
deepsignal3 call_freq --input_path pod5.CG.call_mods.tsv --result_file pod5.CG.call_mods.frequency.tsv
```

Demo commands of using Guppy and deepsignal3 to call 5mC from FAST5 files:

```bash
# Higher versions of Guppy no longer support the output format fast5
# Download and unzip the example data and pre-trained models.
# 1. guppy basecall using GPU
guppy_basecaller -i multi_fast5s/ -r -s fast5s_guppy/ --config dna_r10.4.1_e8.2_400bps_hac_prom.cfg --device CUDA:0 --fast5_out
# multi_fast5s/ is the folder where hg002.r10.4.test.fast5 is stored
# fast5s_guppy/ is the output folder
# 2. deepsignal3 call_mods
# CG
deepsignal3 call_mods --input_path fast5s_guppy/ --model_path *.ckpt --result_file fast5s.CG.call_mods.tsv --reference_path chm13v2.0.fa --motifs CG --nproc 32 --nproc_gpu 4 -b 8192
deepsignal3 call_freq --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.tsv
```

## Usage

#### 1. Basecall

If raw file is pod5, before run deepsignal, the raw reads should be basecalled ([Dorado](https://github.com/nanoporetech/dorado)).

For the example data:

```bash
# 1. basecall using GPU
dorado  basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 --device cuda:0 --emit-moves  pod5/ --reference reference.fa  > example.bam
# or using CPU
dorado  basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 --device cpu --emit-moves  pod5/ --reference reference.fa  > example.bam
```

If the raw reads is in FAST5 format, before running deepsignal, the raw reads should be basecalled ([Guppy](https://nanoporetech.com/community)(version <=6.2.1)).

For the example data:

```bash
# 1. basecall using GPU
guppy_basecaller -i multi_fast5s/ -r -s fast5s_guppy/ --config dna_r10.4.1_e8.2_400bps_hac_prom.cfg --device CUDA:0 --fast5_out
# or using CPU
guppy_basecaller -i multi_fast5s/ -r -s fast5s_guppy/ --config dna_r10.4.1_e8.2_400bps_hac_prom.cfg --fast5_out
```

#### 2. call modifications

To call modifications, either the extracted-feature file or **the raw pod5 files (recommended)** can be used as input.

For the example data:

```bash
# call 5mCpGs for instance

# extracted-feature file as input
deepsignal3 call_mods --input_path pod5s.CG.features.tsv --model_path human.r10.4.CG.epoch7.ckpt --result_file pod5s.CG.call_mods.tsv --motifs CG --nproc 32 --nproc_gpu 4 -b 8192

# pod5/slow5/blow5 files as input, use GPU
deepsignal3 call_mods --input_path pod5/ --bam demo.bam --model_path human.r10.4.CG.epoch7.ckpt --result_file pod5.CG.call_mods.tsv --nproc 32 --nproc_gpu 4  --seq_len 21 --signal_len 15 -b 8192
# fast5 files as input, use GPU
deepsignal3 call_mods --input_path fast5s_guppy --model_path human.r10.4.CG.epoch7.ckpt --result_file fast5s.CG.call_mods.tsv --reference_path chm13v2.0.fa --motifs CG --nproc 32 --nproc_gpu 4 -b 8192
```

The modification_call file is a tab-delimited text file in the following format:

- **chrom**: the chromosome name
- **pos**: 0-based position of the targeted base in the chromosome
- **strand**: +/-, the aligned strand of the read to the reference
- **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
- **readname**: the read name
- **read_strand**: t/c, template or complement
- **prob_0**: [0, 1], the probability of the targeted base predicted as 0 (unmethylated)
- **prob_1**: [0, 1], the probability of the targeted base predicted as 1 (methylated)
- **called_label**: 0/1, unmethylated/methylated
- **k_mer**: the kmer around the targeted base

#### 3. call frequency of modifications

A modification-frequency file can be generated by `call_freq` function with the call_mods file as input:

```bash
# call 5mCpGs for instance

# output in tsv format
deepsignal3 call_freq --input_path pod5s.CG.call_mods.tsv --result_file pod5s.CG.call_mods.frequency.tsv
# output in bedMethyl format
deepsignal3 call_freq --input_path pod5s.CG.call_mods.tsv --result_file pod5s.CG.call_mods.frequency.bed --bed
# use --sort to sort the results
deepsignal3 call_freq --input_path pod5s.CG.call_mods.tsv --result_file pod5s.CG.call_mods.frequency.bed --bed --sort
```

The modification_frequency file can be either saved in [bedMethyl](https://www.encodeproject.org/data-standards/wgbs/) format (by setting `--bed` as above), or saved as a tab-delimited text file in the following format by default:

- **chrom**: the chromosome name
- **pos**: 0-based position of the targeted base in the chromosome
- **strand**: +/-, the aligned strand of the read to the reference
- **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
- **prob_0_sum**: sum of the probabilities of the targeted base predicted as 0 (unmethylated)
- **prob_1_sum**: sum of the probabilities of the targeted base predicted as 1 (methylated)
- **count_modified**: number of reads in which the targeted base counted as modified
- **count_unmodified**: number of reads in which the targeted base counted as unmodified
- **coverage**: number of reads aligned to the targeted base
- **modification_frequency**: modification frequency
- **k_mer**: the kmer around the targeted base

#### 4. extract features

Features of targeted sites can be extracted for training or testing.

For the example data, deepsignal3 extracts 21-mer-seq and 21\*15-signal features of each CpG motif in reads by default.:

```bash
deepsignal3 extract -i pod5/ --bam example.bam --reference_path chm13v2.0.fa -o pod5.CG.features.tsv --nproc 30 --motifs CG &

deepsignal3 extract -i fast5s_guppy --reference_path chm13v2.0.fa -o fast5s.CG.features.tsv --nproc 30 --motifs CG &
```

The extracted_features file is a tab-delimited text file in the following format:

- **chrom**: the chromosome name
- **pos**: 0-based position of the targeted base in the chromosome
- **strand**: +/-, the aligned strand of the read to the reference
- **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
- **readname**: the read name
- **read_strand**: t/c, template or complement
- **k_mer**: the sequence around the targeted base
- **signal_means**: signal means of each base in the kmer
- **signal_stds**: signal stds of each base in the kmer
- **signal_lens**: lens of each base in the kmer
- **raw_signals**: signal values for each base of the kmer, splited by ';'
- **methy_label**: 0/1, the label of the targeted base, for training

#### 5. train new models

A new model can be trained as follows:

```bash
# need to split training samples to two independent datasets for training and validating
# please use deepsignal3 train -h/--help for more details
deepsignal3 train --train_file /path/to/train/file --valid_file /path/to/valid/file --model_dir /dir/to/save/the/new/model
```

## Result

The following table shows the results of 5mCpG calling from publicly avaiable HG002 (R10.4.1) data ([ONT Open Datasets](https://labs.epi2me.io/askenazi-kit14-2022-12/)). The Dorado version for comparison is 0.3.4 and the model version is dna_r10.4.1_e8.2_400bps_sup@v4.1.0. The following table shows the correlations with resutls of WGBS:

|   method   | pearson | rsquare | spearman |  RMSE  | mean_coverage |
| :--------: | :-----: | :-----: | :------: | :----: | :-----------: |
| deepsignal | 0.9307  | 0.8662  |  0.8673  | 0.1413 |    4.5607     |
|   dorado   | 0.9229  | 0.8518  |  0.8687  | 0.1465 |    4.2188     |

The following table shows the read-level performance：

|   method   |   TP    |   FN   |   TN    |   FP   | accuracy | recall | specificity | precision |
| :--------: | :-----: | :----: | :-----: | :----: | :------: | :----: | :---------: | :-------: |
| deepsignal | 97094.4 | 2905.6 | 98097.0 | 1903.0 |  0.9760  | 0.9709 |   0.9810    |  0.9808   |
|   dorado   | 93991.4 | 6008.6 | 99265.8 | 734.2  |  0.9663  | 0.9399 |   0.9927    |  0.9922   |

## Appendix

#### For the VBZ compression issue

Please try adding ont-vbz-hdf-plugin to your environment as follows when all fast5s failed in `tombo resquiggle` and/or `deepsignal3 call_mods`. Normally it will work after setting `HDF5_PLUGIN_PATH`:

```shell
# download ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz (or newer version) and set HDF5_PLUGIN_PATH
# https://github.com/nanoporetech/vbz_compression/releases
wget https://github.com/nanoporetech/vbz_compression/releases/download/v1.0.1/ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz
tar zxvf ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz
export HDF5_PLUGIN_PATH=/abslolute/path/to/ont-vbz-hdf-plugin-1.0.1-Linux/usr/local/hdf5/lib/plugin
```

## Todo

- [ ] add tqdm for progress bar
- [ ] support data output format: bam
