# TSDiff : Diffusion-based Generative AI for Exploring Transition States from 2D Molecular Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/seonghann/tsdiff/tree/master/LICENSE)

Our baseline code is from Geodiff [[arXiv](https://arxiv.org/abs/2203.02923)], [[github](https://github.com/MinkaiXu/GeoDiff)].
The official implementation of TSDiff: TBA.
![cover](assets/figure1.png)

## Environments

### Install via Conda (Recommended)

```bash
# Clone the environment
conda env create -f env.yml
# Activate the environment
conda activate cu102_tsdiff
# Install PyG
conda install pytorch-geometric=1.7.2=py37_torch_1.8.0_cu102 -c rusty1s -c conda-forge
```

## Dataset

### Offical Dataset
The training and test dataset is from the open-source transition state database of Grambow, Colin A. et al. [[zenodo]](https://zenodo.org/record/3715478) [[paper]](https://www.nature.com/articles/s41597-020-0460-4).

### Preprocessed dataset
We provide the preprocessed datasets calculated with $\omega\text{b97x-D3}$ level.
To reproduce paper you can follow the code.
```bash
PARENT_DIR=$(dirname $(pwd))
export PYTHONPATH="$PARENT_DIR:${PYTHONPATH}"
python3 preprocessing.py
ls data/TS/wb97xd3/random_split_42
```

### Prepare your own GEOM dataset from scratch (optional)

You can also download origianl full dataset and prepare your own data split. Follow the data preparation code.
```bash
PARENT_DIR=$(dirname $(pwd))
export PYTHONPATH="$PARENT_DIR:${PYTHONPATH}"

SAVE_DIR="data/path/to/save"
TS_DATA="data/path/of/ts-xyz.xyz"
# example of ts-xyz.xyz : data/TS/wb97xd3/raw_data/wb97xd3_ts.xyz
RXN_SMARTS_FILE="data/path/of/smarts.csv"
# example of rxn_smarts_file.csv : data/TS/wb97xd3/raw_data/wb97xd3_fwd_rev_chemprop.csv
FEAT_DICT="data/path/of/feat_dict.pkl"
# if you don't have predefined feat_dict, let it "" (empty)

python3 preprocessing.py --feat_dict $FEAT_DICT --save_dir $SAVE_DIR --ts_data $TS_DATA --rxn_smarts_file $RXN_SMARTS_FILE --ban_index -1 --seed 2023
```

## Training

All hyper-parameters and training details are provided in config files (`./configs/*.yml`), and free feel to tune these parameters.

You can train the model with the following commands:

```bash
# Default settings
python train.py ./config/train_config.yml
```

The model checkpoints, configuration yaml file as well as training log will be saved into a directory specified by `--logdir` in `train.py`.

## Sampling

We provide the checkpoints of eight trained models, trained with the $\omega\text{b97x-D3}$ data. Each of them are same except for the initial model weight (initial seed). Note that, please put the checkpoints `*.pt` into paths like `${log}/${model}/checkpoints/`, and also put corresponding configuration file `*.yml` into the upper level directory `${log}/${model}/`.

You can reproduce results of the paper by:

```bash
CKPTS="logs/trained_ckpt/ens0/checkpoints/best_ckpt.pt logs/trained_ckpt/ens1/checkpoints/best_ckpt.pt logs/trained_ckpt/ens2/checkpoints/best_ckpt.pt logs/trained_ckpt/ens3/checkpoints/best_ckpt.pt logs/trained_ckpt/ens4/checkpoints/best_ckpt.pt logs/trained_ckpt/ens5/checkpoints/best_ckpt.pt logs/trained_ckpt/ens6/checkpoints/best_ckpt.pt logs/trained_ckpt/ens7/checkpoints/best_ckpt.pt"
python sampling.py \
    $CKPTS \
    --start_idx 0 --end_idx 9999 --sampling_type ld \
    --save_dir reproduce/wb97xd3 --batch_size 100 \
    --test_set data/TS/wb97xd3/random_split_42/test_data.pkl
```
Here `start_idx` and `end_idx` indicate the range of the test set that we want to use. All hyper-parameters related to sampling can be set in `sampling.py` files. 

Examples of generated TS conformers by TSDiff are provided below.

<p align="center">
  <img src="assets/figure2.png" /> 
</p>

## Clustering

After sampling TS candidates, you can cluster it by their geometric features. We provide `clustering.py` for the higherarchy clustering. In Fig S1. of the paper, we used the clustering code to exploring TS conformers of organic reactions in Birkholz and Schlegel's benchmark set. The benchmark reactions are in `birkholz_benchmark`.
To reproduce it, follows below:
```bash
for ((i=0; i<N; i++));
do python3 sampling.py $CKPTS --start_idx $i --end_idx $((i+1)) --test_set birkholz_benchmark/selected_rxns.txt --save_dir birkholz_benchmark/rxn_${i}

python3 clustering.py --save_dir birkholz_benchmark/rxn_${i}/clustering --sample_path birkholz_benchmark/rxn_${i}/samples_all.pkl
done
```


## Citation
Please consider citing the our paper if you find it helpful. Thank you!
```
@inproceedings{
xu2022geodiff,
title={GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation},
author={Minkai Xu and Lantao Yu and Yang Song and Chence Shi and Stefano Ermon and Jian Tang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=PzcvxEMzvQC}
}
```

## Acknowledgement

This repo is built upon the previous work ConfGF's [[codebase]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional). Thanks Chence and Shitong!

## Contact

If you have any question, please contact me at minkai.xu@umontreal.ca or xuminkai@mila.quebec.

## Known issues

1. The current codebase is not compatible with more recent torch-geometric versions.
2. The current processed dataset (with PyD data object) is not compatible with more recent torch-geometric versions.
