import pandas as pd
import pickle
import os
import numpy as np
import torch
import random
import tqdm
import argparse
from utils.datasets import generate_ts_data2
from utils.parse_xyz import parse_xyz_corpus
from typing import List


def random_split(data_list: List, train: float = 0.8, valid: float = 0.1, seed: int = 1234):
    """
    Randomly split a dataset into non-overlapping train/valid/test set.
    args :
        data_list (list): a list of data
        train (float): ratio of train data
        valid (float): ratio of valid data
        seed (int): random seed
    return :
        train_data (list): a list of train data
        valid_data (list): a list of valid data
        test_data (list): a list of test data
    """
    assert train + valid < 1
    random.seed(seed)
    random.shuffle(data_list)
    N = len(data_list)
    n_train = int(N * train)
    n_valid = int(N * valid)
    train_data = data_list[:n_train]
    valid_data = data_list[n_train: n_train + n_valid]
    test_data = data_list[n_train + n_valid:]

    return train_data, valid_data, test_data


def index_split(num_data: int, train: float = 0.8, valid: float = 0.1, seed: int = 1234):
    """
    Generate randomly splitted index of data into non-overlapping train/valid/test set.
    This function assume that the data is augmented so that original samples are placed in even index
    and the corresponding augmented samples are placed in the next index.
    args :
        num_data (int): the number of data of original samples
        train (float): ratio of train data
        valid (float): ratio of valid data
        seed (int): random seed
    return :
        train_index (list): a list of train index of original and augmented samples
        valid_index (list): a list of valid index of original and augmented samples
        test_index (list): a list of test index of original and augmented samples
    """
    assert train + valid < 1
    random.seed(seed)
    index_list = list(range(num_data))
    random.shuffle(index_list)

    n_train = int(num_data * train)
    n_valid = int(num_data * valid)
    train_index = np.array(index_list[:n_train])
    valid_index = np.array(index_list[n_train: n_train + n_valid])
    test_index = np.array(index_list[n_train + n_valid:])

    train_index = list(np.concatenate((train_index * 2, train_index * 2 + 1)))
    valid_index = list(np.concatenate((valid_index * 2, valid_index * 2 + 1)))
    test_index = list(np.concatenate((test_index * 2, test_index * 2 + 1)))

    train_index.sort()
    valid_index.sort()
    test_index.sort()
    return train_index, valid_index, test_index


def check_dir(dir_name):
    """
    Check the directory exists or not
    If not, make the directory
    Check wheather train_data.pkl, valid_data.pkl, test_data.pkl are exist or not.
    If exist, raise error.

    args :
       dir_name (str): directory name
    return :
         None
    """
    os.makedirs(dir_name, exist_ok=True)
    if os.path.isfile(os.path.join(dir_name, "train_data.pkl")):
        raise ValueError("train_data.pkl is already exist.")
    if os.path.isfile(os.path.join(dir_name, "valid_data.pkl")):
        raise ValueError("valid_data.pkl is already exist.")
    if os.path.isfile(os.path.join(dir_name, "test_data.pkl")):
        raise ValueError("test_data.pkl is already exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--valid", type=float, default=0.1)
    parser.add_argument("--feat_dict", type=str, default="data/TS/wb97xd3/feat_dict.pkl", help="path/to/feat_dict.pkl")
    parser.add_argument("--save_dir", type=str, default="data/TS/wb97xd3/random_split_42")
    parser.add_argument("--ts_data", type=str, default="data/TS/wb97xd3/raw_data/wb97xd3_ts.xyz")
    parser.add_argument("--rxn_smarts_file", type=str, default="data/TS/wb97xd3/raw_data/wb97xd3_fwd_rev_chemprop.csv")
    parser.add_argument("--ban_index", type=int, nargs="+", default=[20568, 20569, 20580, 20581])

    args = parser.parse_args()
    # load source chemical reaction dataset
    # The data are augmented reaction data.
    # The original data is placed in even index and the corresponding augmented data is placed in the next index.
    # By chainging reactants and products, the augmented sample is a reverse reaction of original reaction.

    # transition state geometry data of xyz format.
    xyz_blocks = parse_xyz_corpus(args.ts_data)

    # reaction smarts data of csv format.
    df = pd.read_csv(args.rxn_smarts_file)
    rxn_smarts = df.AAM

    # set index of source data to be excluded
    if args.ban_index[0] != -1:
        ban_index = args.ban_index

    # set feature types
    # if there exist pre-defined feat_dict, load the feat_dict
    if os.path.isfile(args.feat_dict):
        feat_dict = pickle.load(open(args.feat_dict, "rb"))
    else:
        print(args.feat_dict, "is not exist. Use default feat_dict.")
        feat_dict = {
            "GetIsAromatic": {},
            "GetFormalCharge": {},
            "GetHybridization": {},
            "GetTotalNumHs": {},
            "GetTotalValence": {},
            "GetTotalDegree": {},
            "GetChiralTag": {},
            "IsInRing": {},
        }

    # generate torch_geometric.data.Data instance
    data_list = []
    for idx, (a_smarts, xyz_block) in tqdm.tqdm(enumerate(zip(rxn_smarts, xyz_blocks))):
        r, p = a_smarts.split(">>")
        data, feat_dict = generate_ts_data2(r, p, xyz_block, feat_dict=feat_dict)
        data_list.append(data)
        data.rxn_index = idx // 2
        data.augmented = False if idx % 2 == 0 else True

    # convert features to one-hot encoding
    num_cls = [len(v) for k, v in feat_dict.items()]
    for data in data_list:
        feat_onehot = []
        feats = data.r_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.r_feat = torch.cat(feat_onehot, dim=-1)

        feat_onehot = []
        feats = data.p_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.p_feat = torch.cat(feat_onehot, dim=-1)

    train_index, valid_index, test_index = index_split(
        int(len(data_list) / 2),
        train=args.train,
        valid=args.valid,
        seed=args.seed
    )
    train_index = [i for i in train_index if i not in ban_index]
    valid_index = [i for i in valid_index if i not in ban_index]
    test_index = [i for i in test_index if i not in ban_index]

    train_data = [data_list[i] for i in train_index]
    valid_data = [data_list[i] for i in valid_index]
    test_data = [data_list[i] for i in test_index]
    index_dict = {
        "train_index": train_index,
        "valid_index": valid_index,
        "test_index": test_index,
    }

    check_dir(args.save_dir)

    # save the data, feat_dict, index_dict at the save_dir with pickle format. (.pkl)
    with open(os.path.join(args.save_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.save_dir, "valid_data.pkl"), "wb") as f:
        pickle.dump(valid_data, f)
    with open(os.path.join(args.save_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    with open(os.path.join(args.save_dir, "feat_dict.pkl"), "wb") as f:
        pickle.dump(feat_dict, f)
    with open(os.path.join(args.save_dir, "index_dict.pkl"), "wb") as f:
        pickle.dump(index_dict, f)
