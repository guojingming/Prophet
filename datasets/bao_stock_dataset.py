import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from argparse import ArgumentParser
import pickle
import pandas as pd

def filter_splits(splits=[], seq_length=30):
    new_splits = {}
    for key in splits:
        new_splits[key] = []
        for i in range(len(splits[key])):
            if splits[key][i] > seq_length:
                new_splits[key].append(splits[key][i])
    return new_splits

def construct_data_item(ori_data, splits=[], seq_length=30, data_version='v1'):
    # Filter splits
    splits = filter_splits(splits, seq_length)
    infos = {}
    if data_version == 'v1':
        # 3, 4, 5, 6
        # Feature: open, ceiling, floor, closed
        # Label: ceiling: reg, floor: reg, closed: reg
        for key in splits:
            infos[key] = {'data':[], 'len': 0}
            for i in range(len(splits[key])):
                item_indices = []
                for j in range(seq_length + 1):
                    item_indices.append(splits[key][i] - j)

                feature_matric = np.zeros((seq_length + 1, 4))
                for i in range(seq_length):
                    feature_matric[i + 1, 0] = ori_data[item_indices[i + 1], 3]
                    feature_matric[i + 1, 1] = ori_data[item_indices[i + 1], 4]
                    feature_matric[i + 1, 2] = ori_data[item_indices[i + 1], 5]
                    feature_matric[i + 1, 3] = ori_data[item_indices[i + 1], 6]
                feature_matric[0] = np.array([ori_data[item_indices[0], 3], 0, 0, 0])
                feature_matric = feature_matric.reshape((-1))

                label = np.array([
                    ori_data[item_indices[0], 4],
                    ori_data[item_indices[0], 5],
                    ori_data[item_indices[0], 6],
                ])

                infos[key]['data'].append({
                    'feature_matric': feature_matric,
                    'label': label,
                })
                infos[key]['len'] += 1
    return infos


def gen_split(ori_data, sampling_mode='segments', train_data_ratio=0.8):
    assert sampling_mode in ['segments', 'uniform']
    splits = {
        'train': [],
        'test': [],
        'val': []
    }

    if sampling_mode == 'segments':
        day_maps = {}
        indices_array = []
        for i in range(ori_data.shape[0]):
            date = ori_data[i, 0]
            if date not in day_maps:
                for j in range(len(indices_array)):
                    if j <= len(indices_array) * train_data_ratio:
                        splits['train'].append(indices_array[j])
                    else:
                        splits['val'].append(indices_array[j])
                        splits['test'].append(indices_array[j])

                day_maps[date] = 1
                indices_array = [i]
            else:
                indices_array.append(i)
    return splits


def gen_data_info(data_file_path, train_ratio, sampling_mode, info_save_path):
    csv_data = pd.read_csv(data_file_path)
    original_data = csv_data.values
    splits = gen_split(original_data, sampling_mode, train_ratio)
    infos = construct_data_item(original_data, splits=splits)
    pickle.dump(infos, open(info_save_path, 'wb'))
    print("Stock infos write to {0}".format(info_save_path))
    for split in infos:
        print('{0} length: {1}'.format(split, infos[split]['len']))



class BaoStockLoader(DataLoader):
    def __init__(self, data_info_path, split_mode):
        assert split_mode in ['train', 'test', 'val']
        infos = pickle.load(open(data_info_path, 'rb'))
        self.split = split_mode

    def __len__(self):
        return self.data_item[self.split].shape[0]

    def __getitem__(self, index):
        return self.data_item[self.split][index]


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_file_path', type=str, required=True)
    args = arg_parser.parse_args()

    gen_data_info(args.data_file_path, 0.8, 'segments', 'data/A_stock_k_15min_20150701_20210701_v1.pkl')



