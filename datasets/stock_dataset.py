import torch
from torch.utils.data.dataloader import Dataset
import numpy as np
from argparse import ArgumentParser
import pickle
import pandas as pd

def filter_splits(splits=[], seq_length=30):
    new_splits = {}
    for key in splits:
        new_splits[key] = []
        for i in range(len(splits[key])):
            new_splits[key].append(splits[key][i])
    return new_splits

def construct_data_item(ori_data, splits=[], seq_length=30, data_version='v1'):
    # Filter splits
    splits = filter_splits(splits, seq_length)
    infos = {}
    if data_version == 'v1':
        # open,high,low,close,volume,amount,adjustflag
        # 3, 4, 5, 6
        # Feature: open, high, low, close
        # Label: high, low, close
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

    elif data_version == 'v2':
        for key in splits:
            infos[key] = {'data': [], 'len': 0}
            for i in range(len(splits[key]) - seq_length - 1):
                feature_matric = np.zeros((seq_length, 4))
                for j in range(0, seq_length + 1):
                    if j != seq_length:
                        feature_matric[j, 0:4] = ori_data[splits[key][i + j], 3:7]
                        print(f"Feat: {j}, Ori: {i+j}")
                    else:
                        label = np.array([
                            ori_data[splits[key][i + j], 4],  # high
                            ori_data[splits[key][i + j], 5],  # low
                            ori_data[splits[key][i + j], 6],   # close
                        ])
                        print(f"Label: {i + j}")
                feature_matric = feature_matric.reshape((-1))
                infos[key]['data'].append({
                    'feature_matric': feature_matric,
                    'label': label,
                })
                infos[key]['len'] += 1


    elif data_version == 'v3':
        ori_data = ori_data[:, 3:]
        for key in splits:
            infos[key] = {'data': [], 'len': 0}
            for i in range(len(splits[key]) - seq_length):
                feature_matric = np.zeros((seq_length, 4))

                for j in range(seq_length):
                    # Ori: open,high,low,close,volume,amount,adjustflag
                    # Transed: open,volume,amount,high,low,close
                    feature_matric[j, 0:4] = ori_data[splits[key][i + j], 0:4]
                    #feature_matric[j, 1:3] = ori_data[splits[key][i + j], 4:6]
                    #feature_matric[j, 3:6] = ori_data[splits[key][i + j], 1:4]

                infos[key]['data'].append(feature_matric)
                infos[key]['len'] += 1
    return infos


def gen_split(ori_data, sampling_mode='segments', train_data_ratio=0.8):
    assert sampling_mode in ['segments', 'sequential']
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
                        splits['test'].append(indices_array[j])
                    splits['val'].append(indices_array[j])
                day_maps[date] = 1
                indices_array = [i]
            else:
                indices_array.append(i)
    elif sampling_mode == 'sequential':
        for i in range(ori_data.shape[0]):
            if i <= ori_data.shape[0] * train_data_ratio:
                splits['train'].append(i)
            else:
                splits['test'].append(i)

    return splits


def gen_data_info(data_file_path, train_ratio, sampling_mode, info_save_path):
    csv_data = pd.read_csv(data_file_path)
    original_data = csv_data.values
    splits = gen_split(original_data, sampling_mode, train_ratio)
    #infos = construct_data_item(original_data, splits=splits, data_version='v3')
    infos = construct_data_item(original_data,
                                splits=splits,
                                seq_length=30,
                                data_version='v3')
    pickle.dump(infos, open(info_save_path, 'wb'))
    print("Stock infos write to {0}".format(info_save_path))
    for split in infos:
        print('{0} length: {1}'.format(split, infos[split]['len']))


class BaoStockDataset(Dataset):
    def __init__(self, data_info_path, split_mode, train_seq=20, test_seq=10):
        assert split_mode in ['train', 'test', 'val']
        infos = pickle.load(open(data_info_path, 'rb'))
        self.infos = infos
        self.split = split_mode
        self.train_seq = train_seq
        self.test_seq = test_seq

    def __len__(self):
        return 100
        #return self.infos[self.split]['len']

    def get_item(self, index):
        index = 0
        data_item = self.infos[self.split]['data'][index]
        data_item = np.array(data_item, dtype=np.float32)
        return data_item

    def __getitem__(self, index):
        ret = self.get_item(index)
        return ret


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_file_path', type=str, required=True)
    args = arg_parser.parse_args()

    gen_data_info(args.data_file_path, 0.8, 'sequential', 'data/A_stock_k_1h_20150701_20210701_v3.pkl')