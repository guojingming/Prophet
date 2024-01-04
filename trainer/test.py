import torch
import os
import numpy as np
from torch import optim
from torch.utils.data.dataloader import DataLoader
from models.MLP import get_model
from datasets.stock_dataset import BaoStockDataset
from utils.plot_utils import draw_price


def test():
    #test_set = BaoStockDataset('data/A_stock_k_1h_20150701_20210701_v3.pkl', 'test')
    test_set = BaoStockDataset('data/A_stock_k_1h_20150701_20210701_v3.pkl', 'train')

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)

    model = torch.load('data/train_mlp/epoch_7.pth')
    model.eval()

    for iters, data_item in enumerate(test_loader):
        # if iters >= 1:
        #     break
        data_item = data_item.squeeze(0)[:, 0:4]
        feature, label = model.data_process(data_item, 20, 10)
        output = model(feature)
        print("--------------------------")
        print("Output: {0}".format(list(output.cpu().detach().numpy().reshape(-1))))
        print("Label: {0}".format(list(label.cpu().detach().numpy().reshape(-1))))
        draw_price(feature, output, label)


if __name__ == '__main__':
    test()