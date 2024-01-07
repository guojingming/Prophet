import torch
import os
from torch import optim
from torch.utils.data.dataloader import DataLoader
from models.mlp import AutoRegMLP, MLP
from models.reg_loss import get_loss
from datasets.stock_dataset import BaoStockDataset
from utils.plot_utils import draw_price,save_draw_price


def train():
    ###################### Settings ######################
    output_dir = 'data/train_mlp/'
    train_set = BaoStockDataset('data/A_stock_k_1h_20150701_20210701_v3.pkl', 'train')
    test_set = BaoStockDataset('data/A_stock_k_1h_20150701_20210701_v3.pkl', 'train')

    batch_size = 16
    learning_rate = 1e-3
    epochs = 50
    feature_seq = 20
    output_seq = 10
    ###################### Settings ######################

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)

    model = MLP(feature_count=3,
                hidden_layers=[128, 256, 256, 256, 256, 256, 512, 3],
                feature_seq=feature_seq, output_seq=output_seq)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=len(train_loader))
    criterian = get_loss('reg_loss', 'default')

    for epoch in range(epochs):
        epoch_loss = 0
        model.train() 
        for iters, data_item in enumerate(train_loader):
            optimizer.zero_grad()
            data_item = data_item[:, :, 0:4]
            feature, label = model.data_process(data_item, feature_seq, output_seq)
            output = model(feature)
            loss = criterian(output, label)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            if iters % (len(train_loader) // 10) == 0:
                print("Epoch: {0}, iters: {1}, lr: {2}, loss: {3}".format(epoch, iters, lr_scheduler.get_last_lr()[0],
                                                                          loss.item()))
                save_name = '{0}_{1}.png'.format(epoch, iters)
                save_draw_price(feature[0], output[0], label[0], save_name=save_name)


        epoch_loss = epoch_loss / len(train_loader)
        print("EpochLoss: {0}".format(epoch_loss))

        ckpt_save_path = os.path.join(output_dir, 'epoch_{0}.pth'.format(epoch))
        torch.save(model, ckpt_save_path)
        print("Epoch {0} model saved to {1}".format(epoch, ckpt_save_path))

        '''
        model.eval()
        for iters, data_item in enumerate(test_loader):
            if iters >= 10:
                break
            output = model(data_item['feature_matric'].float())
            label = data_item['label']
            print("--------------------------")
            print("Output: {0}".format(list(output.cpu().detach().numpy().reshape(-1))))
            print("Label: {0}".format(list(label.cpu().detach().numpy().reshape(-1))))
        '''

if __name__ == '__main__':
    train()