import torch
import torch.nn as nn

def get_model(input_channels, output_channels):
    return MLP(input_channels=input_channels, output_channels=output_channels, hidden_layers=[128, 512, 2048, 4096, 2048, 1024, 512])

#def get_ar_model():


class StockPrediction(nn.Module):
    def __init__(self, feature_count, hidden_layers, train_seq, test_seq):
        super().__init__()
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.feature_count = feature_count
        self.input_layer = nn.Linear(in_features=self.feature_count * self.train_seq + 1, out_features=hidden_layers[0])
        self.output_layer = nn.Linear(in_features=hidden_layers[-1], out_features=self.feature_count)
        self.hidden_layers = []
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[i - 1], out_features=hidden_layers[i]),
                    #nn.Tanh()
                )
            )
            
    def data_process(self, input_tensor, feature_seq, label_seq):
        assert feature_seq + label_seq == input_tensor.shape[0]
        start_value = input_tensor[0:1, 0:1].reshape(-1)
        features = input_tensor[:feature_seq, 1:].reshape(-1)
        labels = input_tensor[feature_seq:, 1:].reshape(-1)

        output_feature = torch.cat([start_value, features], dim=0)
        return output_feature, labels

    def forward(self, x):
        # Input B * N * : T-1 : Open, High, Low, Close,
        #               : T   : High, Low, Close
        outputs = []
        for circle_time in range(self.test_seq):
            hidden_features = self.input_layer(x)
            for hidden_layer in self.hidden_layers:
                hidden_features = hidden_layer(hidden_features)
            output = self.output_layer(hidden_features)
            outputs.append(output)
            x = x[self.feature_count:]
            x = torch.cat([x, output], dim=0)
        return torch.cat(outputs, dim=0)

    def add_sub_module(self, new_module):
        self.input_layer.add_module(new_module)


class MLP(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_layers=[1]):
        super().__init__()

        self.input_layer = nn.Linear(in_features=input_channels, out_features=hidden_layers[0])
        self.output_layer = nn.Linear(in_features=hidden_layers[-1], out_features=output_channels)
        self.hidden_layers = []
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[i-1],out_features=hidden_layers[i]),
                    nn.ReLU()
                )
            )

    def forward(self, x):
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x
    

    def add_sub_module(self, new_module):
        self.input_layer.add_module(new_module)


if __name__ == '__main__':
    import torch
    import numpy as np
    array1 = [
        3.7477644900,3.7616967000,3.7384763500,3.7477644900,
        3.7570526300,3.7570526300,3.7431204200,3.7524085600,
        3.7524085600,3.7709848400,3.7384763500,3.7570526300,
        3.7570526300,3.7756289100,3.7477644900,3.7709848400,
        3.7709848400,3.7849170500,3.7570526300,3.7756289100,
        3.7756289100,3.7756289100,3.7616967000,3.7663407700,
        3.7663407700,3.7709848400,3.7524085600,3.7570526300,
        3.7616967000,3.7756289100,3.7570526300,3.7570526300,
        3.7570526300,3.7942051900,3.7524085600,3.7802729800,
        3.7802729800,3.7849170500,3.7709848400,3.7756289100,
        3.7756289100,3.7802729800,3.7663407700,3.7709848400,
        3.7663407700,3.7709848400,3.7431204200,3.7477644900,
        3.7477644900,3.7524085600,3.7384763500,3.7524085600,
        3.7477644900,3.7524085600,3.7384763500,3.7477644900,
        3.7477644900,3.7616967000,3.7431204200,3.7477644900,
        3.6781034400,3.6920356500,3.6641712300,3.6781034400,
        3.6781034400,3.6827475100,3.6641712300,3.6688153000,
        3.6688153000,3.6873915800,3.6641712300,3.6734593700,
        3.6734593700,3.7013237900,3.6734593700,3.7013237900,
        3.6827475100,3.6873915800,3.6409508800,3.6548830900,
        3.6548830900,3.6595271600,3.5991542500,3.6037983200,
        3.6037983200,3.6084423900,3.5805779700,3.5898661100,
        3.5852220400,3.5898661100,3.5527135500,3.5620016900,
        3.5620016900,3.6037983200,3.5620016900,3.5759339000,
        3.5805779700,3.5852220400,3.5620016900,3.5759339000,
        3.5759339000,3.5898661100,3.5712898300,3.5898661100,
        3.5898661100,3.5945101800,3.5759339000,3.5759339000,
        3.5712898300,3.5945101800,3.5712898300,3.5852220400,
        3.5805779700,3.5898661100,3.5666457600,3.5712898300,
        3.5712898300,3.5945101800,3.5620016900,3.5852220400,
    ]

    from torch import optim


    array1 = np.array(array1).reshape((-1, 4))
    start_price = array1[0, 0]
    label = array1[20:, 1:].reshape(-1)
    array1 = array1[:20, 1:].reshape(-1)
    array1 = np.concatenate([np.array([start_price]), array1])

    tensor1 = torch.from_numpy(array1).float()
    label = torch.from_numpy(label).float()

    model = StockPrediction(feature_count=3, hidden_layers=[512, 1024, 512, 3], train_seq=20, test_seq=10)

    epochs = 2000

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=epochs)
    criterian = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(tensor1)
        #output = torch.cat(output, dim=0)
        loss = criterian(output, label)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print("epoch: {0}, loss: {1}".format(epoch, loss))