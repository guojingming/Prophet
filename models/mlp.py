import torch
import torch.nn as nn


class AutoRegMLP(nn.Module):
    def __init__(self, feature_count, hidden_layers, feature_seq, output_seq):
        super().__init__()
        self.feature_seq = feature_seq
        self.output_seq = output_seq
        self.feature_count = feature_count
        self.input_layer = nn.Linear(in_features=self.feature_count * self.feature_seq + 1, out_features=hidden_layers[0])
        self.output_layer = nn.Linear(in_features=hidden_layers[-1], out_features=self.feature_count)
        self.hidden_layers = []
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[i - 1], out_features=hidden_layers[i]),
                    nn.LeakyReLU()
                )
            )

    def data_process(self, input_tensor, feature_seq, output_seq):
        batch_size = input_tensor.shape[0]
        seq_length = input_tensor.shape[1]

        assert feature_seq + output_seq == seq_length
        start_value = input_tensor[:, 0:1, 0:1].reshape((batch_size, 1))
        features = input_tensor[:, :feature_seq, 1:].reshape((batch_size, -1))
        labels = input_tensor[:, feature_seq:, 1:].reshape((batch_size, -1))

        output_feature = torch.cat([start_value, features], dim=1)
        return output_feature, labels

    def forward(self, x):
        # Input B * N * : T-1 : Open, High, Low, Close,
        #               : T   : High, Low, Close
        outputs = []
        for circle_time in range(self.output_seq):
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
    def __init__(self, feature_count, hidden_layers, feature_seq, output_seq):
        super().__init__()
        self.feature_seq = feature_seq
        self.output_seq = output_seq
        self.feature_count = feature_count
        self.input_layer = nn.Linear(in_features=self.feature_count * self.feature_seq + 1, out_features=hidden_layers[0])
        self.output_layer = nn.Linear(in_features=hidden_layers[-1], out_features=self.feature_count * self.output_seq)
        self.hidden_layers = []
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[i - 1], out_features=hidden_layers[i]),
                    nn.LeakyReLU()
                )
            )

    def data_process(self, input_tensor, feature_seq, output_seq):
        batch_size = input_tensor.shape[0]
        seq_length = input_tensor.shape[1]

        assert feature_seq + output_seq == seq_length
        start_value = input_tensor[:, 0:1, 0:1].reshape((batch_size, 1))
        features = input_tensor[:, :feature_seq, 1:].reshape((batch_size, -1))
        labels = input_tensor[:, feature_seq:, 1:].reshape((batch_size, -1))

        output_feature = torch.cat([start_value, features], dim=1)
        return output_feature, labels

    def forward(self, x):
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x
