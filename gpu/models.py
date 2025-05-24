import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os,sys
from struct import unpack

dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)

from functions import *

class CNN3_128_9_NN2_121_128(nn.Module):
    def __init__(self, weight_parameters_path):
        super().__init__()
        self.params = {}

        with open(weight_parameters_path, "rb") as f:
            binary_data = f.read()
        data = unpack(">2931714f", binary_data)

        self.conv1 = nn.Conv1d(in_channels=553, out_channels=128, kernel_size=9, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, padding=0)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, padding=0)
        
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=121, padding=0)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self._load_weights(data)

    def _load_weights(self, data):
        self.conv1.weight.data.copy_(torch.tensor(data[0:637056], dtype=torch.float32).reshape(9, 553, 128).permute(2, 1, 0))
        self.conv1.bias.data.copy_(torch.tensor(data[637056:637184], dtype=torch.float32))

        self.conv2.weight.data.copy_(torch.tensor(data[637184:784640], dtype=torch.float32).reshape(9, 128, 128).permute(2, 1, 0))
        self.conv2.bias.data.copy_(torch.tensor(data[784640:784768], dtype=torch.float32))

        self.conv3.weight.data.copy_(torch.tensor(data[784768:932224], dtype=torch.float32).reshape(9, 128, 128).permute(2, 1, 0))
        self.conv3.bias.data.copy_(torch.tensor(data[932224:932352], dtype=torch.float32))

        self.conv4.weight.data.copy_(torch.tensor(data[932352:2914816], dtype=torch.float32).reshape(121, 128, 128).permute(2, 1, 0))
        self.conv4.bias.data.copy_(torch.tensor(data[2914816:2914944], dtype=torch.float32))

        self.fc1.weight.data.copy_(torch.tensor(data[2914944:2931328], dtype=torch.float32).reshape(128, 128).T)
        self.fc1.bias.data.copy_(torch.tensor(data[2931328:2931456], dtype=torch.float32))

        self.fc2.weight.data.copy_(torch.tensor(data[2931456:2931712], dtype=torch.float32).reshape(128, 2).T)
        self.fc2.bias.data.copy_(torch.tensor(data[2931712:2931714], dtype=torch.float32))

    def forward(self, x):
        x = F.pad(x, (72, 72), "constant", 0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class DARUMA:
    def __init__(self,gpu_id):
        weight_parameters_path = os.path.join(os.path.dirname(__file__), 'data', 'CNN3_128_9_NN2_121_128.weight')

        if gpu_id == "cpu":
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            torch.cuda.set_device(int(gpu_id))
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')

        print('Using {} device'.format(self.device))

        self.model = CNN3_128_9_NN2_121_128(weight_parameters_path).to(self.device)
        self.model.eval()

        self.feature = load_AAindex()

    def predict_from_sequence(self, seq, threshold=0.5, smoothing_window=17, remove_short_regions=True):
        """
        アミノ酸配列から予測
        """
        x = torch.tensor([self.feature[res] for res in seq], dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(self.device)  # [batch_size, channels, sequence_length]

        with torch.no_grad():
            prob = self.model(x).squeeze(0)[:, 1].tolist()

        if smoothing_window:
            prob = smoothing(smoothing_window, prob)

        pred = classify(prob, threshold=threshold)

        if remove_short_regions:
            pred = remove_short_idr(pred)
            pred = remove_short_stru(pred)

        return prob, pred

