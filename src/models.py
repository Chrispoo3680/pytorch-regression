import torch
from torch import nn 

class RegressionModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_nodes=8):
        super().__init__() 

        self.layer1 = nn.Linear(input_dim, hidden_nodes)
        self.layer2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.layer3 = nn.Linear(hidden_nodes, hidden_nodes)
        self.layer4 = nn.Linear(hidden_nodes, hidden_nodes)
        self.layer5 = nn.Linear(hidden_nodes, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.layer5(x)

        return x