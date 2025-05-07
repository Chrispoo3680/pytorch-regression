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


class ExpRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Learnable parameters
        self.a = nn.Parameter(torch.randn(1))  # a can be positive or negative
        self.log_b = nn.Parameter(
            torch.randn(1)
        )  # Learning log_b instead of b directly

    def forward(self, X):
        b = torch.exp(self.log_b)  # Ensures b is always positive, allows b < 1
        return self.a * torch.pow(b, X)  # a * b^x

    def __str__(self):
        return f"$f(x) = {float(self.a):.3f} * {float(torch.exp(self.log_b)):.3f}^x$"


class LogistRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Learnable parameters
        self.C = nn.Parameter(torch.randn(1))
        self.log_a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, X):
        a = torch.exp(self.log_a)
        return self.C / (1 + (a * torch.pow(torch.e, torch.neg(self.b) * X)))

    def __str__(self):
        return rf"$f(x) = \frac{{{float(self.C):.3f}}}{{1 + {float(torch.exp(self.log_a)):.3f} \cdot e^{{{float(torch.neg(self.b)):.3f}x}}}}$"


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return (self.a * x) + self.b

    def __str__(self):
        return f"$f(x) = {float(self.a):.3f}x + {float(self.b):.3f}$"


class QuadraticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return (self.a * torch.pow(x, 2)) + (self.b * x) + self.c

    def __str__(self):
        return f"$f(x) = {float(self.a):.3f}x^2 + {float(self.b):.3f}x + {float(self.c):.3f}$"
