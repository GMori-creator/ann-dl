"""Passo 2 - a rede: MLP em PyTorch escrita pela equipe (nn.Module próprio)."""
import torch
from torch import nn


class MLP(nn.Module):
    """
    Entrada (119 colunas do pipeline do EDA) -> [Linear -> (Norm) -> ReLU -> (Dropout)] x L -> Linear(1).
    A saída é UMA unidade linear (regressão): o alvo é ln(price), que pode ser qualquer real.
    Inicialização He/Kaiming nas camadas com ReLU; bias = 0.
    """

    def __init__(self, in_features, widths=(128, 64), dropout=0.0, norm=None):
        super().__init__()
        layers, prev = [], in_features
        for w in widths:
            layers.append(nn.Linear(prev, w))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(w))
            elif norm == "layer":
                layers.append(nn.LayerNorm(w))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
