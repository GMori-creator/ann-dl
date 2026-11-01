"""Passo 3 - loop de treino com early stopping na validação e restauração dos melhores pesos."""
import copy
import time

import numpy as np
import torch
from torch import nn

from mlp import MLP, n_params

torch.set_num_threads(2)

DEFAULT = dict(widths=(128, 64), dropout=0.0, norm=None, lr=1e-3, weight_decay=0.0,
               batch_size=256, max_epochs=200, patience=20, huber_delta=0.5)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, X, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs]))
    return torch.cat(out)


def fit(Xtr, ytr, Xva, yva, seed=42, **overrides):
    """
    Treina um MLP com AdamW + Huber (em ln(price)). Para quando a perda de validação não melhora
    por `patience` épocas e devolve o modelo com os MELHORES pesos (não os da última época).
    Devolve (model, history) onde history tem train_loss/val_loss/val_mae_usd por época.
    """
    cfg = {**DEFAULT, **overrides}
    set_seed(seed)
    Xtr, ytr, Xva, yva = (torch.as_tensor(a) for a in (Xtr, ytr, Xva, yva))
    model = MLP(Xtr.shape[1], cfg["widths"], cfg["dropout"], cfg["norm"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.HuberLoss(delta=cfg["huber_delta"])
    g = torch.Generator().manual_seed(seed)

    hist = dict(train_loss=[], val_loss=[], val_mae_usd=[])
    best, best_state, wait, best_epoch = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if len(idx) < 2:                      # BatchNorm não aceita lote de 1
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        pv = predict(model, Xva)
        vloss = loss_fn(pv, yva).item()
        hist["train_loss"].append(tot / len(Xtr))
        hist["val_loss"].append(vloss)
        hist["val_mae_usd"].append(float((torch.exp(pv) - torch.exp(yva)).abs().mean()))
        if vloss < best - 1e-5:
            best, best_state, wait, best_epoch = vloss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    hist.update(best_epoch=best_epoch, stopped_epoch=epoch, seconds=time.time() - t0,
                n_params=n_params(model), cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
    return model, hist
