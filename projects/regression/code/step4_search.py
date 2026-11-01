"""Passo 4 - (a) ablação de regularização; (b) busca de configurações. Tudo medido na VALIDAÇÃO."""
import itertools
import json

import matplotlib.pyplot as plt
import pandas as pd

import reg_common as R
import train as T

C = R.C
C.apply_style()
d = R.load_train_val()
args = (d["X_tr"], d["y_tr"], d["X_va"], d["y_va"])


def run(name, seed=R.SEED, **cfg):
    _, h = T.fit(*args, seed=seed, **cfg)
    b = h["best_epoch"] - 1
    return dict(config=name, **{k: (str(v) if k == "widths" else v) for k, v in cfg.items()},
                params=h["n_params"], best_epoch=h["best_epoch"], train_loss=h["train_loss"][b],
                val_loss=h["val_loss"][b], val_MAE=h["val_mae_usd"][b], gap=h["val_loss"][b] - h["train_loss"][b])


# (a) ablação: rede base de 128-64 e cada técnica sozinha / combinada
abl = [run("sem regularização"),
       run("weight decay 0,1", weight_decay=0.1),
       run("dropout 0,2", dropout=0.2),
       run("BatchNorm", norm="batch"),
       run("LayerNorm", norm="layer"),
       run("weight decay 0,1 + dropout 0,2", weight_decay=0.1, dropout=0.2),
       run("dropout 0,2 + BatchNorm", dropout=0.2, norm="batch")]
A = pd.DataFrame(abl)
A.to_csv(R.RESULTS / "ablation.csv", index=False)
print(A[["config", "best_epoch", "train_loss", "val_loss", "val_MAE", "gap"]].round(4).to_string())

# (b) busca em grade: 3 larguras/profundidades x 3 lr x 4 regularizações = 36 configurações
grid = []
for widths, lr, (reg, kw) in itertools.product(
        [(64,), (128, 64), (256, 128, 64)], [3e-4, 1e-3, 3e-3],
        [("nenhuma", {}), ("dropout 0,2", dict(dropout=0.2)),
         ("wd 0,1", dict(weight_decay=0.1)), ("dropout 0,2 + wd 0,1", dict(dropout=0.2, weight_decay=0.1))]):
    name = f"{'-'.join(map(str, widths))} | lr {lr:g} | {reg}"
    grid.append(run(name, widths=widths, lr=lr, **kw))
S = pd.DataFrame(grid).sort_values("val_MAE").reset_index(drop=True)
S.to_csv(R.RESULTS / "search.csv", index=False)
print(S[["config", "params", "best_epoch", "val_loss", "val_MAE"]].head(12).round(3).to_string())
best = S.iloc[0]
json.dump(dict(config=best["config"], widths=best["widths"], lr=float(best["lr"]),
               dropout=float(best.get("dropout", 0) or 0), weight_decay=float(best.get("weight_decay", 0) or 0),
               val_MAE=float(best["val_MAE"])), open(R.RESULTS / "chosen.json", "w"), indent=1)
print("ESCOLHIDA:", best["config"], round(best["val_MAE"], 2))

