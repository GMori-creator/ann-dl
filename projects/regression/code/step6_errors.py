"""Passo 6 - análise de erro (previsões do teste, semente 42) e importância por permutação (na VALIDAÇÃO)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import reg_common as R
import train as T
from mlp import MLP

C = R.C
C.apply_style()
D = R.load_test_matrices()
test = D["test"].copy()
p = np.load(R.RESULTS / "test_pred_seed42.npy")
test["pred"] = R.target_to_price(p)
test["err"] = test["pred"] - test[C.TARGET]
test["abs_err"] = test["err"].abs()
test["ape_ln"] = np.abs(p - D["y_te"])

def grp(col):
    g = test.groupby(col).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), mediana_preco=(C.TARGET, "median"),
                              vies=("err", "mean"), MAE_ln=("ape_ln", "mean")).round(2)
    return g.sort_values("MAE", ascending=False)

by_group, by_room = grp("neighbourhood_group"), grp("room_type")
bins = [0, 75, 100, 150, 200, 300, 500, 10001]
test["faixa"] = pd.cut(test[C.TARGET], bins=bins, right=False)
by_band = grp("faixa").sort_index()
for n, t in (("neighbourhood_group", by_group), ("room_type", by_room), ("faixa_preco", by_band)):
    t.to_csv(R.RESULTS / f"err_{n}.csv"); print(n); print(t.to_string())
cross = test.groupby(["neighbourhood_group", "room_type"]).agg(n=("abs_err", "size"), MAE=("abs_err", "mean")).round(1)
cross.to_csv(R.RESULTS / "err_cross.csv"); print(cross.sort_values("MAE", ascending=False).head(6).to_string())

worst = test.sort_values("abs_err", ascending=False).head(10)[
    ["neighbourhood_group", "neighbourhood", "room_type", "minimum_nights", "availability_365", C.TARGET, "pred", "abs_err"]]
worst.round(1).to_csv(R.RESULTS / "worst10.csv"); print(worst.round(1).to_string())
top1 = test.sort_values("abs_err", ascending=False)
share = top1["abs_err"].head(int(0.01 * len(top1))).sum() / top1["abs_err"].sum()
print("fração do erro absoluto total nos 1% piores:", round(share, 3), "| nos 10% piores:",
      round(top1["abs_err"].head(int(0.1 * len(top1))).sum() / top1["abs_err"].sum(), 3))

# permutação na validação: aumento do MAE (USD) ao embaralhar cada coluna ORIGINAL (antes do pipeline)
d = R.load_train_val()

ch = json.load(open(R.RESULTS / "chosen.json"))
w = tuple(int(x) for x in ch["widths"].strip("()[], ").split(",") if x.strip())
model = MLP(d["X_tr"].shape[1], w, float(np.nan_to_num(ch["dropout"])), None)
model.load_state_dict(torch.load(R.RESULTS / "final_seed42.pt")); model.eval()
va = d["va"]; rng = np.random.default_rng(R.SEED)
def mae(frame):
    X = torch.as_tensor(d["prep"].transform(frame[C.FEATURES]).astype("float32"))
    return R.metrics_usd(d["y_va"], T.predict(model, X).numpy())["MAE"]
base = mae(va); imp = {}
for col in C.FEATURES:
    inc = []
    for r in range(10):
        f = va.copy(); f[col] = rng.permutation(f[col].to_numpy()); inc.append(mae(f) - base)
    imp[col] = (np.mean(inc), np.std(inc))
I = pd.DataFrame(imp, index=["aumento_MAE", "dp"]).T.sort_values("aumento_MAE", ascending=False).round(2)
I.to_csv(R.RESULTS / "perm_importance.csv"); print("MAE val base", round(base, 2)); print(I.to_string())

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
t = by_band
ax[0].bar(range(len(t)), t["MAE"], color=C.BLUE, label="MAE por faixa de preço")
ax[0].set_xticks(range(len(t))); ax[0].set_xticklabels(["<75", "75–99", "100–149", "150–199", "200–299", "300–499", "≥500"], fontsize=7)
ax[0].set(xlabel="faixa do preço real (USD)", ylabel="MAE (USD)", title="Erro por faixa de preço (teste)")
J = I.iloc[::-1]
ax[1].barh(J.index, J["aumento_MAE"], xerr=J["dp"], color=C.BLUE, label="aumento do MAE ao embaralhar")
ax[1].set(xlabel="aumento do MAE na validação (USD)", title="Importância por permutação")
for a in ax: a.legend(frameon=True, facecolor="white", loc="lower right" if a is ax[1] else "upper left")
fig.tight_layout(); fig.savefig(R.FIGURES / "fig04-erro-faixa-permutacao.png", dpi=150, facecolor=C.SURFACE)
