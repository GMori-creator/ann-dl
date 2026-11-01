"""
Passo 5 - avaliação NO TESTE, feita UMA vez. A configuração já foi escolhida na validação (results/chosen.json).
Treina a rede escolhida com 5 sementes (o conjunto de teste só entra em predict) e compara com os baselines.
Um arquivo-trava impede rodar duas vezes por engano.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

import reg_common as R
import train as T

C = R.C
C.apply_style()
LOCK = R.RESULTS / "TEST_USED.lock"
if LOCK.exists():
    raise SystemExit("O teste já foi usado (apague results/TEST_USED.lock só se for refazer TUDO de propósito).")

ch = json.load(open(R.RESULTS / "chosen.json"))
cfg = dict(widths=tuple(int(x) for x in ch["widths"].strip("()[], ").split(",") if x.strip()),
           lr=ch["lr"], dropout=float(np.nan_to_num(ch["dropout"])), weight_decay=float(np.nan_to_num(ch["weight_decay"])))
crit = json.load(open(R.RESULTS / "criterion.json"))
D = R.load_test_matrices()
d = R.load_train_val()
assert np.allclose(d["X_tr"], D["prep"].transform(D["tr"][R.C.FEATURES]))
X_te, y_te = D["X_te"], D["y_te"]
LOCK.write_text("teste usado")

rows, preds = [], {}
SEEDS = [42, 43, 44, 45, 46]
for s in SEEDS:
    m, h = T.fit(d["X_tr"], d["y_tr"], d["X_va"], d["y_va"], seed=s, **cfg)
    p = T.predict(m, torch.as_tensor(X_te)).numpy()
    rows.append(dict(seed=s, best_epoch=h["best_epoch"], stopped_epoch=h["stopped_epoch"], **R.metrics_usd(y_te, p)))
    preds[s] = p
    if s == R.SEED:
        torch.save(m.state_dict(), R.RESULTS / "final_seed42.pt")
        json.dump(h, open(R.RESULTS / "final_history.json", "w"))
net = pd.DataFrame(rows)
net.to_csv(R.RESULTS / "test_network_seeds.csv", index=False)
print(net.round(3).to_string())

base_models = {"Dummy média (ln)": DummyRegressor(strategy="mean"), "Ridge (alpha=10)": Ridge(alpha=10),
               "Floresta aleatória": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=2, random_state=R.SEED),
               "HistGradientBoosting": HistGradientBoostingRegressor(random_state=R.SEED)}
table = {}
for k, mdl in base_models.items():
    mdl.fit(d["X_tr"], d["y_tr"])
    table[k] = R.metrics_usd(y_te, mdl.predict(X_te))
    if k.startswith("Ridge"):
        preds["ridge"] = mdl.predict(X_te)
table["MLP (média das 5 sementes)"] = {k: float(net[k].mean()) for k in ("MAE", "RMSE", "R2")}
table["MLP (desvio)"] = {k: float(net[k].std(ddof=1)) for k in ("MAE", "RMSE", "R2")}
json.dump(table, open(R.RESULTS / "test_table.json", "w"), indent=1)
for k, v in table.items():
    print(k, {a: round(b, 3) for a, b in v.items()})
ridge_mae = table["Ridge (alpha=10)"]["MAE"]
print("critério (limiar sobre VALIDAÇÃO):", round(crit["threshold"], 2), "| MAE teste rede:", round(net.MAE.mean(), 2),
      "| redução vs Ridge no teste: %.1f%%" % (100 * (1 - net.MAE.mean() / ridge_mae)))

# figuras com a semente 42
p = preds[R.SEED]
yt, yp = R.target_to_price(y_te), R.target_to_price(p)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].scatter(yt, yp, s=5, alpha=0.3, color=C.BLUE, label="anúncios")
lim = [10, 1000]; ax[0].plot(lim, lim, color=C.ORANGE, label="previsão perfeita (y=x)")
ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="preço real (USD, escala log)", ylabel="preço previsto (USD, escala log)", title="Previsto vs real (teste)")
res = p - y_te
ax[1].scatter(p, res, s=5, alpha=0.3, color=C.BLUE, label="resíduos")
ax[1].axhline(0, color=C.ORANGE, label="resíduo zero")
ax[1].set(xlabel="ln(preço) previsto", ylabel="resíduo em ln (previsto − real)", title="Resíduos vs previsto (teste)")
for a in ax: a.legend(frameon=True, facecolor="white", loc="upper left")
fig.tight_layout(); fig.savefig(R.FIGURES / "fig03-previsto-real-residuos.png", dpi=150, facecolor=C.SURFACE)
np.save(R.RESULTS / "test_pred_seed42.npy", p)
