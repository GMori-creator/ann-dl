"""Passo 1 - baselines avaliados na VALIDAÇÃO (teste intocado)."""
import json
import time

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

import reg_common as R

d = R.load_train_val()
print("treino", d["X_tr"].shape, "validação", d["X_va"].shape)
rows = {}


def run(name, model, ytrain=d["y_tr"]):
    t = time.time(); model.fit(d["X_tr"], ytrain)
    rows[name] = R.metrics_usd(d["y_va"], model.predict(d["X_va"]))
    rows[name]["seg"] = round(time.time() - t, 1)
    print(name, {k: round(v, 3) for k, v in rows[name].items()})


run("Dummy média (ln)", DummyRegressor(strategy="mean"))
run("Dummy mediana (ln)", DummyRegressor(strategy="median"))
for a in (0.1, 1, 10, 100):
    run(f"Ridge alpha={a}", Ridge(alpha=a))
run("Floresta aleatória", RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=2, random_state=R.SEED))
run("HistGradientBoosting", HistGradientBoostingRegressor(random_state=R.SEED))
# referência em dólares sem log: prever a média / mediana do preço
price_tr = np.exp(d["y_tr"]); yv = np.exp(d["y_va"])
for nome, c in (("Constante média USD", price_tr.mean()), ("Constante mediana USD", np.median(price_tr))):
    e = c - yv
    rows[nome] = dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                      R2=float(1 - (e ** 2).sum() / ((yv - yv.mean()) ** 2).sum()))
    print(nome, rows[nome])
json.dump(rows, open(R.RESULTS / "baselines_val.json", "w"), indent=1)
