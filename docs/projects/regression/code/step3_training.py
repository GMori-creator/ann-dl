"""Passo 3 - rede base (sem regularização): curvas de treino/validação por época."""
import json

import matplotlib.pyplot as plt

import reg_common as R
import train as T

C = R.C
C.apply_style()
d = R.load_train_val()
model, h = T.fit(d["X_tr"], d["y_tr"], d["X_va"], d["y_va"], seed=R.SEED)
ep = range(1, len(h["train_loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ep, h["train_loss"], color=C.BLUE, label="treino")
ax[0].plot(ep, h["val_loss"], color=C.ORANGE, label="validação")
ax[0].axvline(h["best_epoch"], color=C.MUTED, ls="--", label=f"melhor época ({h['best_epoch']})")
ax[0].set(xlabel="época", ylabel="perda Huber (ln preço)", title="Perda por época")
ax[1].plot(ep, h["val_mae_usd"], color=C.ORANGE, label="MAE validação (USD)")
ax[1].axvline(h["best_epoch"], color=C.MUTED, ls="--", label="melhor época")
ax[1].set(xlabel="época", ylabel="MAE (USD)", title="MAE na validação")
for a in ax:
    a.legend(frameon=True, facecolor="white")
fig.tight_layout()
C.FIGURES = R.FIGURES
fig.savefig(R.FIGURES / "fig01-curvas-base.png", dpi=150, facecolor=C.SURFACE)
json.dump({k: v for k, v in h.items()}, open(R.RESULTS / "base_history.json", "w"))
print({k: h[k] for k in ("best_epoch", "stopped_epoch", "n_params", "seconds")},
      "train/val loss na melhor época:", round(h["train_loss"][h["best_epoch"] - 1], 4), round(h["val_loss"][h["best_epoch"] - 1], 4),
      "última época:", round(h["train_loss"][-1], 4), round(h["val_loss"][-1], 4),
      "MAE val", round(h["val_mae_usd"][h["best_epoch"] - 1], 2))
