"""Figura 2 - ablação de regularização, a partir de results/ablation.csv (gerado por step4_search.py)."""
import matplotlib.pyplot as plt
import pandas as pd

import reg_common as R

C = R.C
C.apply_style()
A = pd.read_csv(R.RESULTS / "ablation.csv")
fig, ax = plt.subplots(figsize=(8.5, 4))
x = range(len(A))
ax.plot(x, A["train_loss"], "o", color=C.BLUE, ms=8, label="treino")
ax.plot(x, A["val_loss"], "s", color=C.ORANGE, ms=8, label="validação")
for i in x:
    ax.plot([i, i], [A["train_loss"][i], A["val_loss"][i]], color=C.AXIS, lw=1, zorder=0)
ax.set_xticks(list(x)); ax.set_xticklabels([c.replace(" + ", "\n+ ") for c in A["config"]], fontsize=7.5)
ax.set(ylabel="perda Huber na melhor época (ln preço)", title="Ablação de regularização: treino vs validação")
ax.legend(frameon=True, facecolor="white", loc="upper left")
fig.tight_layout(); fig.savefig(R.FIGURES / "fig02-ablacao.png", dpi=150, facecolor=C.SURFACE)
