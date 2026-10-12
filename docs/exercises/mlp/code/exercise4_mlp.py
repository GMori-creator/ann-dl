"""
Exercise 4 - uma segunda camada oculta ajuda?
Mesmos dados, lr, épocas e mlp.py do Exercício 3. Muda UMA coisa: a profundidade.
Cada modelo é treinado com 3 sementes de inicialização (42, 43, 44); os dados e o split são fixos.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exercise3_mlp import EPOCHS, LR, load_data  # noqa: E402
from mlp import MLP  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"

SEEDS = [42, 43, 44]
MODELS = {
    "Exercício 3 (1 oculta)": [4, 16, 3],
    "Exercício 4 (2 ocultas)": [4, 16, 16, 3],
}
COLORS = {"Exercício 3 (1 oculta)": "#1f77b4", "Exercício 4 (2 ocultas)": "#d62728"}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = load_data()

    results = {}
    for name, sizes in MODELS.items():
        histories, accs, losses = [], [], []
        for seed in SEEDS:
            net = MLP(sizes, output="softmax", rng=np.random.default_rng(seed))
            hist = net.fit(Xtr, ytr, lr=LR, epochs=EPOCHS)
            histories.append(hist)
            losses.append(hist[-1])
            accs.append(net.accuracy(Xte, yte))
            print(f"{name:26s} seed={seed}: loss final={hist[-1]:.4f}  acc teste={accs[-1]:.4f}")
        results[name] = dict(sizes=sizes, n_params=net.n_params(), histories=np.array(histories),
                             losses=np.array(losses), accs=np.array(accs))

    print("\n=== Tabela (média ± desvio-padrão sobre 3 sementes) ===")
    for name, r in results.items():
        print(f"{name:26s} arq={r['sizes']}  params={r['n_params']}  "
              f"loss treino={r['losses'].mean():.4f}±{r['losses'].std():.4f}  "
              f"acc teste={r['accs'].mean():.4f}±{r['accs'].std():.4f}")

    names = list(results)
    diff = results[names[1]]["accs"].mean() - results[names[0]]["accs"].mean()
    print(f"\ndiferença de acurácia (Ex4 - Ex3), média sobre 3 sementes = {diff:+.4f}")

    # --- Figura 6: curvas de loss dos dois modelos nos mesmos eixos -----------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ep = np.arange(1, EPOCHS + 1)
    for name, r in results.items():
        for h in r["histories"]:
            ax.plot(ep, h, color=COLORS[name], alpha=0.25, linewidth=1)
        ax.plot(ep, r["histories"].mean(axis=0), color=COLORS[name], linewidth=2.2,
                label=f"{name} - média das 3 sementes")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss de treino (entropia cruzada categórica)")
    ax.set_title("Figura 6 - Loss de treino: 1 camada oculta vs. 2 camadas ocultas")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-loss-comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
