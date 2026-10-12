"""
Exercise 3 - multiclasse (3 classes) com o MESMO mlp.py do Exercício 2.
Só mudam: tamanho da saída (3), função de saída/loss (softmax + CCE) e hiperparâmetros.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import MLP  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"

HIDDEN = [16]
LR = 0.5
EPOCHS = 2000
SEED = 42


def load_data():
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, class_sep=1.0, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = load_data()
    print(f"treino={len(Xtr)}  teste={len(Xte)}  contagem por classe (treino)={np.bincount(ytr)}")

    rng = np.random.default_rng(SEED)
    net = MLP([4] + HIDDEN + [3], output="softmax", rng=rng)
    history = net.fit(Xtr, ytr, lr=LR, epochs=EPOCHS)
    acc_tr, acc_te = net.accuracy(Xtr, ytr), net.accuracy(Xte, yte)
    print(f"\n[C] arquitetura = {net.layer_sizes}  ({net.n_params()} parâmetros)")
    print(f"    lr = {LR}, épocas = {EPOCHS}, loss inicial = {history[0]:.4f}, final = {history[-1]:.4f}")
    print(f"    acurácia treino = {acc_tr:.4f}, teste = {acc_te:.4f}")

    # --- Figura 4: loss por época ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(range(1, EPOCHS + 1), history, color="#1f77b4")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss de treino (entropia cruzada categórica)")
    ax.set_title(f"Figura 4 - Loss de treino por época (MLP {net.layer_sizes})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-loss.png", dpi=150)
    plt.close(fig)

    # --- Figura 5: matriz de confusão no teste --------------------------------
    pred = net.predict(Xte)
    cm = confusion_matrix(yte, pred)
    print(f"\nmatriz de confusão (linhas = verdadeiro, colunas = previsto):\n{cm}")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_yticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Figura 5 - Matriz de confusão (teste)\nacc = {acc_te:.3f}")
    fig.colorbar(im, ax=ax, label="Nº de amostras")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-confusion-matrix.png", dpi=150)
    plt.close(fig)

    # par mais confundido: soma dos erros nos dois sentidos
    best, pair = -1, None
    for i in range(3):
        for j in range(i + 1, 3):
            errs = cm[i, j] + cm[j, i]
            print(f"  confusões entre {i} e {j}: {cm[i, j]} ({i}->{j}) + {cm[j, i]} ({j}->{i}) = {errs}")
            if errs > best:
                best, pair = errs, (i, j)
    print(f"par mais confundido: {pair} com {best} erros")

    # baseline linear multiclasse
    lin = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    print(f"\n[D] LogisticRegression multiclasse: acurácia de teste = {lin.score(Xte, yte):.4f}")


if __name__ == "__main__":
    main()
