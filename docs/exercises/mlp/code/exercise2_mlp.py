"""
Exercise 2 - classificação binária que uma reta não resolve.
Baseline linear (LogisticRegression) vs. MLP do zero (mlp.py).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import MLP  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
C0, C1 = "#1f77b4", "#ff7f0e"

HIDDEN = [8]
LR = 0.5
EPOCHS = 2000


def load_data():
    centers = [[0, 0], [3, 3], [-3, -3]]
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                      cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)  # classe 1 = os dois clusters laterais
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), X, y


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    (Xtr, Xte, ytr, yte), X, y = load_data()
    print(f"treino={len(Xtr)}  teste={len(Xte)}  proporção classe 1 (treino)={ytr.mean():.3f}")

    # --- Figura 1: todos os 1000 pontos --------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[y == 0].T, s=10, alpha=0.6, color=C0, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=10, alpha=0.6, color=C1, label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 1 - Dados (1000 pontos): classe 1 = dois clusters laterais")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-data.png", dpi=150)
    plt.close(fig)

    # --- B: baseline linear ---------------------------------------------------
    lin = LogisticRegression().fit(Xtr, ytr)
    acc_lin = lin.score(Xte, yte)
    print(f"\n[B] LogisticRegression: acurácia de teste = {acc_lin:.4f}")

    # --- C: MLP do zero -------------------------------------------------------
    rng = np.random.default_rng(42)
    net = MLP([2] + HIDDEN + [1], output="sigmoid", rng=rng)
    history = net.fit(Xtr, ytr, lr=LR, epochs=EPOCHS)
    acc_tr, acc_te = net.accuracy(Xtr, ytr), net.accuracy(Xte, yte)
    print(f"\n[C] arquitetura = {net.layer_sizes}  ({net.n_params()} parâmetros)")
    print(f"    lr = {LR}, épocas = {EPOCHS}, loss inicial = {history[0]:.4f}, final = {history[-1]:.4f}")
    print(f"    acurácia treino = {acc_tr:.4f}, teste = {acc_te:.4f}")

    # --- Figura 2: loss por época ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(range(1, EPOCHS + 1), history, color=C0)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss de treino (BCE)")
    ax.set_title(f"Figura 2 - Loss de treino por época (MLP {net.layer_sizes})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-loss.png", dpi=150)
    plt.close(fig)

    # --- Figura 3: fronteira de decisão sobre os pontos de teste --------------
    pad = 1.0
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 400)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 400)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.c_[GX.ravel(), GY.ravel()]
    Z = net.predict(grid).reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.contourf(GX, GY, Z, levels=[-0.5, 0.5, 1.5], colors=[C0, C1], alpha=0.18)
    ax.contour(GX, GY, Z, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(*Xte[yte == 0].T, s=18, color=C0, edgecolors="white", linewidths=0.4, label="Classe 0 (teste)")
    ax.scatter(*Xte[yte == 1].T, s=18, color=C1, edgecolors="white", linewidths=0.4, label="Classe 1 (teste)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Figura 3 - Fronteira do MLP nos pontos de teste (acc = {acc_te:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-decision-boundary.png", dpi=150)
    plt.close(fig)

    # Para a análise: a reta da regressão logística, só para citar nos números
    print(f"\n[D] baseline: w = {lin.coef_[0]}, b = {lin.intercept_[0]:.4f}")
    print(f"    fração prevista como classe 1 no teste pelo baseline = {lin.predict(Xte).mean():.3f}")


if __name__ == "__main__":
    main()
