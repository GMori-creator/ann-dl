"""
Exercise 1 - Separable Data: the case the perceptron was designed for.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perceptron import accuracy, init_weights, predict, train  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # same rng used for data AND weight init below

MEAN_0 = np.array([1.5, 1.5])
COV_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
MEAN_1 = np.array([5.0, 5.0])
COV_1 = np.array([[0.5, 0.0], [0.0, 0.5]])
N_PER_CLASS = 1000
ETA = 0.01


def generate_data():
    X0 = RNG.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
    X1 = RNG.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])
    return X, y


def plot_data(X, y):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[y == 0].T, s=10, alpha=0.6, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=10, alpha=0.6, label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Dados linearmente separáveis (1000 amostras/classe)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-separable-data.png", dpi=150)
    plt.close(fig)


def plot_boundary(X, y, w, b, filename, title):
    preds = predict(w, b, X)
    wrong = preds != y
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X[(y == 0) & ~wrong].T, s=10, alpha=0.6, label="Classe 0")
    ax.scatter(*X[(y == 1) & ~wrong].T, s=10, alpha=0.6, label="Classe 1")
    if wrong.any():
        ax.scatter(*X[wrong].T, s=45, facecolors="none", edgecolors="red",
                    linewidths=1.6, label="Mal classificado")
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    if abs(w[1]) > 1e-12:
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.5,
                label=r"Fronteira $\mathbf{w}\cdot\mathbf{x}+b=0$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def plot_accuracy(history, filename, title):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = generate_data()
    plot_data(X, y)

    # Draw the initial weights ONCE, from the same rng used for the data,
    # then reuse them for both eta runs below so eta is the only thing
    # that changes between the two trainings (item D2).
    w0, b0 = init_weights(2, RNG)
    print(f"Initial weights (shared by both eta runs): w0={w0}, b0={b0}")

    # --- C: train with eta = 0.01 ---
    w, b, acc_hist, upd_hist, n_epochs = train(X, y, w0, b0, eta=ETA, max_epochs=100)
    final_acc = accuracy(w, b, X, y)
    print(f"\n=== eta={ETA} ===")
    print(f"final w = {w}")
    print(f"final b = {b:.6f}")
    print(f"epochs to convergence = {n_epochs}")
    print(f"final accuracy = {final_acc:.4f}")
    print(f"updates per epoch = {upd_hist}")

    plot_boundary(X, y, w, b, "fig02-decision-boundary.png",
                  f"Fronteira de decisão (eta={ETA}, {n_epochs} épocas)")
    plot_accuracy(acc_hist, "fig03-accuracy-curve.png",
                  f"Acurácia por época (eta={ETA})")

    # --- D2: re-run with eta = 1.0, same w0, b0, same data ---
    w2, b2, acc_hist2, upd_hist2, n_epochs2 = train(X, y, w0, b0, eta=1.0, max_epochs=100)
    final_acc2 = accuracy(w2, b2, X, y)
    print(f"\n=== eta=1.0 ===")
    print(f"final w = {w2}")
    print(f"final b = {b2:.6f}")
    print(f"epochs to convergence = {n_epochs2}")
    print(f"final accuracy = {final_acc2:.4f}")
    print(f"updates per epoch = {upd_hist2}")

    dir1 = w / np.linalg.norm(w)
    dir2 = w2 / np.linalg.norm(w2)
    cos_sim = float(np.dot(dir1, dir2))
    print(f"\ndirection w/||w|| (eta=0.01) = {dir1}")
    print(f"direction w/||w|| (eta=1.0)  = {dir2}")
    print(f"cosine similarity between directions = {cos_sim:.6f}")


if __name__ == "__main__":
    main()
