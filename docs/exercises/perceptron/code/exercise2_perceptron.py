"""
Exercise 2 - Overlapping Data: the case the perceptron cannot solve.

Reuses perceptron.py UNCHANGED. The only addition to the training loop is
the pocket (best-so-far) bookkeeping, done entirely through the on_update
callback -- train() itself is untouched.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perceptron import accuracy, init_weights, predict, train  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)

MEAN_0 = np.array([3.0, 3.0])
COV_0 = np.array([[1.5, 0.0], [0.0, 1.5]])
MEAN_1 = np.array([4.0, 4.0])
COV_1 = np.array([[1.5, 0.0], [0.0, 1.5]])
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
    ax.scatter(*X[y == 0].T, s=10, alpha=0.5, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=10, alpha=0.5, label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Dados sobrepostos (1000 amostras/classe) -- não separáveis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-overlapping-data.png", dpi=150)
    plt.close(fig)


def plot_both_boundaries(X, y, w_final, b_final, w_pocket, b_pocket, filename):
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, w, b, title in [
        (axes[0], w_final, b_final, "Pesos finais (última época)"),
        (axes[1], w_pocket, b_pocket, "Pesos do bolso (pocket)"),
    ]:
        wrong = predict(w, b, X) != y
        ax.scatter(*X[(y == 0) & ~wrong].T, s=8, alpha=0.4, label="Classe 0")
        ax.scatter(*X[(y == 1) & ~wrong].T, s=8, alpha=0.4, label="Classe 1")
        if wrong.any():
            ax.scatter(*X[wrong].T, s=10, alpha=0.5, color="red", label="Mal classificado")
        if abs(w[1]) > 1e-12:
            ys = -(w[0] * xs + b) / w[1]
            ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.8, label="Fronteira")
        ax.set_xlabel("$x_1$")
        ax.set_title(f"{title}\nacc = {accuracy(w, b, X, y):.3f}")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("$x_2$")

    fig.suptitle("Figura 5 - Fronteira final vs. fronteira do bolso (pocket)")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def plot_accuracy_curves(epoch_acc, pocket_acc_per_epoch, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = range(1, len(epoch_acc) + 1)
    ax.plot(epochs, epoch_acc, marker="o", markersize=3, label="Acurácia (pesos atuais)")
    ax.plot(epochs, pocket_acc_per_epoch, marker="s", markersize=3, label="Melhor acurácia (pocket)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Figura 6 - Acurácia atual vs. melhor acumulada (pocket)")
    ax.set_ylim(0.3, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = generate_data()
    plot_data(X, y)

    w0, b0 = init_weights(2, RNG)
    print(f"Initial weights: w0={w0}, b0={b0}")

    # Pocket bookkeeping lives entirely in this closure -- the training loop
    # in perceptron.train() is not touched, only the on_update hook is used.
    pocket = {"w": w0.copy(), "b": b0, "acc": accuracy(w0, b0, X, y), "epoch": -1}
    pocket_curve_per_update = []  # (epoch, best_acc_so_far), one entry per update

    def track_pocket(w, b, epoch, acc):
        if acc > pocket["acc"]:
            pocket["w"] = w.copy()
            pocket["b"] = b
            pocket["acc"] = acc
            pocket["epoch"] = epoch
        pocket_curve_per_update.append((epoch, pocket["acc"]))

    w_final, b_final, epoch_acc, epoch_updates, n_epochs = train(
        X, y, w0, b0, eta=ETA, max_epochs=100, on_update=track_pocket
    )
    final_acc = accuracy(w_final, b_final, X, y)

    # Down-sample the per-update pocket log to one value per epoch (its last
    # value within that epoch, or the previous epoch's value if an epoch
    # happened to produce zero updates -- the pocket can only improve).
    pocket_acc_per_epoch = []
    running_best = pocket_curve_per_update[0][1] if pocket_curve_per_update else pocket["acc"]
    idx = 0
    for e in range(n_epochs):
        while idx < len(pocket_curve_per_update) and pocket_curve_per_update[idx][0] == e:
            running_best = pocket_curve_per_update[idx][1]
            idx += 1
        pocket_acc_per_epoch.append(running_best)

    print(f"\n=== Final weights (after {n_epochs} epochs, cap reached) ===")
    print(f"w_final = {w_final}")
    print(f"b_final = {b_final:.6f}")
    print(f"accuracy(final) = {final_acc:.4f}")

    print(f"\n=== Pocket (best-so-far) weights ===")
    print(f"w_pocket = {pocket['w']}")
    print(f"b_pocket = {pocket['b']:.6f}")
    print(f"accuracy(pocket) = {pocket['acc']:.4f}")
    print(f"found at epoch (1-indexed) = {pocket['epoch'] + 1}")
    print(f"epochs run = {n_epochs} (no early stop -- data is not separable)")

    # sanity: ||x|| scale mentioned in the statement's hint for item D1
    x_norms = np.linalg.norm(X, axis=1)
    print(f"\nmean ||x|| = {x_norms.mean():.3f} (statement's hint uses ~5)")

    plot_both_boundaries(X, y, w_final, b_final, pocket["w"], pocket["b"], "fig05-boundaries.png")
    plot_accuracy_curves(epoch_acc, pocket_acc_per_epoch, "fig06-accuracy-curves.png")


if __name__ == "__main__":
    main()
