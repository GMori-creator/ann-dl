from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}
N_PER_CLASS = 100


def generate(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Amostra 100 pontos por classe, com os desvios multiplicados por ``scale``."""
    xs, ys = [], []
    for label, params in CLASSES.items():
        mean = np.asarray(params["mean"])
        std = np.asarray(params["std"]) * scale
        xs.append(RNG.normal(mean, std, size=(N_PER_CLASS, 2)))
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def separation_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Distancia media entre centroides dividida pela dispersao media intraclasse."""
    centroids = np.stack([X[y == c].mean(axis=0) for c in CLASSES])
    spreads = np.array([np.linalg.norm(X[y == c] - centroids[c], axis=1).mean() for c in CLASSES])
    pairwise = [
        np.linalg.norm(centroids[i] - centroids[j])
        for i in CLASSES
        for j in CLASSES
        if i < j
    ]
    return float(np.mean(pairwise) / spreads.mean())


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Fracao de pontos cujo centroide mais proximo nao e o da propria classe."""
    centroids = np.stack([X[y == c].mean(axis=0) for c in CLASSES])
    d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    return float(np.mean(nearest != y))


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = generate()
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in CLASSES:
        ax.scatter(*X[y == c].T, s=14, alpha=0.75, label=f"Classe {c}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Nuvens de pontos gaussianas (scale = 1.0)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
    plt.close(fig)  # (2)!

    print("=== Separation ratio e mixing rate por escala ===")
    ratios, mixings, tested_scales = [], [], (0.5, 1.0, 2.0)
    for scale in tested_scales:
        Xs, ys = generate(scale)
        sr = separation_ratio(Xs, ys)
        mr = mixing_rate(Xs, ys)
        ratios.append(sr)
        mixings.append(mr)
        print(f"scale={scale:>4} | separation ratio = {sr:.4f} | mixing rate = {mr:.4f}")

    # Figura 1b - fronteiras tipo Voronoi (distancia minima aos centroides)
    centroids = np.stack([X[y == c].mean(axis=0) for c in CLASSES])
    xs_grid = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 400)
    ys_grid = np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 300)
    XX, YY = np.meshgrid(xs_grid, ys_grid)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    d = np.linalg.norm(grid_pts[:, None, :] - centroids[None, :, :], axis=2)
    region = np.argmin(d, axis=1).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(XX, YY, region, levels=np.arange(-0.5, 4, 1), alpha=0.15)
    ax.contour(XX, YY, region, levels=np.arange(-0.5, 4, 1), colors="black", linewidths=1.0, linestyles="--")
    for c in CLASSES:
        ax.scatter(*X[y == c].T, s=14, alpha=0.75, label=f"Classe {c}")
        ax.scatter(*centroids[c], marker="X", s=150, edgecolor="black", linewidth=1.2, zorder=5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Fronteiras tipo Voronoi (distancia minima aos centroides)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01b-voronoi.png", dpi=150)
    plt.close(fig)

    # Figura 1c - mixing rate por escala
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(tested_scales, mixings, marker="o", color="#333333")
    ax.set_xlabel("scale")
    ax.set_ylabel("mixing rate")
    ax.set_title("Mixing rate vs. fator de escala")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01c-mixing-rate.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
