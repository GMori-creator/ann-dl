from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)

# --- Dataset I: shifted gaussians -------------------------------------------------
MU_A = np.zeros(5)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(5, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])
N = 500


def generate_dataset_i():
    A = RNG.multivariate_normal(MU_A, SIGMA_A, size=N)
    B = RNG.multivariate_normal(MU_B, SIGMA_B, size=N)
    return A, B


def generate_shell(n: int, radius_mean: float, radius_std: float, dim: int = 5):
    """Amostra pontos numa casca esferica de raio ~N(radius_mean, radius_std) em R^dim."""
    directions = RNG.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = RNG.normal(loc=radius_mean, scale=radius_std, size=(n, 1))
    return radii * directions


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Dataset I ----------------------------------------------------------------
    A, B = generate_dataset_i()
    dist_centers_I = float(np.linalg.norm(MU_A - MU_B))
    pca_I = PCA(n_components=2).fit(np.vstack([A, B]))
    var_I = pca_I.explained_variance_ratio_

    # Dataset II -----------------------------------------------------------------
    core = generate_shell(N, radius_mean=2.0, radius_std=0.4)
    shell = generate_shell(N, radius_mean=5.0, radius_std=0.4)
    dist_centers_II = float(np.linalg.norm(core.mean(axis=0) - shell.mean(axis=0)))
    pca_II = PCA(n_components=2).fit(np.vstack([core, shell]))
    var_II = pca_II.explained_variance_ratio_

    radius_core_mean = float(np.linalg.norm(core, axis=1).mean())
    radius_shell_mean = float(np.linalg.norm(shell, axis=1).mean())

    print(f"Distancia entre centros - Dataset I: {dist_centers_I:.4f}")
    print(f"Distancia entre centros - Dataset II: {dist_centers_II:.4f}")
    print(f"Variancia explicada PC1+PC2 - Dataset I: {var_I[0]+var_I[1]:.4f}")
    print(f"Variancia explicada PC1+PC2 - Dataset II: {var_II[0]+var_II[1]:.4f}")
    print(f"Raio medio - casca interna (core): {radius_core_mean:.4f}")
    print(f"Raio medio - casca externa (shell): {radius_shell_mean:.4f}")

    # Figures --------------------------------------------------------------------
    X_I = pca_I.transform(np.vstack([A, B]))
    X_II = pca_II.transform(np.vstack([core, shell]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(*X_I[:N].T, s=12, alpha=0.6, label="Classe A")
    axes[0].scatter(*X_I[N:].T, s=12, alpha=0.6, label="Classe B")
    axes[0].set_title(f"Dataset I (PC1+PC2 = {var_I[0]+var_I[1]:.1%})")
    axes[0].legend()

    axes[1].scatter(*X_II[:N].T, s=12, alpha=0.6, label="Nucleo")
    axes[1].scatter(*X_II[N:].T, s=12, alpha=0.6, label="Casca")
    axes[1].set_title(f"Dataset II (PC1+PC2 = {var_II[0]+var_II[1]:.1%})")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-pca-projection.png", dpi=150)
    plt.close(fig)

    # Figura 2b - histograma de raio por classe
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(np.linalg.norm(A, axis=1), bins=30, alpha=0.6, label="Classe A")
    axes[0].hist(np.linalg.norm(B, axis=1), bins=30, alpha=0.6, label="Classe B")
    axes[0].set_title("Dataset I - norma ||x||")
    axes[0].set_xlabel("||x||")
    axes[0].legend()

    axes[1].hist(np.linalg.norm(core, axis=1), bins=30, alpha=0.6, label="Nucleo")
    axes[1].hist(np.linalg.norm(shell, axis=1), bins=30, alpha=0.6, label="Casca")
    axes[1].set_title("Dataset II - norma ||x||")
    axes[1].set_xlabel("||x||")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02b-radius-hist.png", dpi=150)
    plt.close(fig)

    g = np.sum(np.vstack([core, shell]) ** 2, axis=1)
    threshold = (radius_core_mean**2 + radius_shell_mean**2) / 2
    labels = np.array([0] * N + [1] * N)
    pred = (g > threshold).astype(int)
    acc = float(np.mean(pred == labels))
    print(f"g(x)=||x||^2, limiar={threshold:.4f}, acuracia={acc:.4f}")


if __name__ == "__main__":
    main()
