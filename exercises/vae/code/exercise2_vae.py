"""
Exercise 2 - espaço latente 2-D: mesmo modelo, dados, semente, beta e orçamento, com d = 2.
Figuras 4 (codificação do teste), 5 (plano decodificado) e 6 (interpolação).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae as V  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_test = V.load_mnist()

    # --- A: modelo d = 2 (mesmo orçamento do d = 16) -------------------------
    model2, hist2 = V.train_vae(2, X_train, X_val)
    V.save_model(model2, 2)
    model16 = V.get_model(16, X_train, X_val)   # o mesmo modelo do Exercício 1

    r2, k2 = V.evaluate(model2, X_test)
    r16, k16 = V.evaluate(model16, X_test)
    print(f"\n[A] teste d= 2: reconstrução = {r2:.2f}, KL = {k2:.2f}, -ELBO = {r2 + k2:.2f} nats")
    print(f"    teste d=16: reconstrução = {r16:.2f}, KL = {k16:.2f}, -ELBO = {r16 + k16:.2f} nats")
    print(f"    KL por dimensão (d=2, teste): {np.round(V.kl_per_dim(model2, X_test), 3)}")

    # --- B: Figura 4 - mu de todo o teste, colorido por dígito ----------------
    mu = V.encode_mu(model2, X_test).numpy()
    y = y_test.numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.get_cmap("tab10")
    for k in range(10):
        ax.scatter(mu[y == k, 0], mu[y == k, 1], s=5, alpha=0.6, color=cmap(k), label=str(k))
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("Figura 4 - Média posterior $\\mu$ das 10.000 imagens de teste (d = 2)")
    ax.legend(title="Dígito", markerscale=3, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-latent-scatter.png", dpi=150)
    plt.close(fig)

    # evidência numérica da sobreposição: kNN em 2-D (ajustado em mu do treino, avaliado no teste)
    mu_train = V.encode_mu(model2, X_train).numpy()
    # rótulos do treino: refaz o mesmo split para obter y de treino
    from torchvision import datasets
    from torch.utils.data import random_split
    y_full = datasets.MNIST(V.DATA_DIR, train=True, download=True).targets
    tr_idx, _ = random_split(range(60000), [50_000, 10_000], generator=torch.Generator().manual_seed(42))
    y_train = y_full[list(tr_idx)].numpy()
    knn = KNeighborsClassifier(n_neighbors=15).fit(mu_train, y_train)
    pred = knn.predict(mu)
    cm = confusion_matrix(y, pred)
    print(f"\n[E] kNN (k=15) sobre mu em 2-D: acurácia de teste = {(pred == y).mean():.4f}")
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append((cm[i, j] + cm[j, i], i, j))
    pairs.sort(reverse=True)
    print("    pares de dígitos mais confundidos em 2-D (erros nos dois sentidos):")
    for n, i, j in pairs[:6]:
        print(f"      {i} e {j}: {n}")
    rec_digit = cm.diagonal() / cm.sum(axis=1)
    print("    recall por dígito:", {k: round(float(r), 3) for k, r in enumerate(rec_digit)})
    print("    centroide de mu por dígito:",
          {k: tuple(np.round(mu[y == k].mean(0), 2)) for k in range(10)})
    print(f"    faixa de mu: x=[{mu[:, 0].min():.2f}, {mu[:, 0].max():.2f}]  "
          f"y=[{mu[:, 1].min():.2f}, {mu[:, 1].max():.2f}]")

    # --- C: Figura 5 - grade 15x15 decodificada, espaçada por Phi^-1 ----------
    n = 15
    q = np.linspace(0.05, 0.95, n)
    zs = norm.ppf(q)                      # z = Phi^{-1}(q): cada célula tem a mesma massa no prior
    print(f"\n[C] z na grade (Phi^-1 de q em [0.05, 0.95]): {np.round(zs, 3)}")
    # linha de cima = maior z2, para o eixo vertical crescer para cima (como na Figura 4)
    grid = np.array([[zx, zy] for zy in zs[::-1] for zx in zs], dtype=np.float32)
    with torch.no_grad():
        imgs = model2.decode(torch.from_numpy(grid)).numpy()
    canvas = V.images_to_grid(imgs, n, n)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, cmap="gray", extent=(0, n * 28, n * 28, 0))
    ticks = [28 * i + 14 for i in (0, 3, 7, 11, 14)]
    ax.set_xticks(ticks, [f"{zs[i]:.2f}" for i in (0, 3, 7, 11, 14)])
    ax.set_yticks(ticks, [f"{zs[::-1][i]:.2f}" for i in (0, 3, 7, 11, 14)])
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q_1)$")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q_2)$")
    ax.set_title("Figura 5 - Plano latente 2-D decodificado (grade 15×15, $q \\in [0.05, 0.95]$)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-latent-grid.png", dpi=150)
    plt.close(fig)

    # --- D: Figura 6 - interpolação pixel vs. latente (modelo d = 16) ---------
    # Escolha (regra fixa, antes de ver o resultado): primeira imagem de teste dos dígitos 3 e 7.
    ia = int((y_test == 3).nonzero()[0])
    ib = int((y_test == 7).nonzero()[0])
    print(f"\n[D] índices de teste interpolados: {ia} (dígito {int(y_test[ia])}) e "
          f"{ib} (dígito {int(y_test[ib])})")
    xa, xb = X_test[ia], X_test[ib]
    ts = torch.linspace(0, 1, 10)
    pixel_row = torch.stack([(1 - t) * xa + t * xb for t in ts])
    with torch.no_grad():
        mu_a, _ = model16.encode(xa[None])
        mu_b, _ = model16.encode(xb[None])
        latent_row = model16.decode(torch.cat([(1 - t) * mu_a + t * mu_b for t in ts]))

    fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
    for k, t in enumerate(ts):
        axes[0, k].imshow(pixel_row[k].reshape(28, 28), cmap="gray")
        axes[1, k].imshow(latent_row[k].reshape(28, 28), cmap="gray")
        axes[0, k].set_title(f"t = {t:.2f}", fontsize=9)
        for r in range(2):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
    axes[0, 0].set_ylabel("Pixels", fontsize=10)
    axes[1, 0].set_ylabel("Latente", fontsize=10)
    fig.suptitle(f"Figura 6 - Interpolação entre as imagens de teste {ia} (dígito 3) e {ib} (dígito 7): "
                 "pixels (topo) vs. latente d = 16 (embaixo)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-interpolation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
