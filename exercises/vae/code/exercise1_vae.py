"""
Exercise 1 - VAE no MNIST com espaço latente d = 16.
Treino, números de teste e Figuras 1, 2 e 3.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae as V  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
D = 16


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_test = V.load_mnist()
    print(f"treino={len(X_train)}  validação={len(X_val)}  teste={len(X_test)}")

    # --- C: treino ----------------------------------------------------------
    model, hist = V.train_vae(D, X_train, X_val)
    V.save_model(model, D)

    # números de teste: usados SÓ aqui, depois de o treino terminar
    te_r, te_k = V.evaluate(model, X_test)
    print(f"\n[C1] teste (d={D}): reconstrução = {te_r:.2f} nats, KL = {te_k:.2f} nats, "
          f"-ELBO = {te_r + te_k:.2f} nats")
    print(f"     validação KL: época 1 = {hist['val_kl'][0]:.2f}, época {V.EPOCHS} = {hist['val_kl'][-1]:.2f}")
    print(f"     validação recon: época 1 = {hist['val_recon'][0]:.2f}, época {V.EPOCHS} = {hist['val_recon'][-1]:.2f}")
    print(f"     treino -ELBO: época 1 = {hist['train_recon'][0] + hist['train_kl'][0]:.2f}, "
          f"época {V.EPOCHS} = {hist['train_recon'][-1] + hist['train_kl'][-1]:.2f}")
    kl_peak = int(np.argmax(hist["val_kl"])) + 1
    print(f"     KL de validação máxima na época {kl_peak}: {max(hist['val_kl']):.2f}")

    # dimensões latentes realmente usadas (KL média por dimensão, no teste)
    kd = V.kl_per_dim(model, X_test)
    print("\nKL por dimensão latente (teste, nats):", np.round(kd, 3))
    print(f"dimensões ativas (KL > 0.1 nat): {(kd > 0.1).sum()} de {D}")

    # --- Figura 1: curvas de treino -----------------------------------------
    epochs = np.arange(1, V.EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylabel, title in [
        (axes[0], "recon", "Reconstrução (BCE, nats/imagem)", "Termo de reconstrução"),
        (axes[1], "kl", "KL (nats/imagem)", "Termo de KL"),
    ]:
        ax.plot(epochs, hist[f"train_{key}"], marker="o", markersize=3, label="Treino", color="#1f77b4")
        ax.plot(epochs, hist[f"val_{key}"], marker="s", markersize=3, label="Validação", color="#ff7f0e")
        ax.set_xlabel("Época")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Figura 1 - Termos da loss por época (VAE, d = {D})")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-loss-terms.png", dpi=150)
    plt.close(fig)

    # --- Figura 2: reconstruções (decodificando mu, não um z amostrado) -------
    idx = [int((y_test == k).nonzero()[0]) for k in range(10)]
    print("\nprimeiro índice de teste de cada dígito 0-9:", idx)
    x = X_test[idx]
    with torch.no_grad():
        mu, _ = model.encode(x)
        rec = model.decode(mu)

    fig, axes = plt.subplots(2, 10, figsize=(13, 3.2))
    for k in range(10):
        axes[0, k].imshow(x[k].reshape(28, 28), cmap="gray")
        axes[1, k].imshow(rec[k].reshape(28, 28), cmap="gray")
        axes[0, k].set_title(f"dígito {k}", fontsize=9)
        for r in range(2):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recon.", fontsize=10)
    fig.suptitle("Figura 2 - Originais (topo) e reconstruções decodificando $\\mu$ (embaixo)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-reconstructions.png", dpi=150)
    plt.close(fig)

    # --- Figura 3: amostras do prior, grade 8x8 ------------------------------
    torch.manual_seed(0)
    z = torch.randn(64, D)
    with torch.no_grad():
        samples = model.decode(z)
    fig, ax = plt.subplots(figsize=(6, 6.4))
    ax.imshow(V.images_to_grid(samples.numpy(), 8, 8), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("64 amostras independentes")
    ax.set_ylabel("Amostras")
    ax.set_title("Figura 3 - Amostras de $z \\sim \\mathcal{N}(0, I)$ decodificadas")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-prior-samples.png", dpi=150)
    plt.close(fig)

    # evidência para a análise: a codificação agregada ocupa o prior?
    mu_test = V.encode_mu(model, X_test)
    print(f"\n||mu|| médio no teste = {mu_test.norm(dim=1).mean():.3f}   "
          f"(||z|| esperado sob o prior, d=16: ~{np.sqrt(D - 0.5):.3f})")
    print("média de mu por dimensão:", np.round(mu_test.mean(0).numpy(), 2))
    print("desvio de mu por dimensão:", np.round(mu_test.std(0).numpy(), 2))


if __name__ == "__main__":
    main()
