"""
vae.py - VAE escrito do zero em PyTorch para o MNIST (Exercícios 1 e 2).

Escrito à mão (sem classes de VAE de bibliotecas e sem torch.distributions.kl_divergence):
  - a reparametrização z = mu + sigma * eps,
  - os dois termos da loss (BCE somada sobre os 784 pixels e KL em forma fechada),
  - a amostragem a partir do prior.
O autograd do PyTorch é usado só para os gradientes.

Unidades: todas as losses reportadas são em nats POR IMAGEM
(somadas sobre os 784 pixels / d dimensões latentes, médias sobre as imagens).
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets

# O MNIST fica FORA do repositório (cache no diretório do usuário) para não ser commitado
# por engano. Dá para trocar o local com a variável de ambiente MNIST_DIR.
DATA_DIR = Path(os.environ.get("MNIST_DIR", Path.home() / ".cache" / "ann-dl-mnist"))
CKPT_DIR = DATA_DIR / "checkpoints"

BETA = 1.0
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20


# ------------------------------------------------------------------- dados
def load_mnist():
    """
    Retorna (X_train, X_val, X_test, y_test) com X em [0,1], achatado em 784.

    Mesmo resultado que ToTensor() + flatten (ToTensor divide os bytes por 255).
    O split 50.000 / 10.000 usa random_split com Generator().manual_seed(42);
    o random_split só depende do tamanho do dataset, então os índices são os mesmos.
    """
    full = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)
    X_full = full.data.float().div(255.0).reshape(-1, 784)
    X_test = test.data.float().div(255.0).reshape(-1, 784)

    train_idx, val_idx = random_split(
        range(len(X_full)), [50_000, 10_000], generator=torch.Generator().manual_seed(42)
    )
    return X_full[list(train_idx)], X_full[list(val_idx)], X_test, test.targets


# ------------------------------------------------------------------ modelo
class VAE(nn.Module):
    """Encoder 784-512-256, cabeças mu / log sigma^2 (256 -> d), decoder d-256-512-784."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.enc = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, d)
        self.fc_logvar = nn.Linear(256, d)
        self.dec = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),   # probabilidade de cada pixel (Bernoulli)
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar, generator=None):
        """z = mu + sigma * eps, eps ~ N(0, I). O ruído eps é a única parte aleatória,
        então o gradiente passa por mu e sigma (truque da reparametrização)."""
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.shape, generator=generator)
        return mu + sigma * eps

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar


# -------------------------------------------------------------------- loss
def loss_terms(x_hat, x, mu, logvar):
    """
    Os dois termos da -ELBO, SOMADOS sobre o batch (dividir por N fora daqui).

    recon = -sum_j [ x_j log x_hat_j + (1 - x_j) log(1 - x_hat_j) ]     (BCE, reduction="sum")
    kl    = 1/2 sum_i ( mu_i^2 + sigma_i^2 - 1 - log sigma_i^2 )         (forma fechada
            de KL( N(mu, diag sigma^2) || N(0, I) ), com sigma^2 = exp(logvar))
    """
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return recon, kl


@torch.no_grad()
def evaluate(model, X, seed=123, chunk=2000):
    """Reconstrução e KL médios por imagem (nats). A reconstrução usa UMA amostra de z por
    imagem (estimador de Monte Carlo da esperança), com gerador fixo para ser reprodutível."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    recon_sum, kl_sum = 0.0, 0.0
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        x_hat, mu, logvar = model(xb, generator=g)
        r, k = loss_terms(x_hat, xb, mu, logvar)
        recon_sum += r.item()
        kl_sum += k.item()
    model.train()
    return recon_sum / len(X), kl_sum / len(X)


@torch.no_grad()
def kl_per_dim(model, X, chunk=2000):
    """KL média por imagem de cada dimensão latente (nats): mostra quais dimensões o modelo usa."""
    model.eval()
    total = torch.zeros(model.d)
    for i in range(0, len(X), chunk):
        mu, logvar = model.encode(X[i:i + chunk])
        total += 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=0)
    model.train()
    return (total / len(X)).numpy()


@torch.no_grad()
def encode_mu(model, X, chunk=2000):
    """Média posterior mu de cada imagem (a codificação determinística)."""
    model.eval()
    out = torch.cat([model.encode(X[i:i + chunk])[0] for i in range(0, len(X), chunk)])
    model.train()
    return out


# ------------------------------------------------------------------ treino
def train_vae(d, X_train, X_val, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, beta=BETA,
              verbose=True):
    """
    Treina um VAE com Adam. Semente fixada ANTES de construir o modelo (torch.manual_seed(42)).
    No fim de cada época, mede reconstrução e KL (separados) no treino e na validação.
    Nenhuma decisão de treino usa o conjunto de teste (não há early stopping nem seleção).
    """
    torch.manual_seed(42)
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {k: [] for k in ("train_recon", "train_kl", "val_recon", "val_kl")}

    n = len(X_train)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)  # embaralha a cada época (RNG global, já semeado)
        for i in range(0, n, batch_size):
            xb = X_train[perm[i:i + batch_size]]
            x_hat, mu, logvar = model(xb)
            recon, kl = loss_terms(x_hat, xb, mu, logvar)
            loss = (recon + beta * kl) / xb.shape[0]   # média sobre o batch, em nats/imagem
            opt.zero_grad()
            loss.backward()
            opt.step()

        tr_r, tr_k = evaluate(model, X_train)
        va_r, va_k = evaluate(model, X_val)
        for k, v in zip(hist, (tr_r, tr_k, va_r, va_k)):
            hist[k].append(v)
        if verbose:
            print(f"  [d={d}] época {epoch:2d}/{epochs}  train: recon={tr_r:7.2f} kl={tr_k:6.2f}"
                  f" | val: recon={va_r:7.2f} kl={va_k:6.2f}")
    return model, hist


def save_model(model, d):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / f"vae_d{d}.pt")


def get_model(d, X_train, X_val):
    """Carrega o checkpoint de d, se existir; senão treina (determinístico, mesma semente)."""
    path = CKPT_DIR / f"vae_d{d}.pt"
    if path.exists():
        model = VAE(d)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    model, _ = train_vae(d, X_train, X_val)
    save_model(model, d)
    return model


# --------------------------------------------------------------- utilidades
def images_to_grid(imgs, nrows, ncols):
    """(nrows*ncols, 784) -> imagem única (nrows*28, ncols*28), preenchida por linhas."""
    a = np.asarray(imgs).reshape(nrows, ncols, 28, 28)
    return a.transpose(0, 2, 1, 3).reshape(nrows * 28, ncols * 28)
