"""
Etapa 2 - Análise univariada (somente treino): numéricas (2A) e categóricas (2B).
Máximo de 3 figuras por item (aqui: 2 em 2A e 2 em 2B).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

C.apply_style()

HEAVY = ["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]


def numerical(train):
    print("=" * 70, "\n2A. NUMÉRICAS (treino)\n")
    num = train[C.NUMERIC_FEATURES]
    desc = pd.DataFrame({
        "n": num.count(), "média": num.mean(), "mediana": num.median(), "desvio": num.std(),
        "mín": num.min(), "Q1": num.quantile(.25), "Q3": num.quantile(.75), "máx": num.max(),
        "assimetria": num.skew(), "ausentes": num.isna().sum(), "zeros": (num == 0).sum(),
    })
    desc.index.name = "feature"
    print(C.md_table(desc, floatfmt="{:.2f}"))

    print("\nAmplitudes (para justificar a padronização na etapa 4):")
    rng = pd.DataFrame({"mín": num.min(), "máx": num.max(), "amplitude": num.max() - num.min()})
    rng.index.name = "feature"
    print(C.md_table(rng, floatfmt="{:.2f}"))

    print("\nPicos e modas:")
    mn = train.minimum_nights.value_counts()
    print(f"  minimum_nights: valor 1 = {mn.get(1, 0):,} ({mn.get(1, 0)/len(train)*100:.1f}%), "
          f"2 = {mn.get(2, 0):,} ({mn.get(2, 0)/len(train)*100:.1f}%), "
          f"30 = {mn.get(30, 0):,} ({mn.get(30, 0)/len(train)*100:.1f}%), "
          f">30 = {(train.minimum_nights > 30).sum():,} ({(train.minimum_nights > 30).mean()*100:.1f}%)")
    a = train.availability_365
    print(f"  availability_365: == 0 em {(a == 0).sum():,} ({(a == 0).mean()*100:.1f}%), "
          f"== 365 em {(a == 365).sum():,} ({(a == 365).mean()*100:.1f}%), "
          f"entre 1 e 364: {((a > 0) & (a < 365)).sum():,} ({((a > 0) & (a < 365)).mean()*100:.1f}%)")
    r = train.number_of_reviews
    print(f"  number_of_reviews: == 0 em {(r == 0).sum():,} ({(r == 0).mean()*100:.1f}%), "
          f"<= 5 em {(r <= 5).mean()*100:.1f}%")
    h = train.calculated_host_listings_count
    print(f"  calculated_host_listings_count: == 1 em {(h == 1).sum():,} ({(h == 1).mean()*100:.1f}%), "
          f">= 5 em {(h >= 5).sum():,} ({(h >= 5).mean()*100:.1f}%), >= 50 em {(h >= 50).sum():,}")
    print(f"  reviews_per_month: > 5 em {(train.reviews_per_month > 5).sum():,} linhas, máx = {train.reviews_per_month.max()}")
    lon = train.longitude
    print(f"  longitude: assimetria {lon.skew():.2f}; P1 = {lon.quantile(.01):.3f}, P99 = {lon.quantile(.99):.3f}")

    print("\nEfeito do log1p nas 4 features de cauda pesada (assimetria):")
    sk = pd.DataFrame({
        "bruta": train[HEAVY].skew(),
        "log1p": np.log1p(train[HEAVY].clip(lower=0)).skew(),
        "P99,5": train[HEAVY].quantile(.995),
        "linhas > P99,5": [(train[c] > train[c].quantile(.995)).sum() for c in HEAVY],
    })
    sk.index.name = "feature"
    print(C.md_table(sk, floatfmt="{:.2f}"))

    # ---- Figura 2: histogramas das 7 numéricas
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    axes = axes.ravel()
    for ax, col in zip(axes, C.NUMERIC_FEATURES):
        s = train[col].dropna()
        hi = s.quantile(.995)
        s_plot = s[s <= hi]
        bins = np.arange(0.5, hi + 1.5, 1) if col == "minimum_nights" else 50
        ax.hist(s_plot, bins=bins, color=C.BLUE, label=f"n = {len(s):,}")
        ax.axvline(s.median(), color=C.INK, ls="--", lw=1.3, label=f"mediana = {s.median():g}")
        ax.axvline(s.mean(), color=C.ORANGE, lw=1.3, label=f"média = {s.mean():.2f}")
        ax.set_title(f"{col}\nassimetria = {s.skew():.2f}", fontsize=10)
        ax.set_xlabel(f"valor (cortado em P99,5 = {hi:g})" if (s > hi).any() else "valor")
        ax.set_ylabel("Nº de anúncios")
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Figura 2 - Distribuição das features numéricas (treino)", y=1.01)
    fig.tight_layout()
    C.save_fig(fig, "fig02-numeric-histograms.png")

    # ---- Figura 3: antes e depois do log1p nas 4 features de cauda pesada
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for j, col in enumerate(HEAVY):
        s = train[col].dropna()
        hi = s.quantile(.995)
        ax = axes[0, j]
        bins = np.arange(0.5, hi + 1.5, 1) if col == "minimum_nights" else 50
        ax.hist(s[s <= hi], bins=bins, color=C.BLUE, label=f"bruta, assimetria = {s.skew():.1f}")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"valor (cortado em P99,5 = {hi:g})")
        ax.set_ylabel("Nº de anúncios")
        ax.legend(fontsize=8)
        ax = axes[1, j]
        ls = np.log1p(s)
        ax.hist(ls, bins=50, color=C.AQUA, label=f"log1p, assimetria = {ls.skew():.2f}")
        ax.set_xlabel("log1p(valor)")
        ax.set_ylabel("Nº de anúncios")
        ax.legend(fontsize=8)
    fig.suptitle("Figura 3 - Features de cauda pesada: valor bruto (topo) e após log1p (embaixo)", y=1.01)
    fig.tight_layout()
    C.save_fig(fig, "fig03-log-transform.png")


def categorical(train):
    print("\n" + "=" * 70, "\n2B. CATEGÓRICAS (treino)\n")
    summary = []
    for c in C.CATEGORICAL_FEATURES:
        vc = train[c].value_counts()
        summary.append({
            "feature": c, "categorias": vc.size, "mais frequente": vc.index[0],
            "% da moda": vc.iloc[0] / len(train) * 100, "menor contagem": int(vc.min()),
            "ausentes": int(train[c].isna().sum()),
        })
    print(C.md_table(pd.DataFrame(summary).set_index("feature"), floatfmt="{:.1f}"))

    for c in ["neighbourhood_group", "room_type"]:
        vc = train[c].value_counts()
        tab = pd.DataFrame({"n": vc, "%": vc / len(train) * 100})
        tab.index.name = c
        print(f"\nFrequências de {c}:")
        print(C.md_table(tab, floatfmt="{:.1f}"))

    vc = train.neighbourhood.value_counts()
    print("\nneighbourhood - dez mais frequentes:")
    top = pd.DataFrame({"n": vc.head(10), "%": vc.head(10) / len(train) * 100})
    top.index.name = "neighbourhood"
    print(C.md_table(top, floatfmt="{:.1f}"))
    cum = vc.cumsum() / len(train)
    print(f"\ncobertura acumulada: top-10 = {cum.iloc[9]*100:.1f}%, top-20 = {cum.iloc[19]*100:.1f}%, "
          f"top-50 = {cum.iloc[49]*100:.1f}%, top-100 = {cum.iloc[99]*100:.1f}%")
    for t in (1, 5, 10, 30, 100):
        small = vc[vc < t]
        print(f"  bairros com menos de {t:>3} anúncios: {len(small):>3} ({len(small)/vc.size*100:.0f}% das categorias), "
              f"cobrem {small.sum():,} linhas ({small.sum()/len(train)*100:.1f}%)")
    print(f"  bairros com exatamente 1 anúncio: {(vc == 1).sum()}")
    n_freq = (vc >= 30).sum()
    print(f"  bairros com >= 30 anúncios: {n_freq} -> colunas após agrupar raros: {n_freq} + 1 (infrequente) = {n_freq + 1}")

    # ---- Figura 4: barras de neighbourhood_group e room_type
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, c in zip(axes, ["neighbourhood_group", "room_type"]):
        v = train[c].value_counts()
        bars = ax.bar(v.index, v.values, color=C.BLUE, width=0.6, label=f"Treino (n = {len(train):,})")
        for b, n in zip(bars, v.values):
            ax.text(b.get_x() + b.get_width() / 2, n, f"{n / len(train) * 100:.1f}%", ha="center", va="bottom",
                    fontsize=9, color=C.INK2)
        ax.set_xlabel(c)
        ax.set_ylabel("Nº de anúncios")
        ax.set_title(f"{c} ({v.size} categorias)")
        ax.legend()
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Figura 4 - Frequência das categóricas de baixa cardinalidade (treino)", y=1.02)
    fig.tight_layout()
    C.save_fig(fig, "fig04-categorical-bars.png")

    # ---- Figura 5: cauda longa de neighbourhood
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    t20 = vc.head(20)[::-1]
    ax.barh(t20.index, t20.values, color=C.BLUE, height=0.65, label="20 bairros mais frequentes")
    ax.set_xlabel("Nº de anúncios")
    ax.set_ylabel("neighbourhood")
    ax.set_title("Os 20 bairros mais frequentes")
    ax.legend(loc="lower right")
    ax = axes[1]
    rank = np.arange(1, vc.size + 1)
    ax.plot(rank, cum.values * 100, color=C.BLUE, lw=2, label="Cobertura acumulada de anúncios")
    n_rare_start = int((vc >= 30).sum())
    ax.axvline(n_rare_start, color=C.ORANGE, ls="--", lw=1.4,
               label=f"Posição {n_rare_start}: daí em diante cada bairro tem < 30 anúncios")
    ax.scatter([20, 50], [cum.iloc[19] * 100, cum.iloc[49] * 100], color=C.BLUE, zorder=3, s=30)
    ax.annotate(f"top-20: {cum.iloc[19]*100:.1f}%", (20, cum.iloc[19] * 100), textcoords="offset points",
                xytext=(8, -14), fontsize=9, color=C.INK2)
    ax.annotate(f"top-50: {cum.iloc[49]*100:.1f}%", (50, cum.iloc[49] * 100), textcoords="offset points",
                xytext=(8, -14), fontsize=9, color=C.INK2)
    ax.set_xlabel("Posição do bairro (do mais ao menos frequente)")
    ax.set_ylabel("% acumulada dos anúncios")
    ax.set_title(f"Cauda longa: {vc.size} bairros")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")
    fig.suptitle("Figura 5 - Alta cardinalidade de neighbourhood (treino)", y=1.02)
    fig.tight_layout()
    C.save_fig(fig, "fig05-neighbourhood-long-tail.png")


def main():
    train, _ = C.load_split()
    print(f"treino: {train.shape}")
    numerical(train)
    categorical(train)


if __name__ == "__main__":
    main()
