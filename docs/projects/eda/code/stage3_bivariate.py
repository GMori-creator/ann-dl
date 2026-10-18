"""
Etapa 3 - Análise bivariada (somente treino): feature x alvo (3A), feature x feature (3B).
O alvo é analisado em ln(price), a escala em que a relação é mais legível (assimetria 0,59).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as C

C.apply_style()


def eta_squared(groups):
    """Fração da variância de y explicada pelo grupo (0 a 1)."""
    allv = np.concatenate(groups)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    return ss_b / ((allv - allv.mean()) ** 2).sum()


def main():
    train, _ = C.load_split()
    train = train.copy()
    train["ln_price"] = np.log(train.price)
    y = train.ln_price

    # ------------------------------------------------------------------ 3A numéricas x alvo
    print("=" * 70, "\n3A (parte 1). NUMÉRICAS x ln(price) (treino)\n")
    rows = []
    for c in C.NUMERIC_FEATURES:
        s = train[[c, "ln_price"]].dropna()
        sp = stats.spearmanr(s[c], s.ln_price)
        pe = stats.pearsonr(s[c], s.ln_price)
        rows.append({"feature": c, "Spearman": sp.statistic, "Pearson": pe.statistic, "n": len(s)})
    corr = pd.DataFrame(rows).set_index("feature")
    print(C.md_table(corr.sort_values("Spearman", key=abs, ascending=False), floatfmt="{:.3f}"))
    # Pearson no preço bruto, para mostrar o efeito da cauda
    raw_pe = {c: stats.pearsonr(train[c].fillna(0), train.price).statistic for c in C.NUMERIC_FEATURES}
    print("\nPearson com price BRUTO (NaN->0, só para comparação):")
    print({k: round(float(v), 3) for k, v in raw_pe.items()})

    print("\nPreço mediano (USD) por quartil de cada feature (monotonia):")
    for c in ["availability_365", "minimum_nights", "number_of_reviews", "calculated_host_listings_count"]:
        q = pd.qcut(train[c], 4, duplicates="drop")
        print(f"  {c}: ", train.groupby(q, observed=True).price.median().round(0).to_dict())
    nr = train.number_of_reviews == 0
    print(f"\nsem reviews (n={nr.sum():,}): preço mediano {train.price[nr].median():.0f} | com reviews: {train.price[~nr].median():.0f}")
    a0 = train.availability_365 == 0
    print(f"availability_365 == 0 (n={a0.sum():,}): preço mediano {train.price[a0].median():.0f} | >0: {train.price[~a0].median():.0f}")
    m30 = train.minimum_nights >= 30
    print(f"minimum_nights >= 30 (n={m30.sum():,}): preço mediano {train.price[m30].median():.0f} | <30: {train.price[~m30].median():.0f}")

    # Figura 6: hexbin ln(price) x 4 features
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    cols = ["availability_365", "minimum_nights", "number_of_reviews", "calculated_host_listings_count"]
    for j, c in enumerate(cols):
        x = np.log1p(train[c]) if c != "availability_365" else train[c]
        xl = f"log1p({c})" if c != "availability_365" else c
        ax = axes[0, j]
        hb = ax.hexbin(x, y, gridsize=35, cmap=C.SEQ, mincnt=1, bins="log")
        rho = corr.loc[c, "Spearman"]
        ax.set_title(f"{c}\nSpearman = {rho:.2f}", fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("ln(price)")
        ax.plot([], [], " ", label="cor = nº de anúncios (log)")
        ax.legend(loc="upper right", fontsize=8)
        if j == 3:
            fig.colorbar(hb, ax=ax, label="Nº de anúncios (escala log)")
        # mediana por faixa
        ax = axes[1, j]
        q = pd.qcut(train[c], 10, duplicates="drop")
        g = train.groupby(q, observed=True).agg(med=("price", "median"), q1=("price", lambda v: v.quantile(.25)),
                                                q3=("price", lambda v: v.quantile(.75)))
        xs = np.arange(len(g))
        ax.fill_between(xs, g.q1, g.q3, color=C.BLUE, alpha=.25, label="Q1-Q3")
        ax.plot(xs, g.med, color=C.BLUE, lw=2, marker="o", ms=4, label="Mediana")
        ax.set_xlabel(f"faixa de quantil de {c} ({len(g)} faixas; empates)")
        ax.set_ylabel("Preço (USD)")
        ax.legend()
    fig.suptitle("Figura 8 - Preço x features numéricas: densidade (topo) e mediana por faixa de quantil (embaixo)", y=1.01)
    fig.tight_layout()
    C.save_fig(fig, "fig08-numeric-vs-target.png")

    # ------------------------------------------------------------------ 3A categóricas x alvo
    print("\n" + "=" * 70, "\n3B. CATEGÓRICAS x ALVO ln(price)\n")
    for c in ["room_type", "neighbourhood_group"]:
        g = train.groupby(c).price.agg(["count", "median", "mean"])
        g["q1"] = train.groupby(c).price.quantile(.25)
        g["q3"] = train.groupby(c).price.quantile(.75)
        g = g.sort_values("median", ascending=False)
        print(f"\n{c}:")
        print(C.md_table(g.rename(columns={"count": "n", "median": "mediana", "mean": "média"}), floatfmt="{:.0f}"))
        groups = [train.ln_price[train[c] == k].values for k in g.index]
        h = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis H = {h.statistic:.1f} (p = {h.pvalue:.2g}); eta² em ln(price) = {eta_squared(groups):.3f}")
    # neighbourhood
    ng = train.groupby("neighbourhood").agg(n=("price", "size"), mediana=("price", "median"))
    big = ng[ng.n >= 30]
    groups = [train.ln_price[train.neighbourhood == k].values for k in big.index]
    h = stats.kruskal(*groups)
    print(f"\nneighbourhood (>=30 anúncios, {len(big)} bairros): Kruskal H = {h.statistic:.0f} (p = {h.pvalue:.2g}); "
          f"eta² = {eta_squared(groups):.3f}")
    print("  5 bairros mais caros (mediana, n>=30):", big.sort_values("mediana", ascending=False).head(5).mediana.to_dict())
    print("  5 mais baratos:", big.sort_values("mediana").head(5).mediana.to_dict())
    # room_type dentro de grupo: o efeito do bairro persiste controlando o tipo?
    ent = train[train.room_type == "Entire home/apt"]
    print("\nApenas 'Entire home/apt' - mediana por neighbourhood_group:",
          ent.groupby("neighbourhood_group").price.median().round(0).to_dict())
    combo = train.pivot_table(index="neighbourhood_group", columns="room_type", values="price", aggfunc="median").round(0)
    combo.index.name = "grupo"
    print("\nMediana de preço: grupo x tipo")
    print(C.md_table(combo, floatfmt="{:.0f}"))

    # Figura 7: boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, c in zip(axes, ["room_type", "neighbourhood_group"]):
        order = train.groupby(c).ln_price.median().sort_values(ascending=False).index
        data = [train.ln_price[train[c] == k] for k in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False, widths=.55,
                        medianprops=dict(color=C.INK, lw=1.6))
        for b in bp["boxes"]:
            b.set(facecolor=C.BLUE, alpha=.55, edgecolor=C.BLUE)
        ax.set_ylabel("ln(price)")
        ax.set_xlabel(c)
        eta = eta_squared([d.values for d in data])
        ax.set_title(f"{c}  (eta² = {eta:.2f})")
        ax.plot([], [], color=C.INK, lw=1.6, label="Mediana")
        ax.plot([], [], color=C.BLUE, alpha=.55, lw=8, label="Q1-Q3 (bigodes 1,5 IQR)")
        ax.legend(loc="upper right")
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Figura 9 - ln(price) por categoria (sem outliers plotados)", y=1.02)
    fig.tight_layout()
    C.save_fig(fig, "fig09-categorical-vs-target.png")

    # Figura 8: mapa de preço
    fig, ax = plt.subplots(figsize=(7.5, 7))
    s = train.sample(frac=1, random_state=C.SEED)
    sc = ax.scatter(s.longitude, s.latitude, c=s.ln_price, cmap=C.SEQ, s=3, alpha=.7,
                    vmin=y.quantile(.02), vmax=y.quantile(.98), label=f"Anúncio (n = {len(s):,})")
    cb = fig.colorbar(sc, ax=ax, shrink=.8)
    cb.set_label("ln(price)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1 / np.cos(np.radians(40.73)))
    ax.legend(loc="upper left", markerscale=4)
    ax.set_title("Figura 10 - Preço no espaço (treino)")
    fig.tight_layout()
    C.save_fig(fig, "fig10-price-map.png")

    lat_c = stats.spearmanr(train.latitude, y).statistic
    lon_c = stats.spearmanr(train.longitude, y).statistic
    print(f"\nSpearman lat x ln(price) = {lat_c:.3f}; lon x ln(price) = {lon_c:.3f} (efeito linear fraco, o mapa mostra estrutura espacial)")
    # distância ao centro (Times Square) como checagem do padrão radial
    d = np.hypot((train.latitude - 40.758) * 111, (train.longitude + 73.9855) * 84.4)
    print(f"Spearman distância a Times Square (km) x ln(price) = {stats.spearmanr(d, y).statistic:.3f}")
    print(f"  e dentro de room_type=Entire home/apt: {stats.spearmanr(d[ent.index], ent.price).statistic:.3f}")

    # ------------------------------------------------------------------ 3B feature x feature
    print("\n" + "=" * 70, "\n3A (parte 2). NUMÉRICA x NUMÉRICA e CATEGÓRICA x CATEGÓRICA\n")
    cm = train[C.NUMERIC_FEATURES + ["ln_price"]].corr(method="spearman")
    print(C.md_table(cm.rename_axis("").round(2), floatfmt="{:.2f}"))
    pairs = []
    for i, a in enumerate(C.NUMERIC_FEATURES):
        for b in C.NUMERIC_FEATURES[i + 1:]:
            pairs.append((a, b, cm.loc[a, b]))
    pairs.sort(key=lambda t: -abs(t[2]))
    print("\nPares de features mais correlacionados (|rho|):")
    for a, b, r in pairs[:6]:
        print(f"  {a} x {b}: {r:.3f}")
    # VIF nas numéricas após log1p + padronização (imputando reviews_per_month=0)
    X = train[C.NUMERIC_FEATURES].copy()
    X["reviews_per_month"] = X.reviews_per_month.fillna(0)
    for c in ["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]:
        X[c] = np.log1p(X[c])
    Z = (X - X.mean()) / X.std()
    R = np.linalg.inv(np.corrcoef(Z.values.T))
    vif = pd.Series(np.diag(R), index=X.columns)
    print("\nVIF (numéricas após log1p, imputação 0 e padronização):")
    print(vif.round(2).to_string())
    # categóricas: nested
    ct = pd.crosstab(train.neighbourhood, train.neighbourhood_group)
    print(f"\nneighbourhood aninhado em neighbourhood_group? bairros com >1 grupo: {((ct > 0).sum(axis=1) > 1).sum()} de {len(ct)}")
    ct2 = pd.crosstab(train.neighbourhood_group, train.room_type, normalize="index") * 100
    print("\n% de room_type dentro de neighbourhood_group:")
    ct2.index.name = "grupo"
    print(C.md_table(ct2, floatfmt="{:.1f}"))
    chi = stats.chi2_contingency(pd.crosstab(train.neighbourhood_group, train.room_type))
    n = len(train)
    print(f"qui-quadrado: chi2 = {chi.statistic:.0f}, p = {chi.pvalue:.2g}, V de Cramér = {np.sqrt(chi.statistic / (n * 2)):.3f}")
    nr_by = train.groupby("room_type").agg(med_nights=("minimum_nights", "median"), med_avail=("availability_365", "median"),
                                           med_rev=("number_of_reviews", "median"))
    print("\nMedianas por room_type:")
    print(C.md_table(nr_by.rename_axis("room_type"), floatfmt="{:.0f}"))

    # Figura 9: matriz de correlação
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    names = list(cm.columns)
    im = ax.imshow(cm.values, cmap=C.DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > .6 else C.INK)
    fig.colorbar(im, ax=ax, shrink=.8, label="Correlação de Spearman")
    ax.set_title("Figura 6 - Spearman entre numéricas e ln(price) (treino)")
    fig.tight_layout()
    C.save_fig(fig, "fig06-correlation-matrix.png")

    # Figura 7: pares redundantes / correlacionados
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    samp = train.sample(8000, random_state=C.SEED)
    for ax, (a, b) in zip(axes, [("number_of_reviews", "reviews_per_month"),
                                  ("calculated_host_listings_count", "availability_365")]):
        d = samp[[a, b]].dropna()
        xa = np.log1p(d[a])
        yb = np.log1p(d[b]) if b != "availability_365" else d[b]
        ax.scatter(xa, yb, s=5, alpha=.25, color=C.BLUE, label=f"Anúncio (amostra de {len(d):,})")
        q = pd.qcut(xa, 15, duplicates="drop")
        med = yb.groupby(q, observed=True).median()
        xm = xa.groupby(q, observed=True).median()
        ax.plot(xm, med, color=C.ORANGE, lw=2.2, marker="o", ms=4, label="Mediana por faixa")
        rho = cm.loc[a, b]
        ax.set_xlabel(f"log1p({a})")
        ax.set_ylabel(f"log1p({b})" if b != "availability_365" else b)
        ax.set_title(f"{a} x {b}\nSpearman = {rho:.2f}", fontsize=10)
        ax.legend(loc="upper left", frameon=True, facecolor=C.SURFACE, edgecolor=C.GRID, framealpha=.95)
    fig.suptitle("Figura 7 - Os dois pares de features mais correlacionados (treino)", y=1.02)
    fig.tight_layout()
    C.save_fig(fig, "fig07-redundant-pairs.png")

    # Figura 10: mediana de preço por grupo x tipo + tamanho de grupo
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    order = train.groupby("neighbourhood_group").price.median().sort_values(ascending=False).index
    types = ["Entire home/apt", "Private room", "Shared room"]
    w = .26
    for k, (t, col) in enumerate(zip(types, [C.BLUE, C.ORANGE, C.AQUA])):
        vals = [train.price[(train.neighbourhood_group == g) & (train.room_type == t)].median() for g in order]
        ns = [((train.neighbourhood_group == g) & (train.room_type == t)).sum() for g in order]
        bars = ax.bar(np.arange(len(order)) + (k - 1) * w, vals, w * .92, color=col, label=t)
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=C.INK2)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("Preço mediano (USD)")
    ax.set_xlabel("neighbourhood_group")
    ax.legend(title="room_type")
    ax.set_title("Figura 11 - O efeito do local se soma ao do tipo de acomodação (treino)")
    fig.tight_layout()
    C.save_fig(fig, "fig11-group-by-room.png")
    print("\nn por célula (grupo x tipo):")
    print(pd.crosstab(train.neighbourhood_group, train.room_type).to_string())

    # ------------------------------------------------------------------ 3C numérica x categórica
    print("\n" + "=" * 70, "\n3C. NUMÉRICA x CATEGÓRICA (localização e dispersão)\n")
    feats = [("availability_365", lambda v: v, "availability_365"),
             ("minimum_nights", np.log1p, "log1p(minimum_nights)"),
             ("number_of_reviews", np.log1p, "log1p(number_of_reviews)")]
    rows = []
    for cat in ["room_type", "neighbourhood_group"]:
        for f, tf, _ in feats:
            groups = [tf(train[f][train[cat] == k]).values for k in sorted(train[cat].unique())]
            h = stats.kruskal(*groups)
            med = train.groupby(cat)[f].median()
            iqr = train.groupby(cat)[f].apply(lambda v: v.quantile(.75) - v.quantile(.25))
            rows.append({"par": f"{cat} | {f}", "H de Kruskal": h.statistic, "eta²": eta_squared(groups),
                         "mediana mín-máx entre grupos": f"{med.min():.0f}-{med.max():.0f}",
                         "IQR mín-máx entre grupos": f"{iqr.min():.0f}-{iqr.max():.0f}"})
    print(C.md_table(pd.DataFrame(rows).set_index("par"), floatfmt="{:.3f}"))
    sh = train[train.room_type == "Shared room"].availability_365
    print(f"\nShared room: availability_365 mediana {sh.median():.0f}, IQR {sh.quantile(.75) - sh.quantile(.25):.0f}; "
          f"demais tipos: mediana {train.availability_365[train.room_type != 'Shared room'].median():.0f}")
    print("minimum_nights, % com >= 30 noites, por grupo:",
          (train.minimum_nights.ge(30).groupby(train.neighbourhood_group).mean() * 100).round(1).to_dict())
    print("minimum_nights, % com >= 30 noites, por tipo:",
          (train.minimum_nights.ge(30).groupby(train.room_type).mean() * 100).round(1).to_dict())

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for i, cat in enumerate(["room_type", "neighbourhood_group"]):
        order = list(train.groupby(cat).price.median().sort_values(ascending=False).index)
        for j, (f, tf, lab) in enumerate(feats):
            ax = axes[i, j]
            data = [tf(train[f][train[cat] == k]) for k in order]
            bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False, widths=.55,
                            medianprops=dict(color=C.INK, lw=1.6))
            for b in bp["boxes"]:
                b.set(facecolor=C.BLUE, alpha=.55, edgecolor=C.BLUE)
            eta = eta_squared([d.values for d in data])
            ax.set_title(f"{lab} por {cat}\neta² = {eta:.3f}", fontsize=10)
            ax.set_ylabel(lab)
            ax.set_xlabel(cat)
            ax.tick_params(axis="x", labelrotation=15)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C.INK, lw=1.6, label="Mediana"),
                        Line2D([], [], color=C.BLUE, alpha=.55, lw=8, label="Q1-Q3 (caixa); bigodes = 1,5 x IQR")],
               loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("Figura 12 - Features numéricas por categoria: mediana (localização) e Q1-Q3 (dispersão), sem outliers", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    C.save_fig(fig, "fig12-numeric-by-category.png")


if __name__ == "__main__":
    main()
