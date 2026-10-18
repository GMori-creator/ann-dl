"""
Etapa 4 - Pré-processamento: verificação do pipeline (4A/4B), outliers (4C) e
redução de dimensionalidade PCA / t-SNE / UMAP (4D).
Tudo é ajustado SOMENTE no treino; o teste só passa por transform.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

import common as C
import preprocessing as P

C.apply_style()


def checks(train, test):
    print("=" * 70, "\n4A/4B. PIPELINE AJUSTADO NO TREINO\n")
    pipe = P.build_pipeline().fit(train[C.FEATURES])
    Xtr = pipe.transform(train[C.FEATURES])
    Xte = pipe.transform(test[C.FEATURES])
    names = pipe.get_feature_names_out()
    print(f"treino: {train[C.FEATURES].shape} -> {Xtr.shape} | teste: {test[C.FEATURES].shape} -> {Xte.shape}")
    print(f"NaN após o pipeline: treino {np.isnan(Xtr).sum()}, teste {np.isnan(Xte).sum()}")
    print(f"colunas de saída: {len(names)} = 4 cauda pesada + 3 numéricas + "
          f"{sum(n.startswith('neighbourhood_group_') for n in names)} grupos + "
          f"{sum(n.startswith('neighbourhood_') and not n.startswith('neighbourhood_group_') for n in names)} bairros "
          f"(inclui 'infrequente') + {sum(n.startswith('room_type_') for n in names)} tipos")

    prep = pipe.named_steps["prep"]
    heavy = prep.named_transformers_["heavy"]
    clip = heavy.named_steps["clip"]
    print("\nLimites do winsorizing aprendidos NO TREINO (P99,5):")
    print({c: float(round(u, 2)) for c, u in zip(P.HEAVY_TAIL, clip.upper_)})

    # padronização: treino ~ N(0,1) por construção; teste só aproximadamente (prova de que não houve vazamento)
    num = list(names[:7])
    summ = pd.DataFrame({"treino média": Xtr[:, :7].mean(0), "treino desvio": Xtr[:, :7].std(0),
                         "teste média": Xte[:, :7].mean(0), "teste desvio": Xte[:, :7].std(0)}, index=num)
    summ.index.name = "feature"
    print("\nMédia e desvio das 7 numéricas transformadas:")
    print(C.md_table(summ, floatfmt="{:.3f}"))

    # categorias
    ohe = prep.named_transformers_["cat"]
    nb_idx = C.CATEGORICAL_FEATURES.index("neighbourhood")
    cats = ohe.categories_[nb_idx]
    unseen = set(test.neighbourhood) - set(train.neighbourhood)
    print(f"\nbairros no treino: {len(cats)} | só no teste (nunca vistos): {len(unseen)} "
          f"({test.neighbourhood.isin(unseen).sum()} linhas de teste)")
    inf = ohe.infrequent_categories_[nb_idx]
    print(f"bairros agrupados em 'infrequente' (< {P.MIN_CATEGORY_COUNT} no treino): {len(inf)}")
    inf_col = list(names).index("neighbourhood_infrequent_sklearn")
    print(f"linhas na coluna 'infrequente': treino {int(Xtr[:, inf_col].sum()):,} ({Xtr[:, inf_col].mean()*100:.1f}%), "
          f"teste {int(Xte[:, inf_col].sum()):,} ({Xte[:, inf_col].mean()*100:.1f}%)")
    # sanidade: uma linha com bairro inédito não quebra
    probe = test[C.FEATURES].iloc[:1].copy()
    probe["neighbourhood"] = "Bairro Inexistente"
    pr = pipe.transform(probe)
    print(f"linha com bairro inédito -> transform OK, 'infrequente' = {pr[0, inf_col]:.0f}, NaN = {np.isnan(pr).sum()}")

    # estruturalidade da imputação
    nan_rpm = train.reviews_per_month.isna()
    print(f"\nreviews_per_month ausente: {nan_rpm.sum():,}; destes com number_of_reviews == 0: "
          f"{(train.number_of_reviews[nan_rpm] == 0).sum():,} -> imputação por 0 é exata")

    # alvo
    y = P.target_to_model(train.price)
    print(f"\nalvo ln(price): assimetria {pd.Series(y).skew():.2f} (bruto {train.price.skew():.2f}); "
          f"volta a USD com exp(): máx erro de ida e volta = {np.abs(P.target_to_price(y) - train.price).max():.1e}")
    return pipe, Xtr, Xte, names


def outliers(train, test):
    print("\n" + "=" * 70, "\n4C. OUTLIERS\n")
    # features: linhas atingidas pelo winsorizing
    flagged = pd.DataFrame(index=train.index)
    rows = []
    for c in P.HEAVY_TAIL:
        hi = train[c].quantile(P.CLIP_QUANTILE)
        f = train[c] > hi
        flagged[c] = f
        rows.append({"feature": c, "limite P99,5": hi, "linhas winsorizadas": int(f.sum()),
                     "teste acima do limite": int((test[c] > hi).sum())})
    tab = pd.DataFrame(rows).set_index("feature")
    print(C.md_table(tab, floatfmt="{:.2f}"))
    union = flagged.any(axis=1)
    print(f"linhas com pelo menos uma feature winsorizada (união): {union.sum()} ({union.mean()*100:.2f}%)")
    print(f"  ... e linhas com >= 2 features winsorizadas: {(flagged.sum(axis=1) >= 2).sum()}")

    # alvo: Tukey em ln(price), limites do treino
    lp = np.log(train.price)
    q1, q3 = lp.quantile([.25, .75])
    lo, hi = np.exp(q1 - 1.5 * (q3 - q1)), np.exp(q3 + 1.5 * (q3 - q1))
    t_out = (train.price < lo) | (train.price > hi)
    print(f"\ncerca de Tukey em ln(price) (treino): [{lo:.1f}, {hi:.1f}] USD -> {t_out.sum()} linhas "
          f"({t_out.mean()*100:.2f}%): {(train.price < lo).sum()} abaixo, {(train.price > hi).sum()} acima")
    print(f"  teste fora dos limites do treino: {((test.price < lo) | (test.price > hi)).sum()} linhas")
    print(f"  preço máximo = {train.price.max():,}; >= 1000 USD: {(train.price >= 1000).sum()}; >= 5000: {(train.price >= 5000).sum()}")
    both = t_out & union
    print(f"  interseção (preço outlier E feature winsorizada): {both.sum()}")
    print(f"  composição dos outliers de preço por room_type:")
    print(train.room_type[t_out].value_counts().to_dict())
    print(f"  mediana de minimum_nights: outliers de preço {train.minimum_nights[t_out].median():.0f} vs demais {train.minimum_nights[~t_out].median():.0f}")


def dimred(train, Xtr, names):
    print("\n" + "=" * 70, "\n4D. REDUÇÃO DE DIMENSIONALIDADE (treino)\n")
    pca = PCA(random_state=C.SEED).fit(Xtr)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"dimensão de entrada: {Xtr.shape[1]}")
    for k in (2, 5, 10, 20, 30, 50):
        print(f"  {k:>2} componentes: {cum[k-1]*100:.1f}% da variância")
    for t in (.80, .90, .95):
        print(f"  componentes para {t*100:.0f}%: {int(np.searchsorted(cum, t) + 1)}")
    print(f"  variância de PC1..PC5: {[round(float(v)*100, 1) for v in evr[:5]]}")

    # cargas: quais colunas dominam PC1-PC3
    load = pd.DataFrame(pca.components_[:3].T, index=names, columns=["PC1", "PC2", "PC3"])
    for pc in load.columns:
        top = load[pc].reindex(load[pc].abs().sort_values(ascending=False).index).head(6)
        print(f"\n  maiores cargas em {pc}:")
        print("   ", {k: round(float(v), 2) for k, v in top.items()})

    # amostra para os métodos não lineares
    rng = np.random.RandomState(C.SEED)
    idx = rng.choice(len(Xtr), 5000, replace=False)
    Xs = Xtr[idx]
    lp = np.log(train.price.values[idx])
    room = train.room_type.to_numpy(dtype=object)[idx]
    Z_pca = pca.transform(Xs)[:, :2]

    results = {}
    for perp in (15, 50):
        t0 = time.time()
        Z = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=C.SEED).fit_transform(Xs)
        results[("t-SNE", f"perplexity = {perp}")] = Z
        print(f"\n  t-SNE perplexity {perp}: {time.time()-t0:.0f}s")
    for nn in (15, 50):
        t0 = time.time()
        Z = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=C.SEED).fit_transform(Xs)
        results[("UMAP", f"n_neighbors = {nn}")] = Z
        print(f"  UMAP n_neighbors {nn}: {time.time()-t0:.0f}s")

    # métrica objetiva: trustworthiness (k=10) e associação com o alvo (Spearman do alvo vs. média dos 10 vizinhos 2D)
    from sklearn.neighbors import NearestNeighbors
    from scipy import stats

    def nn_target_corr(Z):
        nbr = NearestNeighbors(n_neighbors=11).fit(Z)
        ind = nbr.kneighbors(Z, return_distance=False)[:, 1:]
        return stats.spearmanr(lp, lp[ind].mean(1)).statistic

    grp = train.neighbourhood_group.to_numpy(dtype=object)[idx]

    def nn_agree(Z, labels):
        """Fração dos 10 vizinhos mais próximos (no 2D) que têm o mesmo rótulo do ponto."""
        ind = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        return float((labels[ind] == labels[:, None]).mean())

    def row(m, p, Z):
        return {"método": m, "parâmetro": p, "trustworthiness": trustworthiness(Xs, Z, n_neighbors=10),
                "ρ alvo~vizinhos": nn_target_corr(Z), "mesmo room_type": nn_agree(Z, room),
                "mesmo grupo": nn_agree(Z, grp)}

    rows = [row("PCA", "2 componentes", Z_pca)]
    for (m, p), Z in results.items():
        rows.append(row(m, p, Z))
    base_room = float(pd.Series(room).value_counts(normalize=True).pow(2).sum())
    base_grp = float(pd.Series(grp).value_counts(normalize=True).pow(2).sum())
    print(f"\n  acordo esperado ao acaso: room_type {base_room:.3f}, neighbourhood_group {base_grp:.3f}")
    cmp_ = pd.DataFrame(rows)
    cmp_.insert(0, "configuração", [f"{r['método']} ({r['parâmetro']})" for r in rows])
    cmp_ = cmp_.drop(columns=["método", "parâmetro"]).set_index("configuração")
    # referência: mesma métrica no espaço original (124 dim)
    nbr = NearestNeighbors(n_neighbors=11).fit(Xs)
    ind = nbr.kneighbors(Xs, return_distance=False)[:, 1:]
    print(f"\n  referência: ρ(alvo, média do alvo dos 10 vizinhos) no espaço original de 124 dim = {stats.spearmanr(lp, lp[ind].mean(1)).statistic:.3f}")
    print("\n  comparação (5.000 pontos do treino, k = 10):")
    print(C.md_table(cmp_, floatfmt="{:.3f}"))

    # Figura 11: PCA
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    n = 30
    ax.bar(np.arange(1, n + 1), evr[:n] * 100, color=C.BLUE, label="Variância de cada componente")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, n + 1), cum[:n] * 100, color=C.ORANGE, lw=2, label="Variância acumulada")
    ax2.set_ylabel("% acumulada")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 100)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("% da variância")
    ax.set_title("Variância explicada (30 primeiros)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax = axes[1]
    sc = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=lp, cmap=C.SEQ, s=5, alpha=.8,
                    vmin=np.quantile(lp, .02), vmax=np.quantile(lp, .98))
    fig.colorbar(sc, ax=ax, label="ln(price)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PC1 x PC2 colorido pelo alvo")
    ax = axes[2]
    for t, col in zip(["Entire home/apt", "Private room", "Shared room"], [C.BLUE, C.ORANGE, C.AQUA]):
        m = room == t
        ax.scatter(Z_pca[m, 0], Z_pca[m, 1], s=5, alpha=.6, color=col, label=t)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PC1 x PC2 por room_type")
    ax.legend(markerscale=3)
    fig.suptitle("Figura 13 - PCA sobre as 124 colunas pré-processadas (treino, 5.000 pontos nos gráficos)", y=1.03)
    fig.tight_layout()
    C.save_fig(fig, "fig13-pca.png")

    for fig_n, method in ((14, "t-SNE"), (15, "UMAP")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for j, ((m, p), Z) in enumerate(k for k in results.items() if k[0][0] == method):
            ax = axes[0, j]
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=lp, cmap=C.SEQ, s=5, alpha=.8,
                            vmin=np.quantile(lp, .02), vmax=np.quantile(lp, .98))
            fig.colorbar(sc, ax=ax, label="ln(price)")
            ax.set_title(f"{method}, {p}")
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            ax = axes[1, j]
            for t, col in zip(["Entire home/apt", "Private room", "Shared room"], [C.BLUE, C.ORANGE, C.AQUA]):
                mk = room == t
                ax.scatter(Z[mk, 0], Z[mk, 1], s=5, alpha=.6, color=col, label=t)
            ax.set_title(f"{method}, {p}, por room_type")
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            ax.legend(markerscale=3)
        fig.suptitle(f"Figura {fig_n} - {method} (treino, 5.000 pontos, semente 42)", y=1.01)
        fig.tight_layout()
        C.save_fig(fig, f"fig{fig_n}-{method.lower().replace('-', '')}.png")


def main():
    train, test = C.load_split()
    pipe, Xtr, Xte, names = checks(train, test)
    outliers(train, test)
    dimred(train, Xtr, names)


if __name__ == "__main__":
    main()
