"""
Etapa 1 - Inspeção inicial: dicionário (A), qualidade (B), alvo (C) e split (D).

Ordem de EXECUÇÃO: a qualidade (B) usa regras que não aprendem nada dos dados (contagens de
valores ausentes e impossíveis), então roda no arquivo bruto; em seguida a limpeza por regra e o
split (D) acontecem ANTES de qualquer estatística do alvo, e o item C usa só o treino.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as C

C.apply_style()


def main():
    raw = C.load_raw()

    # ------------------------------------------------------------------ A
    print("=" * 70, "\nA. DICIONÁRIO - tipos, cardinalidade e exemplo\n")
    info = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "n_unicos": raw.nunique(),
        "exemplo": raw.iloc[0].astype(str).str.slice(0, 32),
    })
    info.index.name = "coluna"
    print(C.md_table(info))
    print(f"\nforma bruta: {raw.shape[0]:,} linhas x {raw.shape[1]} colunas")

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 70, "\nB. QUALIDADE (arquivo bruto)\n")
    miss = pd.DataFrame({"ausentes": raw.isna().sum(), "pct": raw.isna().mean() * 100})
    miss = miss[miss.ausentes > 0].sort_values("ausentes", ascending=False)
    miss.index.name = "coluna"
    print("Valores ausentes por coluna:")
    print(C.md_table(miss, floatfmt="{:.2f}"))
    print(f"colunas sem nenhum ausente: {(raw.isna().sum() == 0).sum()} de {raw.shape[1]}")

    print(f"\nlinhas duplicadas (todas as colunas): {raw.duplicated().sum()}")
    print(f"linhas duplicadas ignorando 'id': {raw.drop(columns=['id']).duplicated().sum()}")
    print(f"ids repetidos: {raw['id'].duplicated().sum()}")

    # ausência estrutural: reviews_per_month e last_review só faltam quando não há reviews?
    no_rev = raw.number_of_reviews == 0
    print("\nAusência estrutural de reviews:")
    print(f"  number_of_reviews == 0:                      {no_rev.sum():,}")
    print(f"  reviews_per_month ausente:                   {raw.reviews_per_month.isna().sum():,}")
    print(f"  last_review ausente:                         {raw.last_review.isna().sum():,}")
    print(f"  ausente em reviews_per_month E reviews == 0: {(raw.reviews_per_month.isna() & no_rev).sum():,}")
    print(f"  ausente em last_review E reviews == 0:       {(raw.last_review.isna() & no_rev).sum():,}")
    print(f"  reviews == 0 mas reviews_per_month preenchido: {(no_rev & raw.reviews_per_month.notna()).sum()}")

    print("\nValores impossíveis ou inconsistentes:")
    print(f"  price <= 0:                          {(raw.price <= 0).sum()}")
    print(f"  minimum_nights > 365 (mais de 1 ano): {(raw.minimum_nights > 365).sum()}  (máx = {raw.minimum_nights.max()})")
    print(f"  latitude fora de [40.4, 41.0]:        {((raw.latitude < 40.4) | (raw.latitude > 41.0)).sum()}")
    print(f"  longitude fora de [-74.3, -73.7]:     {((raw.longitude < -74.3) | (raw.longitude > -73.7)).sum()}")
    print(f"  availability_365 fora de [0, 365]:    {((raw.availability_365 < 0) | (raw.availability_365 > 365)).sum()}")
    print(f"  number_of_reviews < 0:                {(raw.number_of_reviews < 0).sum()}")
    print(f"  availability_365 == 0 (válido, mas atípico): {(raw.availability_365 == 0).sum():,} ({(raw.availability_365 == 0).mean()*100:.1f}%)")
    dates = pd.to_datetime(raw.last_review)
    print(f"  last_review: de {dates.min().date()} a {dates.max().date()}")

    print("\nColunas candidatas a descarte:")
    print(f"  constantes: {[c for c in raw.columns if raw[c].nunique() <= 1]}")
    print(f"  id: {raw.id.nunique():,} valores únicos em {len(raw):,} linhas (identificador)")
    print(f"  host_id: {raw.host_id.nunique():,} únicos (identificador)")
    print(f"  name: {raw.name.nunique():,} únicos (texto livre) | host_name: {raw.host_name.nunique():,} únicos")

    # ------------------------------------------------------------------ D (executado antes do C)
    print("\n" + "=" * 70, "\nD. LIMPEZA POR REGRA E SPLIT (seed 42)\n")
    clean = C.clean(raw)
    print(f"linhas removidas por regra (price <= 0): {len(raw) - len(clean)}  -> {len(clean):,} linhas")
    train, test = C.split(clean)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    print(f"treino = {len(train):,} ({len(train)/len(clean)*100:.1f}%) | teste = {len(test):,} ({len(test)/len(clean)*100:.1f}%)")
    qs = [0.25, 0.5, 0.75, 0.9, 0.99]
    comp = pd.DataFrame({
        "treino": [train.price.mean(), *train.price.quantile(qs).values],
        "teste": [test.price.mean(), *test.price.quantile(qs).values],
    }, index=["média", "Q1", "mediana", "Q3", "P90", "P99"])
    comp.index.name = "preço (USD)"
    print(C.md_table(comp, floatfmt="{:.1f}"))
    ks = stats.ks_2samp(train.price, test.price)
    print(f"KS treino vs teste: estatística = {ks.statistic:.4f} (p = {ks.pvalue:.3f})")
    print(f"unicos de 'neighbourhood': treino {train.neighbourhood.nunique()}, teste {test.neighbourhood.nunique()}")

    # ------------------------------------------------------------------ C (só treino)
    print("\n" + "=" * 70, "\nC. ALVO 'price' (somente treino)\n")
    p = train.price
    desc = pd.Series({
        "n": len(p), "média": p.mean(), "desvio": p.std(), "mínimo": p.min(),
        "Q1": p.quantile(.25), "mediana": p.median(), "Q3": p.quantile(.75),
        "P95": p.quantile(.95), "P99": p.quantile(.99), "P99,5": p.quantile(.995), "máximo": p.max(),
    })
    print(desc.round(2).to_string())
    print(f"\nassimetria (skew): {p.skew():.2f} | curtose: {p.kurt():.1f}")
    print(f"assimetria de ln(price): {np.log(p).skew():.2f} | de log1p(price): {np.log1p(p).skew():.2f}")
    print(f"média/mediana = {p.mean() / p.median():.2f}")
    q1, q3 = p.quantile([.25, .75])
    print(f"cerca de Tukey em escala linear: > {q3 + 1.5 * (q3 - q1):.0f} -> {(p > q3 + 1.5 * (q3 - q1)).sum():,} linhas ({(p > q3 + 1.5 * (q3 - q1)).mean()*100:.1f}%)")
    lp = np.log(p)
    lq1, lq3 = lp.quantile([.25, .75])
    lo, hi = np.exp(lq1 - 1.5 * (lq3 - lq1)), np.exp(lq3 + 1.5 * (lq3 - lq1))
    out_tr = ((p < lo) | (p > hi)).sum()
    out_te = ((test.price < lo) | (test.price > hi)).sum()
    print(f"cerca de Tukey em ln(price): [{lo:.1f}, {hi:.1f}] -> treino {out_tr} linhas ({out_tr/len(p)*100:.2f}%), "
          f"teste {out_te} linhas ({out_te/len(test)*100:.2f}%)")
    print(f"price >= 1000: {(p >= 1000).sum()} | >= 5000: {(p >= 5000).sum()}")

    # Figura 1 - distribuição do alvo
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cap = p.quantile(.995)
    ax = axes[0]
    ax.hist(p[p <= cap], bins=np.arange(0, cap + 20, 20), color=C.BLUE, label=f"Treino (n = {len(p):,})")
    ax.axvline(p.median(), color=C.INK, lw=1.4, ls="--", label=f"Mediana = {p.median():.0f}")
    ax.axvline(p.mean(), color=C.ORANGE, lw=1.4, ls="-", label=f"Média = {p.mean():.0f}")
    ax.set_xlabel("Preço por noite (USD)")
    ax.set_ylabel("Nº de anúncios")
    ax.set_title(f"Preço, escala linear (cortado em {cap:.0f}; {(p > cap).sum()} acima, máx = {p.max():,})")
    ax.legend()
    ax = axes[1]
    ax.hist(np.log10(p), bins=60, color=C.BLUE, label=f"Treino (n = {len(p):,})")
    ax.axvline(np.log10(p.median()), color=C.INK, lw=1.4, ls="--", label="Mediana")
    ax.set_xlabel("log10(preço) (USD)")
    ax.set_ylabel("Nº de anúncios")
    ax.set_title(f"Preço em escala log (assimetria {np.log(p).skew():.2f} vs. {p.skew():.1f})")
    ax.legend()
    fig.suptitle("Figura 1 - Distribuição do alvo (price)", y=1.03)
    fig.tight_layout()
    C.save_fig(fig, "fig01-target-distribution.png")

    # leakage: nenhum atributo é (quase) função do preço?
    print("\nInvestigação de vazamento (correlação de Spearman com price, treino):")
    sp = train[C.NUMERIC_FEATURES + ["price"]].corr(method="spearman")["price"].drop("price")
    print(sp.round(3).sort_values(key=abs, ascending=False).to_string())
    print(f"|rho| máximo = {sp.abs().max():.3f}")


if __name__ == "__main__":
    main()
