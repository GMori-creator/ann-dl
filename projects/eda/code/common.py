"""
common.py - constantes, carregamento, limpeza por regra, split e estilo dos gráficos.

Todos os scripts da EDA importam este módulo, então o split e as listas de colunas são
exatamente os mesmos em todas as etapas. Caminhos são relativos a este arquivo, de modo
que tudo roda de um clone limpo do repositório.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

SEED = 42

ROOT = Path(__file__).resolve().parents[1]           # docs/projects/eda
DATA_FILE = ROOT / "data" / "AB_NYC_2019.csv"
FIGURES = ROOT / "figures"

TARGET = "price"

# Papéis das colunas (definidos na inspeção, etapa 1)
ID_COLS = ["id", "host_id"]                          # identificadores
TEXT_COLS = ["name", "host_name"]                    # texto livre
DATE_COLS = ["last_review"]                          # data (redundante com as contagens de reviews)
DROP_COLS = ID_COLS + TEXT_COLS + DATE_COLS

NUMERIC_FEATURES = [
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365",
]
CATEGORICAL_FEATURES = ["neighbourhood_group", "neighbourhood", "room_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------- dados
def load_raw():
    """CSV original, sem nenhuma alteração (48.895 linhas x 16 colunas)."""
    return pd.read_csv(DATA_FILE)


def clean(df):
    """
    Limpeza POR REGRA (não aprende nada dos dados, então pode vir antes do split):
    remove anúncios com preço <= 0, que são valores impossíveis para uma diária.
    """
    return df[df[TARGET] > 0].reset_index(drop=True)


def split(df):
    """
    Split 80/20 com semente fixa (random_state=42), estratificado por decis do preço:
    o alvo tem cauda muito pesada e a estratificação garante que treino e teste tenham
    a mesma distribuição de preços (inclusive na cauda). Não há data confiável, então
    um split temporal não se aplica.
    """
    deciles = pd.qcut(df[TARGET], q=10, labels=False, duplicates="drop")
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=deciles)


def load_split():
    """(train, test) com todas as colunas. A partir daqui, toda estatística vem do treino."""
    train, test = split(clean(load_raw()))
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ------------------------------------------------------------------- gráficos
# Paleta de dados validada (azul / laranja / aqua são os 3 primeiros slots categóricos;
# azul->vermelho com meio cinza para divergente; rampa de um só matiz para sequencial)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY = "#e34948", "#f0efec"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", [BLUE, GRAY, RED])


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "regular",
        "axes.labelsize": 10, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "figure.dpi": 100, "savefig.dpi": 150,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA]),
    })


def md_table(df, floatfmt="{:.2f}", index=True):
    """Imprime um DataFrame como tabela Markdown (sem dependências extras)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx in df.index:
        # df.at preserva o tipo de cada célula (inteiros não viram float, como em iterrows)
        cells = ([str(idx)] if index else []) + [fmt(df.at[idx, c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_fig(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura salva: {name}")
