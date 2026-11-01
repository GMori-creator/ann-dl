"""
reg_common.py - ponto de partida do projeto de regressão (passo 0).

Reaproveita, sem copiar, o código da EDA (docs/projects/eda/code): mesma limpeza, mesmo split
treino/teste (80/20, seed 42, estratificado por decis do preço) e o mesmo pipeline de
pré-processamento. A validação sai do TREINO. O teste só é tocado por `load_test_matrices`,
que é chamada apenas em step5_test.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]                 # docs/projects/regression
EDA_CODE = ROOT.parent / "eda" / "code"
sys.path.insert(0, str(EDA_CODE))

import common as C                                         # noqa: E402  (EDA)
from preprocessing import build_pipeline, target_to_model, target_to_price  # noqa: E402

SEED = 42
VAL_FRACTION = 0.15
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def split_train_val(train_df):
    """Validação = 15% do treino do EDA, estratificada por decis do preço, seed 42."""
    deciles = pd.qcut(train_df[C.TARGET], q=10, labels=False, duplicates="drop")
    tr, va = train_test_split(train_df, test_size=VAL_FRACTION, random_state=SEED, stratify=deciles)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def load_train_val():
    """Treino/validação (sem teste). O pré-processador é ajustado só em `tr`."""
    train_full, _ = C.load_split()          # o teste é descartado aqui de propósito
    tr, va = split_train_val(train_full)
    prep = build_pipeline().fit(tr[C.FEATURES])
    X_tr, X_va = prep.transform(tr[C.FEATURES]), prep.transform(va[C.FEATURES])
    y_tr, y_va = target_to_model(tr[C.TARGET]), target_to_model(va[C.TARGET])
    return dict(prep=prep, tr=tr, va=va, X_tr=X_tr.astype("float32"), X_va=X_va.astype("float32"),
                y_tr=y_tr.astype("float32"), y_va=y_va.astype("float32"))


def load_test_matrices():
    """SÓ para o passo 5. Treino completo (tr+va) NÃO é usado: o pré-processador continua o de `tr`."""
    train_full, test = C.load_split()
    tr, va = split_train_val(train_full)
    prep = build_pipeline().fit(tr[C.FEATURES])
    X_te = prep.transform(test[C.FEATURES]).astype("float32")
    return dict(prep=prep, tr=tr, va=va, test=test, X_te=X_te,
                y_te=target_to_model(test[C.TARGET]).astype("float32"))


def metrics_usd(y_true_ln, y_pred_ln):
    """MAE, RMSE (USD) e R² (USD) a partir de valores em ln(price)."""
    yt, yp = target_to_price(y_true_ln), target_to_price(y_pred_ln)
    err = yp - yt
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))),
                R2=float(1 - np.sum(err ** 2) / np.sum((yt - yt.mean()) ** 2)),
                MAE_ln=float(np.mean(np.abs(y_pred_ln - y_true_ln))),
                RMSE_ln=float(np.sqrt(np.mean((y_pred_ln - y_true_ln) ** 2))))
