"""
preprocessing.py - pipeline de pré-processamento da EDA (importável pelo projeto de modelagem).

Tudo o que "aprende" algo dos dados (quantis do winsorizing, médias e desvios do
StandardScaler, categorias do OneHotEncoder) é aprendido em `fit`, que só deve ser chamado
com o TREINO. O teste só passa por `transform`.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import common as C

# Numéricas de cauda pesada: winsorizar -> log1p -> padronizar
HEAVY_TAIL = ["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]
# Numéricas bem comportadas: só padronizar
PLAIN_NUMERIC = ["latitude", "longitude", "availability_365"]
MIN_CATEGORY_COUNT = 30        # categorias com menos de 30 anúncios no treino viram "infrequente"
CLIP_QUANTILE = 0.995


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorizing superior: limita cada coluna ao quantil `q` aprendido no treino (ignora NaN)."""

    def __init__(self, q=CLIP_QUANTILE):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.upper_ = np.nanquantile(X, self.q, axis=0)
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor():
    heavy = Pipeline([
        # 1) winsorizar ANTES de imputar: o quantil é medido só nos valores observados
        #    (os zeros estruturais imputados depois não devem puxar o P99,5 para baixo)
        ("clip", QuantileClipper(q=CLIP_QUANTILE)),
        # 2) reviews_per_month é NaN exatamente quando number_of_reviews == 0 (ausência estrutural):
        #    "nenhuma review" => taxa 0, então a imputação por constante 0 é o valor correto.
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("heavy", heavy, HEAVY_TAIL),
            ("plain", StandardScaler(), PLAIN_NUMERIC),
            ("cat", OneHotEncoder(min_frequency=MIN_CATEGORY_COUNT, handle_unknown="infrequent_if_exist",
                                  sparse_output=False), C.CATEGORICAL_FEATURES),
        ],
        remainder="drop",              # id, host_id, name, host_name, last_review e price ficam de fora
        verbose_feature_names_out=False,
    )


def build_pipeline():
    """Pipeline completo. O estimador do projeto de modelagem entra como um passo depois de 'prep'."""
    return Pipeline([("prep", build_preprocessor())])


# --------------------------------------------------------------------- alvo
def target_to_model(price):
    """Alvo para o modelo: ln(price) (assimetria 18,8 -> 0,59 no treino)."""
    return np.log(np.asarray(price, dtype=float))


def target_to_price(ln_price):
    """Volta para dólares: exp(ln_price). Métricas de negócio (RMSE/MAE em USD) usam esta escala."""
    return np.exp(np.asarray(ln_price, dtype=float))
