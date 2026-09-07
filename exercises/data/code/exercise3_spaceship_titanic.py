"""
Exercicio 3 - Preparing Real-World Data for a Neural Network (Spaceship Titanic)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
FIGDIR = str(Path(__file__).resolve().parent.parent / "figures")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"

df = pd.read_csv(DATA_PATH)
print("Shape bruto:", df.shape)
print(df.dtypes)

# ---------------------------------------------------------------
# A - Get to know the data
# ---------------------------------------------------------------
balance = df["Transported"].value_counts(normalize=True)
print("\nClass balance (Transported):")
print(balance)
share_true = balance[True]

numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

print("\nNumerical:", numerical_cols)
print("Categorical:", categorical_cols)
print("Other (dropped later):", ["Cabin", "Name", "PassengerId"])

cols_with_missing = int((df.isna().sum() > 0).sum())
print(f"Colunas com valores ausentes: {cols_with_missing} de {df.shape[1]} colunas totais")
missing = df.isna().sum().to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
print("\nMissing values per column:")
print(missing.sort_values("missing_count", ascending=False))

spend_stats = df[spending_cols].agg(["mean", "median", "max"]).T
print("\nSpending columns stats (mean/median/max):")
print(spend_stats)

foodcourt_mean_before = df["FoodCourt"].mean()
foodcourt_median_before = df["FoodCourt"].median()

# ---------------------------------------------------------------
# B - Split before you transform
# ---------------------------------------------------------------
X = df.drop(columns=["Transported"])
y = df["Transported"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Train class balance:", y_train.mean(), "Test class balance:", y_test.mean())

# ---------------------------------------------------------------
# C - Preprocess
# ---------------------------------------------------------------
X_train = X_train.copy()
X_test = X_test.copy()

# Feature engineering: TotalSpend (before imputing, using raw spending; NaN treated as 0 spend is a modeling choice we make explicit)
for X_ in (X_train, X_test):
    X_[spending_cols] = X_[spending_cols]  # keep as is for now (imputed below)

# Drop identifier / free-text / high-cardinality columns
drop_cols = ["Cabin", "Name", "PassengerId"]
X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# --- Imputation ---
# Numerical: median (robust to outliers/skew), fit on train only
num_imputer = SimpleImputer(strategy="median")
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Categorical: most frequent category, fit on train only
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

assert X_train[numerical_cols + categorical_cols].isna().sum().sum() == 0
assert X_test[numerical_cols + categorical_cols].isna().sum().sum() == 0

# --- Feature engineering: TotalSpend (after imputing spend columns) ---
X_train["TotalSpend"] = X_train[spending_cols].sum(axis=1)
X_test["TotalSpend"] = X_test[spending_cols].sum(axis=1)

# --- Heavy tails: log1p on spending cols + TotalSpend ---
foodcourt_before_train = X_train["FoodCourt"].copy()

log_cols = spending_cols + ["TotalSpend"]
for col in log_cols:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])

foodcourt_after_train = X_train["FoodCourt"].copy()

# Figure 6: before/after
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(foodcourt_before_train, bins=40, color="#1f77b4")
axes[0].set_title("FoodCourt - antes (bruto)")
axes[0].set_xlabel("FoodCourt")
axes[0].set_ylabel("contagem")

axes[1].hist(foodcourt_after_train, bins=40, color="#ff7f0e")
axes[1].set_title("FoodCourt - depois de log(1+x)")
axes[1].set_xlabel("log(1+FoodCourt)")
axes[1].set_ylabel("contagem")

fig.suptitle("Efeito de log(1+x) em FoodCourt (treino)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig03-log-transform.png", dpi=150)
plt.close(fig)

# --- Categorical encoding: one-hot, handle unseen categories in test ---
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[categorical_cols])

train_ohe = pd.DataFrame(
    ohe.transform(X_train[categorical_cols]),
    columns=ohe.get_feature_names_out(categorical_cols),
    index=X_train.index,
)
test_ohe = pd.DataFrame(
    ohe.transform(X_test[categorical_cols]),
    columns=ohe.get_feature_names_out(categorical_cols),
    index=X_test.index,
)

X_train_num = X_train.drop(columns=categorical_cols)
X_test_num = X_test.drop(columns=categorical_cols)

X_train_final = pd.concat([X_train_num, train_ohe], axis=1)
X_test_final = pd.concat([X_test_num, test_ohe], axis=1)

# --- Scaling: standardization ONLY on the genuinely numerical / continuous
# columns (Age + spending + TotalSpend). The one-hot columns are already
# binary (0/1), which is a compatible sub-range of tanh's [-1, 1] output and
# should NOT be re-standardized: standardizing a rare dummy (e.g. VIP=True,
# ~2% of rows) inflates it into a large z-score outlier, which would be a
# self-inflicted heavy tail right after we just fixed the real ones.
cols_to_scale = numerical_cols + ["TotalSpend"]
onehot_cols = [c for c in X_train_final.columns if c not in cols_to_scale]

scaler = StandardScaler()
X_train_final[cols_to_scale] = scaler.fit_transform(X_train_final[cols_to_scale])
X_test_final[cols_to_scale] = scaler.transform(X_test_final[cols_to_scale])

feature_cols = X_train_final.columns.tolist()
X_train_scaled = X_train_final[feature_cols].copy()
X_test_scaled = X_test_final[feature_cols].copy()

print("\nScaled train min/max:", X_train_scaled.values.min(), X_train_scaled.values.max())
print("Scaled test min/max:", X_test_scaled.values.min(), X_test_scaled.values.max())

# ---------------------------------------------------------------
# D - Verify and visualize
# ---------------------------------------------------------------
nan_train = X_train_scaled.isna().sum().sum()
nan_test = X_test_scaled.isna().sum().sum()
print(f"\nNaNs remaining - train: {nan_train}, test: {nan_test}")
print("Final train shape:", X_train_scaled.shape)
print("Final test shape:", X_test_scaled.shape)

results_ex3 = {
    "share_transported_true": share_true,
    "foodcourt_mean_before": foodcourt_mean_before,
    "foodcourt_median_before": foodcourt_median_before,
    "train_shape": X_train_scaled.shape,
    "test_shape": X_test_scaled.shape,
    "train_min": X_train_scaled.values.min(),
    "train_max": X_train_scaled.values.max(),
    "test_min": X_test_scaled.values.min(),
    "test_max": X_test_scaled.values.max(),
    "nan_train": nan_train,
    "nan_test": nan_test,
}
print("\nRESULTS_EX3 =", results_ex3)

# save artifacts for report writing
missing.to_csv(DATA_PATH.parent / "missing_table.csv")
spend_stats.to_csv(DATA_PATH.parent / "spend_stats.csv")
