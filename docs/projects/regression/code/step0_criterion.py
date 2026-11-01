"""
Passo 0 - métrica principal e critério de sucesso, FIXADOS ANTES de treinar qualquer rede.

Os números de referência vêm dos baselines na validação (results/baselines_val.json, passo 1).
Nenhuma rede havia sido treinada quando este arquivo foi escrito.
"""
import json

import reg_common as R

MAIN_METRIC = "MAE em dólares (USD), no preço original"
REL_IMPROVEMENT = 0.05            # a rede precisa ficar >= 5% abaixo do melhor baseline LINEAR

base = json.load(open(R.RESULTS / "baselines_val.json"))
ridge = min((v["MAE"], k) for k, v in base.items() if k.startswith("Ridge"))
THRESHOLD = ridge[0] * (1 - REL_IMPROVEMENT)
print(f"melhor linear: {ridge[1]} MAE_val={ridge[0]:.2f}  ->  critério: MAE <= {THRESHOLD:.2f} USD")
json.dump(dict(metric=MAIN_METRIC, ref=ridge[1], ref_mae=ridge[0], rel=REL_IMPROVEMENT, threshold=THRESHOLD),
          open(R.RESULTS / "criterion.json", "w"), indent=1)
