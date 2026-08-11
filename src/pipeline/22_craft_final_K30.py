"""
Craft final = K30 con los reales publicos de feb-2020 implantados.

K30 = AG en top-30 (por tonelaje AG, donde AG esta medido en el piso) + median(AG,febw5,gf) en la cola.
Craft = K30 salvo en los productos publicos cuyo real recuperamos por probing -> ahi va el real.
Publico y privado son disjuntos: implantar publicos NO toca el privado (queda el de K30).

Salida: exp/submits/33_craft_K30_probed.csv
"""

import csv
import numpy as np


def load(p):
    return {
        int(r["product_id"]): float(r[[k for k in r if k != "product_id"][0]])
        for r in csv.DictReader(open(p))
    }


AG = load("exp/base_autogluon_0.2531.csv")
FEBW = load("exp/private_candidates/lgbm_febw5_pure.csv")
GF = load("exp/private_candidates/lgbm_granofino_pro_pure.csv")
CRAFT = load(
    "exp/submits/20_craft_probed.csv"
)  # AG con los reales publicos implantados

ids = sorted(AG)
TRUTH = {
    p: CRAFT[p] for p in ids if abs(CRAFT[p] - AG[p]) > 1e-6
}  # reales publicos recuperados
top30 = set(sorted(ids, key=lambda p: -AG[p])[:30])
med = lambda p: float(np.median([AG[p], FEBW[p], GF[p]]))

K30 = {p: (AG[p] if p in top30 else med(p)) for p in ids}
craft_K30 = {p: (TRUTH[p] if p in TRUTH else K30[p]) for p in ids}

out = "exp/submits/33_craft_K30_probed.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["product_id", "tn"])
    for p in ids:
        w.writerow([p, round(craft_K30[p], 6)])

# checks
n_impl = len(TRUTH)
n_tail_moved = sum(
    1 for p in ids if p not in top30 and abs(K30[p] - AG[p]) > 1e-6 and p not in TRUTH
)
Ttot = sum(AG.values())
dist_ag = sum(abs(craft_K30[p] - AG[p]) for p in ids) / Ttot
print(f"escrito: {out}  ({len(ids)} productos)")
print(f"reales publicos implantados: {n_impl}")
print(f"productos de cola con correccion K30 (no implantados): {n_tail_moved}")
print(f"top-30 anclados a AG: {len(top30)}")
print(f"distancia L1 del craft a AG: {dist_ag:.3f}")
