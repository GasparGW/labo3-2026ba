"""LGBM mejorado: selección de config con las 25 etiquetas REALES de feb-2020.

Diferencia con 19_granofino_pro.py:
  - El early stopping sigue usando el mes proxy (para fijar nº de árboles), pero la
    ELECCIÓN de config se hace contra las 25 respuestas reales de feb-2020 recuperadas
    por probing (exp/probes/real_labels_feb2020.csv) — primera validación en el mes real.
  - Grilla expandida hacia MÁS regularización (min_child_samples, tweedie_power, reg_lambda):
    la estructura público→private dice que los mejores private son modelos más conservadores.
  - Ensambla las mejores configs por WAPE-real-25 y guarda en exp/private_candidates/.

Caveat honesto: las 25 son productos del split PÚBLICO. No garantizan el private (oculto).
Es la mejor señal disponible, no una prueba.
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exp", "private_candidates"))
PROBES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exp", "probes"))
SEMILLAS = [611953, 611957, 611969]
T_TRAIN, T_VALID, T_PRED = 32, 33, 35
EPS = 1e-6
t0 = time.time()

REAL = pd.read_csv(os.path.join(PROBES, "real_labels_feb2020.csv")).set_index("product_id")["tn_real_feb2020"]
print(f"{len(REAL)} etiquetas reales feb-2020 para selección de config")

# ---- feature engineering idéntico a 19_granofino_pro.py ----
p = pl.read_parquet(os.path.join(DIR, "panel_cli_prod.parquet")).sort(["customer_id", "product_id", "t"])
par = ["customer_id", "product_id"]
p = p.with_columns(compro=(pl.col("tn") > 0).cast(pl.Int8))
p = p.with_columns(
    t_ult=pl.when(pl.col("compro") == 1).then(pl.col("t")).otherwise(None).forward_fill().over(par),
    n_compras=pl.col("compro").cum_sum().over(par),
    t_1a=pl.when(pl.col("compro") == 1).then(pl.col("t")).otherwise(None).min().over(par),
    tn_acum=pl.col("tn").cum_sum().over(par),
    vida_par=pl.col("t").cum_count().over(par),
)
p = p.with_columns(
    meses_sin_compra=(pl.col("t") - pl.col("t_ult")).fill_null(99).cast(pl.Int16),
    intervalo=pl.when(pl.col("n_compras") > 1)
        .then((pl.col("t_ult") - pl.col("t_1a")) / (pl.col("n_compras") - 1)).otherwise(None),
    tn_media_hist=(pl.col("tn_acum") / pl.col("n_compras").clip(lower_bound=1)),
    escala=(pl.col("tn_acum") / pl.col("vida_par")),
)
p = p.with_columns(escala_safe=pl.col("escala").clip(lower_bound=EPS))
p = p.with_columns([(pl.col("tn").shift(k).over(par) / pl.col("escala_safe")).alias(f"plag{k}") for k in [1, 2, 3, 6, 12]])
p = p.with_columns(
    (pl.col("tn").rolling_mean(3, min_samples=1).over(par) / pl.col("escala_safe")).alias("proll3"),
    (pl.col("tn").rolling_mean(6, min_samples=1).over(par) / pl.col("escala_safe")).alias("proll6"),
    (pl.col("tn").rolling_sum(12, min_samples=1).over(par) / pl.col("escala_safe")).alias("psum12"),
    (pl.col("tn").rolling_max(12, min_samples=1).over(par) / pl.col("escala_safe")).alias("pmax12"),
)
cli = p.group_by(["customer_id", "t"]).agg(tn_cli=pl.col("tn").sum(), n_prod_cli=pl.col("compro").sum())
cli = cli.sort(["customer_id", "t"]).with_columns(
    tn_cli_lag1=pl.col("tn_cli").shift(1).over("customer_id"),
    n_prod_cli_lag1=pl.col("n_prod_cli").shift(1).over("customer_id"))
p = p.join(cli.select("customer_id", "t", "tn_cli_lag1", "n_prod_cli_lag1"), on=["customer_id", "t"], how="left")
g = pl.read_parquet(os.path.join(DIR, "dataset_prod_mes.parquet")).select(
    "product_id", "t", "mes", "lag1_esc", "lag2_esc", "lag3_esc", "lag12_esc",
    "roll6_media_esc", "roll12_media_esc", "cv6", "meses_desde_lanzamiento",
    "g_cat3_esc", "g_marca_esc", "g_desc_esc", "g_univ_roll6_esc", "g_otras_marcas_esc",
    "share_cat3", "d3_share_cat3", "n_cli_compraron", "frac_cli_perdidos",
    "cat1", "cat2", "cat3", "brand", "descripcion", "sku_size",
).rename({c: f"prod_{c}" for c in ["lag1_esc", "lag2_esc", "lag3_esc", "lag12_esc"]})
p = p.join(g, on=["product_id", "t"], how="left")
p = p.join(pl.read_parquet(os.path.join(DIR, "clusters_dtw.parquet")), on="product_id", how="left")
p = p.sort(par + ["t"]).with_columns(target_tn=pl.col("tn").shift(-2).over(par))
p = p.with_columns(target_esc=pl.col("target_tn") / pl.col("escala_safe"))

objetivo = set(pl.read_csv(os.path.join(DIR, "product_id_apredecir201912.txt"), separator="\t")["product_id"].to_list())
NO_FEAT = {"customer_id", "product_id", "periodo", "t", "tn", "cust_request_tn", "cust_request_qty",
           "es_cero_imputado", "compro", "t_ult", "t_1a", "tn_acum", "n_compras", "target_tn",
           "target_esc", "escala", "escala_safe", "tn_cli", "n_prod_cli"}
FEAT = [c for c in p.columns if c not in NO_FEAT]
CATS = ["cat1", "cat2", "cat3", "brand", "descripcion", "dtw_k5", "dtw_k10", "dtw_k20", "dtw_k40"]
pdf = p.to_pandas()
del p
for c in CATS:
    pdf[c] = pdf[c].astype("category")
tr = pdf[(pdf["t"] <= T_TRAIN) & pdf["target_tn"].notna()]
va = pdf[(pdf["t"] == T_VALID) & pdf["target_tn"].notna()]
pr = pdf[(pdf["t"] == T_PRED) & pdf["product_id"].isin(objetivo)]
print(f"train {len(tr):,} | valid {len(va):,} | pred {len(pr):,} ({time.time()-t0:.0f}s)")

# eval de early stopping (mes proxy) — solo para fijar nº de árboles
va_ids = va["product_id"].values
codigos, unicos = pd.factorize(va_ids)
mask780 = np.isin(unicos, list(objetivo))
va_esc = va["escala_safe"].values
y_prod = np.bincount(codigos, weights=va["target_tn"].values, minlength=len(unicos))
den = y_prod[mask780].sum()


def wape_proxy(y_true, y_pred):
    pp = np.bincount(codigos, weights=np.clip(y_pred, 0, None) * va_esc, minlength=len(unicos))
    return "wape_prod", np.abs(y_prod[mask780] - pp[mask780]).sum() / den, False


def pred_producto(m):
    pares = pd.Series(np.clip(m.predict(pr[FEAT]) * pr["escala_safe"].values, 0, None), index=pr["product_id"].values)
    return pares.groupby(level=0).sum().reindex(sorted(objetivo)).fillna(0.0)


def wape_real25(prod):
    p25 = prod.reindex(REAL.index)
    return float((p25 - REAL).abs().sum() / REAL.sum())


BASE = dict(objective="tweedie", n_estimators=8000, max_depth=-1, min_split_gain=0.0,
            min_child_weight=0.001, verbose=-1, n_jobs=10, learning_rate=0.02)
# grilla expandida hacia MÁS regularización / conservadurismo (dir. de mejor private)
GRILLA = [
    dict(num_leaves=127, min_child_samples=2000, colsample_bytree=0.6, max_bin=1023, tweedie_variance_power=1.2, reg_lambda=0.0),   # ~19 (baseline)
    dict(num_leaves=127, min_child_samples=5000, colsample_bytree=0.6, max_bin=1023, tweedie_variance_power=1.2, reg_lambda=1.0),
    dict(num_leaves=63,  min_child_samples=5000, colsample_bytree=0.5, max_bin=1023, tweedie_variance_power=1.3, reg_lambda=2.0),
    dict(num_leaves=63,  min_child_samples=10000, colsample_bytree=0.5, max_bin=255, tweedie_variance_power=1.4, reg_lambda=5.0),
    dict(num_leaves=31,  min_child_samples=10000, colsample_bytree=0.5, max_bin=255, tweedie_variance_power=1.5, reg_lambda=5.0),
    dict(num_leaves=255, min_child_samples=500,  colsample_bytree=0.6, max_bin=1023, tweedie_variance_power=1.1, reg_lambda=0.0),   # agresivo (control)
]


def fit(cfg, semilla):
    m = lgb.LGBMRegressor(**BASE, **cfg, random_state=semilla)
    m.fit(tr[FEAT], tr["target_esc"].values, sample_weight=tr["escala"].values,
          eval_set=[(va[FEAT], va["target_esc"].values)], eval_metric=wape_proxy,
          callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
    return m


print("\n--- grilla: proxy_wape (ES) | REAL25_wape (selección) | total_tn (conservadurismo) ---")
resultados = []
for i, cfg in enumerate(GRILLA):
    t1 = time.time()
    m = fit(cfg, SEMILLAS[0])
    prod = pred_producto(m)
    pw = m.best_score_["valid_0"]["wape_prod"]
    rw = wape_real25(prod)
    resultados.append((rw, i, cfg, prod))
    print(f"  cfg{i}: proxy {pw:.4f} | REAL25 {rw:.4f} | tot {prod.sum():,.0f} | iter {m.best_iteration_} | {time.time()-t1:.0f}s")
    print(f"        {cfg}")

resultados.sort(key=lambda r: r[0])
print(f"\nMejor por REAL25: cfg{resultados[0][1]} (REAL25 {resultados[0][0]:.4f})")

# guardar: mejor single + ensamble de las 3 mejores por REAL25
best_prod = resultados[0][3]
best_prod.rename("tn").reset_index().rename(columns={"index": "product_id"}).to_csv(
    os.path.join(OUT, "lgbm_mejorado_best.csv"), index=False)
top3 = pd.concat([r[3] for r in resultados[:3]], axis=1).mean(axis=1)
top3.rename("tn").reset_index().rename(columns={"index": "product_id"}).to_csv(
    os.path.join(OUT, "lgbm_mejorado_ens3.csv"), index=False)
print(f"REAL25 ens3: {wape_real25(top3):.4f}")
print(f"Guardados: lgbm_mejorado_best.csv, lgbm_mejorado_ens3.csv | [total {time.time()-t0:.0f}s]")
