"""
Track B - Modelo 'residuo' estilo Rosario (la unica tecnica con evidencia real de PRIVADO
en este dataset: sus `_residuo` ganaron a los `_nivel` 2/2 en privado).

Idea: baseline simple por producto (media movil ponderada) + LGBM que predice el RESIDUO
(target - baseline), no el nivel. Final = baseline + residuo_predicho. El baseline captura
la persistencia robusta (transfiere); el LGBM solo corrige lo chico -> sobreajusta menos.

Valida honesto en feb-2019/2018 (target_period<=stand) y chequea los 25 reales feb-2020.
Uso: python src/trackB/33_residuo.py
"""

import polars as pl, numpy as np, warnings, csv

warnings.filterwarnings("ignore")
import lightgbm as lgb

PANEL = "datasets/panel_producto.parquet"
APRE = "datasets/product_id_apredecir201912.txt"
LAGS = [1, 2, 3, 4, 5, 6, 12]
ROLL = [3, 6, 12]


def build_long():
    prod = pl.read_parquet(PANEL)
    periods = sorted([c for c in prod.columns if c != "product_id"], key=int)
    long = (
        prod.melt(
            id_vars="product_id",
            value_vars=periods,
            variable_name="period",
            value_name="tn",
        )
        .with_columns(pl.col("period").cast(pl.Int64))
        .sort(["product_id", "period"])
    )
    long = long.with_columns((pl.col("period") % 100).alias("month"))
    for L in LAGS:
        long = long.with_columns(
            pl.col("tn").shift(L).over("product_id").alias(f"lag{L}")
        )
    for W in ROLL:
        long = long.with_columns(
            pl.col("tn").rolling_mean(W).over("product_id").alias(f"rm{W}")
        )
    # baselines (forecast para t+2 con data hasta t)
    long = long.with_columns(
        [
            pl.col("tn").alias("base_last"),
            ((pl.col("tn") + pl.col("lag1") + pl.col("lag2")) / 3).alias("base_ma3"),
            (
                (
                    6 * pl.col("tn")
                    + 5 * pl.col("lag1")
                    + 4 * pl.col("lag2")
                    + 3 * pl.col("lag3")
                    + 2 * pl.col("lag4")
                    + 1 * pl.col("lag5")
                )
                / 21
            ).alias("base_pond"),
        ]
    )
    long = long.with_columns(
        [
            pl.col("tn").shift(-2).over("product_id").alias("target"),
            pl.col("period").shift(-2).over("product_id").alias("target_period"),
            (pl.col("tn") - pl.col("tn").shift(3).over("product_id")).alias("d3"),
        ]
    )
    return long, periods


BASEFEATS = ["tn", "month", "d3"] + [f"lag{L}" for L in LAGS] + [f"rm{W}" for W in ROLL]


def predict(long, stand, apre, base_col, mode="residuo", objective="l1"):
    feats = BASEFEATS + [base_col]
    tr = long.filter(
        (pl.col("target").is_not_null())
        & (pl.col("target_period") <= stand)
        & (pl.col(base_col).is_not_null())
    )
    X = tr.select(feats).to_numpy()
    if mode == "residuo":
        y = (
            tr.select("target").to_numpy().ravel()
            - tr.select(base_col).to_numpy().ravel()
        )
    else:  # nivel
        y = tr.select("target").to_numpy().ravel()
    m = lgb.train(
        dict(
            objective=objective,
            num_leaves=63,
            learning_rate=0.05,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1,
        ),
        lgb.Dataset(X, label=y),
        num_boost_round=300,
    )
    pr = long.filter(pl.col("period") == stand)
    raw = m.predict(pr.select(feats).to_numpy())
    base = pr.select(base_col).to_numpy().ravel()
    out = (base + raw) if mode == "residuo" else raw
    d = {
        int(pid): max(0.0, float(v)) for pid, v in zip(pr["product_id"].to_list(), out)
    }
    return {p: d.get(p, 0.0) for p in apre}


def tfe(pred, real, ids):
    T = sum(real[p] for p in ids)
    return sum(abs(pred.get(p, 0.0) - real[p]) for p in ids) / T


if __name__ == "__main__":
    long, periods = build_long()
    prod = pl.read_parquet(PANEL)
    P = {r["product_id"]: r for r in prod.iter_rows(named=True)}
    apre = pl.read_csv(APRE, separator="\t")["product_id"].to_list()

    for stand, target, tag in [
        (201812, 201902, "FEB-2019"),
        (201712, 201802, "FEB-2018"),
    ]:
        real = {p: (P[p].get(str(target), 0.0) if p in P else 0.0) for p in apre}
        last = {p: (P[p].get(str(stand), 0.0) if p in P else 0.0) for p in apre}
        rows = [("last_value (baseline puro)", last)]
        for base in ["base_last", "base_ma3", "base_pond"]:
            rows.append(
                (f"NIVEL  lgbm ({base})", predict(long, stand, apre, base, "nivel"))
            )
            rows.append(
                (f"RESIDUO lgbm ({base})", predict(long, stand, apre, base, "residuo"))
            )
        print(f"\n=== {tag} (parado {stand}, +2) — TFE honesto 780 ===")
        for name, v in rows:
            print(f"    {name:30} {tfe(v, real, apre):.4f}")
