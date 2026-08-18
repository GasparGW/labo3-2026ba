#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 LGBM baseline a nivel producto — labo-iii-2026
============================================================================

Un modelo honesto y estándar para la competencia: predice las toneladas de
202002 parado en 201912 (horizonte +2), a nivel PRODUCTO (sumando clientes,
que es como mide Kaggle). Rinde ~0.25 en público — mitad de tabla, como la
mayoría. Sirve como punto de partida sólido.

Qué hace, en orden:
  1. Baja el sell-in del bucket público del curso.
  2. Arma un panel producto-mes (colapsa la dimensión cliente).
  3. Genera features causales: lags y promedios móviles (cada fila t usa sólo
     información hasta t).
  4. Entrena un LightGBM para predecir tn a t+2.
  5. Predice 202002 y escribe el submission en formato Kaggle.

Todo el entrenamiento es honesto: sólo usa filas cuyo target (t+2) ya está
observado al momento del corte (nada de mirar el futuro).

Requisitos:  pip install polars lightgbm
Uso:         python submission_lgbm.py   ->  salida/submission_lgbm.csv
============================================================================
"""

import os
import tempfile
import urllib.request
import warnings

warnings.filterwarnings("ignore")
import lightgbm as lgb
import polars as pl

BUCKET = "https://storage.googleapis.com/open-courses/austral2026-5da5/labo3"
STAND = 201912  # último mes observado (corte)
LAGS = [1, 2, 3, 4, 6, 12]
ROLL = [3, 6, 12]
FEATS = ["tn", "month"] + [f"lag{L}" for L in LAGS] + [f"rm{W}" for W in ROLL]


def bajar(nombre):
    dest = os.path.join(tempfile.gettempdir(), nombre)
    if not os.path.exists(dest):
        print(f"  bajando {nombre} ...")
        urllib.request.urlretrieve(f"{BUCKET}/{nombre}", dest)
    return dest


def main():
    # 1) datos
    sell = pl.read_csv(bajar("sell-in.txt.gz"), separator="\t")
    apre = pl.read_csv(bajar("product_id_apredecir201912.txt"), separator="\t")[
        "product_id"
    ].to_list()

    # 2) panel producto-mes densificado (fill 0 donde no hubo ventas)
    prod = (
        sell.group_by(["product_id", "periodo"])
        .agg(pl.col("tn").sum())
        .pivot(values="tn", index="product_id", on="periodo")
        .fill_null(0.0)
    )
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

    # 3) features causales
    long = long.with_columns((pl.col("period") % 100).alias("month"))
    for L in LAGS:
        long = long.with_columns(
            pl.col("tn").shift(L).over("product_id").alias(f"lag{L}")
        )
    for W in ROLL:
        long = long.with_columns(
            pl.col("tn").rolling_mean(W).over("product_id").alias(f"rm{W}")
        )
    long = long.with_columns(
        [
            pl.col("tn").shift(-2).over("product_id").alias("target"),
            pl.col("period").shift(-2).over("product_id").alias("target_period"),
        ]
    )

    # 4) entrenar (HONESTO: sólo filas cuyo target t+2 ya está observado)
    tr = long.filter(
        (pl.col("target").is_not_null()) & (pl.col("target_period") <= STAND)
    )
    modelo = lgb.train(
        dict(
            objective="l1",
            num_leaves=63,
            learning_rate=0.05,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1,
        ),
        lgb.Dataset(
            tr.select(FEATS).to_numpy(), label=tr.select("target").to_numpy().ravel()
        ),
        num_boost_round=300,
    )

    # 5) predecir 202002 (fila del corte) y escribir submission
    pr = long.filter(pl.col("period") == STAND)
    pred = {
        int(pid): max(0.0, float(v))
        for pid, v in zip(
            pr["product_id"].to_list(), modelo.predict(pr.select(FEATS).to_numpy())
        )
    }

    os.makedirs("salida", exist_ok=True)
    sub = pl.DataFrame({"product_id": apre}).with_columns(
        pl.col("product_id")
        .map_elements(lambda p: pred.get(p, 0.0), return_dtype=pl.Float64)
        .alias("tn")
    )
    sub.write_csv("salida/submission_lgbm.csv")

    print(
        f"\nOK -> salida/submission_lgbm.csv  ({sub.height} productos, "
        f"tn total {sub['tn'].sum():.0f})"
    )


if __name__ == "__main__":
    main()
