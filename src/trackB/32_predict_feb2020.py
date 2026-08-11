"""
Track B - Genera la receta VALIDADA (diverse-median) para feb-2020 (parado 201912 -> 202002)
y la compara contra AG (0.2531 medido) y K30 en las verdades reales feb-2020 recuperadas.

Receta validada en 2 febreros honestas: median(last_value, lgbm_l1, lgbm_tweedie, arima_ratio).
Escribe candidatos a scratchpad para submitear.
"""

import polars as pl, numpy as np, warnings, csv, os

warnings.filterwarnings("ignore")
import lightgbm as lgb

SP = "/private/tmp/claude-501/-Users-gaspargonzalezwulfsohn-labo3-2026ba/c8701a5c-96f6-43ce-81c6-6349ab686997/scratchpad"
PANEL = "datasets/panel_producto.parquet"
APRE = "datasets/product_id_apredecir201912.txt"
STAND, TARGET = 201912, 202002
LAGS = [1, 2, 3, 4, 6, 12]
ROLL = [3, 6, 12]
FEATS = ["tn", "month", "d3"] + [f"lag{L}" for L in LAGS] + [f"rm{W}" for W in ROLL]


def load_vec(p):
    return {
        int(r["product_id"]): float(r[[k for k in r if k != "product_id"][0]])
        for r in csv.DictReader(open(p))
    }


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
    long = long.with_columns(
        [
            pl.col("tn").shift(-2).over("product_id").alias("target"),
            pl.col("period").shift(-2).over("product_id").alias("target_period"),
            (pl.col("tn") - pl.col("tn").shift(3).over("product_id")).alias("d3"),
        ]
    )
    return long, periods


def lgbm_pred(long, stand, apre, objective="l1"):
    tr = long.filter(
        (pl.col("target").is_not_null()) & (pl.col("target_period") <= stand)
    )
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
            **({"tweedie_variance_power": 1.2} if objective == "tweedie" else {}),
        ),
        lgb.Dataset(
            tr.select(FEATS).to_numpy(), label=tr.select("target").to_numpy().ravel()
        ),
        num_boost_round=300,
    )
    pr = long.filter(pl.col("period") == stand)
    pred = {
        int(pid): max(0.0, float(v))
        for pid, v in zip(
            pr["product_id"].to_list(), m.predict(pr.select(FEATS).to_numpy())
        )
    }
    return {p: pred.get(p, 0.0) for p in apre}


def arima_ratio(P, apre, periods, stand):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    pers = [int(x) for x in periods if int(x) <= stand]
    rows = [
        (p, i, (P[p].get(str(per), 0.0) if p in P else 0.0))
        for p in apre
        for i, per in enumerate(pers)
    ]
    sf_df = pl.DataFrame(
        rows, schema=["unique_id", "ds", "y"], orient="row"
    ).to_pandas()
    fc = StatsForecast(models=[AutoARIMA(season_length=12)], freq=1, n_jobs=1).forecast(
        df=sf_df, h=2
    )
    out = {}
    for uid, g in fc.groupby("unique_id"):
        s1, s2 = float(g["AutoARIMA"].iloc[0]), float(g["AutoARIMA"].iloc[1])
        last = P[uid].get(str(stand), 0.0)
        out[uid] = max(0.0, last * (np.clip(s2 / s1, 0.5, 2.0) if s1 > 1e-6 else 1.0))
    return {p: out.get(p, 0.0) for p in apre}


def tfe(pred, real, ids):
    T = sum(real[p] for p in ids)
    return sum(abs(pred.get(p, 0.0) - real[p]) for p in ids) / T


def write(name, vec, apre):
    with open(f"{SP}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "tn"])
        for p in apre:
            w.writerow([p, round(vec[p], 6)])


if __name__ == "__main__":
    long, periods = build_long()
    prod = pl.read_parquet(PANEL)
    P = {r["product_id"]: r for r in prod.iter_rows(named=True)}
    apre = pl.read_csv(APRE, separator="\t")["product_id"].to_list()

    last = {p: (P[p].get(str(STAND), 0.0) if p in P else 0.0) for p in apre}
    lg_l1 = lgbm_pred(long, STAND, apre, "l1")
    lg_tw = lgbm_pred(long, STAND, apre, "tweedie")
    ar = arima_ratio(P, apre, periods, STAND)

    AG = load_vec("exp/base_autogluon_0.2531.csv")
    CRAFT = load_vec("exp/submits/20_craft_probed.csv")
    TRUTH = {
        p: CRAFT[p] for p in apre if abs(CRAFT[p] - AG[p]) > 1e-6
    }  # 17 reales feb-2020
    top30 = set(sorted(apre, key=lambda p: -AG[p])[:30])

    def m(*ds):
        return {p: float(np.median([d[p] for d in ds])) for p in apre}

    DM4 = m(last, lg_l1, lg_tw, ar)  # receta validada (sin AG)
    DM5 = m(last, lg_l1, lg_tw, ar, AG)  # + AG (mas diverso, no validable)
    DM4_ag30 = {
        p: (AG[p] if p in top30 else DM4[p]) for p in apre
    }  # AG en grandes, DM4 en cola

    cands = {
        "AG (0.2531 medido)": AG,
        "last_value_2020": last,
        "lgbm_l1": lg_l1,
        "lgbm_tw": lg_tw,
        "arima_ratio": ar,
        "DM4 (validado)": DM4,
        "DM5 (+AG)": DM5,
        "DM4_agK30": DM4_ag30,
    }
    Ttot = sum(AG.values())
    print(f"{'candidato':24} {'TFE_17real':>10} {'dist_a_AG':>10} {'dist_a_last':>11}")
    print("-" * 60)
    for name, v in cands.items():
        d_ag = sum(abs(v[p] - AG[p]) for p in apre) / Ttot
        d_la = sum(abs(v[p] - last[p]) for p in apre) / sum(last.values())
        print(f"{name:24} {tfe(v, TRUTH, list(TRUTH)):10.4f} {d_ag:10.3f} {d_la:11.3f}")
    for name, v in [
        ("trackB_DM4", DM4),
        ("trackB_DM5_AG", DM5),
        ("trackB_DM4_agK30", DM4_ag30),
    ]:
        write(name, v, apre)
    print(f"\nescritos en {SP}: trackB_DM4, trackB_DM5_AG, trackB_DM4_agK30")
