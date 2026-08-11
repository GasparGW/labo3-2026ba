"""
Track B - LGBM a nivel producto sobre el harness honesto de febrero, + blends diversos.

- LGBM con lags/rollings, target = tn a +2 (mismo gap que la competencia).
- ARIMA anclado por RATIO: en vez del nivel de ARIMA (ruidoso), aplico su crecimiento
  esperado al ultimo valor observado: pred = last * clip(arima_step2 / arima_step1).
- Blends: mediana robusta de familias diversas (last, lgbm, arima-ratio).
Se valida en feb-2019 (parado 201812) y feb-2018 (parado 201712).
Uso: python src/trackB/31_lgbm_febharness.py
"""

import polars as pl, numpy as np, warnings, time

warnings.filterwarnings("ignore")
import lightgbm as lgb

PANEL = "datasets/panel_producto.parquet"
APRE = "datasets/product_id_apredecir201912.txt"
LAGS = [1, 2, 3, 4, 6, 12]
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
    long = long.with_columns(
        [
            pl.col("tn").shift(-2).over("product_id").alias("target"),
            pl.col("period").shift(-2).over("product_id").alias("target_period"),
            (pl.col("tn") - pl.col("tn").shift(3).over("product_id")).alias("d3"),
        ]
    )
    return long, periods


FEATS = ["tn", "month", "d3"] + [f"lag{L}" for L in LAGS] + [f"rm{W}" for W in ROLL]


def lgbm_pred(long, stand, objective="l1"):
    # HONESTO: el target (t+2) debe estar observado al momento de 'stand' -> target_period <= stand
    tr = long.filter(
        (pl.col("target").is_not_null()) & (pl.col("target_period") <= stand)
    )
    X = tr.select(FEATS).to_numpy()
    y = tr.select("target").to_numpy().ravel()
    ds = lgb.Dataset(X, label=y)
    params = dict(
        objective=objective,
        num_leaves=63,
        learning_rate=0.05,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbose=-1,
    )
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.2
    m = lgb.train(params, ds, num_boost_round=300)
    pr = long.filter(pl.col("period") == stand)
    pred = m.predict(pr.select(FEATS).to_numpy())
    return {
        int(pid): max(0.0, float(v)) for pid, v in zip(pr["product_id"].to_list(), pred)
    }


def arima_ratio(P, apre, periods, stand):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    pers = [int(x) for x in periods if int(x) <= stand]
    rows = []
    for p in apre:
        for i, per in enumerate(pers):
            rows.append((p, i, P[p].get(str(per), 0.0) if p in P else 0.0))
    sf_df = pl.DataFrame(
        rows, schema=["unique_id", "ds", "y"], orient="row"
    ).to_pandas()
    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq=1, n_jobs=1)
    fc = sf.forecast(df=sf_df, h=2)
    out = {}
    for uid, g in fc.groupby("unique_id"):
        s1, s2 = float(g["AutoARIMA"].iloc[0]), float(g["AutoARIMA"].iloc[1])
        last = P[uid].get(str(stand), 0.0)
        ratio = np.clip(s2 / s1, 0.5, 2.0) if s1 > 1e-6 else 1.0
        out[uid] = max(0.0, last * ratio)  # crecimiento esperado aplicado al nivel real
    return out


def tfe(pred, real, apre):
    T = sum(real[p] for p in apre)
    return sum(abs(pred.get(p, 0.0) - real[p]) for p in apre) / T


def med(*ds):
    return lambda p: float(np.median([d.get(p, 0.0) for d in ds]))


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
        t0 = time.time()
        lg_l1 = lgbm_pred(long, stand, "l1")
        lg_tw = lgbm_pred(long, stand, "tweedie")
        ar = arima_ratio(P, apre, periods, stand)
        dt = time.time() - t0
        cands = {
            "last_value": last,
            "lgbm_l1": lg_l1,
            "lgbm_tweedie": lg_tw,
            "arima_ratio": ar,
            "median(last,lgbm_l1)": {p: med(last, lg_l1)(p) for p in apre},
            "median(last,lgbm_l1,arima_r)": {p: med(last, lg_l1, ar)(p) for p in apre},
            "median(last,lgbm_l1,lgbm_tw)": {
                p: med(last, lg_l1, lg_tw)(p) for p in apre
            },
            "median(last,lgbm_l1,lgbm_tw,arima_r)": {
                p: med(last, lg_l1, lg_tw, ar)(p) for p in apre
            },
        }
        print(f"\n=== {tag} (parado {stand}, +2) — TFE honesto 780 | {dt:.0f}s ===")
        for k, v in cands.items():
            print(f"    {k:38} {tfe(v, real, apre):.4f}")
