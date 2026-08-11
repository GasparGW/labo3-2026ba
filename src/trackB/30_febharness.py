"""
Track B - Harness honesto de febrero.

El target real es predecir 202002 parado en 201912 (gap +2: 201912 -> 202001 -> 202002).
El validador honesto dentro de los datos es predecir un FEBRERO pasado con el mismo gap:
  - feb-2019: parado 201812, predecir 201902
  - feb-2018: parado 201712, predecir 201802
Metrica: TFE = Sum|pred-real| / Sum(real) sobre los 780 productos a predecir.

Ejecuta baselines + AutoARIMA (3a familia, errores independientes de AG/LGBM) y sus medianas.
Uso: python src/trackB/30_febharness.py
"""

import polars as pl, numpy as np, warnings, time, sys

warnings.filterwarnings("ignore")

PANEL = "datasets/panel_producto.parquet"
APRE = "datasets/product_id_apredecir201912.txt"


def load():
    prod = pl.read_parquet(PANEL)
    apre = pl.read_csv(APRE, separator="\t")["product_id"].to_list()
    periods = [c for c in prod.columns if c != "product_id"]
    P = {r["product_id"]: r for r in prod.iter_rows(named=True)}
    return P, apre, periods


def get(P, p, per):
    return P[p].get(str(per), 0.0) if p in P else 0.0


def tfe(pred, real, apre):
    T = sum(real[p] for p in apre)
    return sum(abs(pred.get(p, 0.0) - real[p]) for p in apre) / T


def run_arima(P, apre, periods, stand, n_jobs=1):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    pers = [int(x) for x in periods if int(x) <= stand]
    rows = []
    for p in apre:
        for i, per in enumerate(pers):
            rows.append((p, i, get(P, p, per)))
    sf_df = pl.DataFrame(
        rows, schema=["unique_id", "ds", "y"], orient="row"
    ).to_pandas()
    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq=1, n_jobs=n_jobs)
    fc = sf.forecast(df=sf_df, h=2)
    pred = {}
    for uid, g in fc.groupby("unique_id"):
        pred[uid] = max(0.0, float(g["AutoARIMA"].iloc[1]))  # h=2 = +2
    return pred


def harness(P, apre, periods, stand, target, n_jobs=1):
    real = {p: get(P, p, target) for p in apre}
    lastv = {p: get(P, p, stand) for p in apre}
    # meses previos al stand (para ma3)
    prev = [int(x) for x in periods if int(x) <= stand][-3:]
    ma3 = {p: sum(get(P, p, x) for x in prev) / len(prev) for p in apre}
    t0 = time.time()
    ar = run_arima(P, apre, periods, stand, n_jobs)
    dt = time.time() - t0
    med_la = {p: float(np.median([lastv[p], ar.get(p, 0.0)])) for p in apre}
    med3 = {p: float(np.median([lastv[p], ar.get(p, 0.0), ma3[p]])) for p in apre}
    out = {
        "last_value": tfe(lastv, real, apre),
        "ma3": tfe(ma3, real, apre),
        "AutoARIMA": tfe(ar, real, apre),
        "median(last,arima)": tfe(med_la, real, apre),
        "median(last,arima,ma3)": tfe(med3, real, apre),
    }
    return out, dt, real


if __name__ == "__main__":
    P, apre, periods = load()
    for stand, target, tag in [
        (201812, 201902, "FEB-2019"),
        (201712, 201802, "FEB-2018"),
    ]:
        res, dt, real = harness(P, apre, periods, stand, target, n_jobs=1)
        print(
            f"\n=== HARNESS {tag} (parado {stand}, +2) — TFE honesto sobre 780 | ARIMA {dt:.0f}s ==="
        )
        print(f"    Sigma real = {sum(real[p] for p in apre):.0f} tn")
        for k, v in res.items():
            print(f"    {k:26} {v:.4f}")
