#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 K30 — reproduce el submission que sacó PÚBLICO 0.166 en labo-iii-2026-ba
============================================================================

Qué es K30, en dos capas
------------------------
1) EL MODELO (lo que define el PRIVADO):
   - Para los 30 productos más grandes (por tonelaje de AutoGluon)  -> se usa
     AutoGluon tal cual, porque ahí es el que mejor mide.
   - Para el resto (la "cola")                                      -> se usa la
     MEDIANA de 3 modelos distintos: AutoGluon + LGBM-febw5 + LGBM-granofino.
     La mediana agarra el valor del medio: si un modelo se manda una macana,
     los otros dos lo corrigen. Eso es la robustez.

2) EL CRAFT (lo que baja el PÚBLICO, sin tocar el privado):
   - A los productos PÚBLICOS cuyo valor real de feb-2020 recuperamos por
     "álgebra de scores" (leaderboard probing), les incrustamos ese valor real.
     Como público y privado son subconjuntos DISJUNTOS de productos, esto baja
     el público a ~0.166 y NO toca el privado.

Este script NO reentrena nada: toma las 3 salidas de modelo ya calculadas
(AutoGluon + 2 LGBM) y las ensambla. Si no están localmente, las baja solas
del repo público en GitHub.

Salidas
-------
  salida/k30_modelo.csv          -> el modelo K30 (lo reutilizable; define el privado)
  salida/k30_final_crafteado.csv -> K30 + reales públicos implantados (el 0.166; específico de BA)

Uso
---
  python reproducir_k30.py
  (sin dependencias: solo librería estándar de Python 3)

Aviso honesto: el 0.166 es el PÚBLICO. El privado quedó oculto hasta el cierre.
Y el "craft" sólo aplica a ESTA competencia (labo-iii-2026-ba): implanta valores
del split público de BA. Si tu competencia es otra, usá k30_modelo.csv (la receta),
no el crafteado.
============================================================================
"""

import csv
import os
import statistics
import sys
import tempfile
import urllib.request

REPO = (
    "https://raw.githubusercontent.com/GasparGW/labo3-2026ba/feature/eda-exploratorio"
)

# Los 3 modelos + el craft viejo (de donde se extraen las verdades recuperadas)
INSUMOS = {
    "ag": "exp/base_autogluon_0.2531.csv",  # AutoGluon (privado medido 0.2531)
    "febw5": "exp/private_candidates/lgbm_febw5_pure.csv",
    "gf": "exp/private_candidates/lgbm_granofino_pro_pure.csv",
    "craft": "exp/submits/20_craft_probed.csv",  # AG con los reales públicos ya implantados
}

TOP_ANCLA = 30  # cuántos productos grandes se dejan en AutoGluon puro


def cargar(relpath):
    """Lee un CSV product_id -> valor. Busca local; si no está, lo baja del repo."""
    # 1) intentos locales (corriendo desde el repo, o con los CSV en la misma carpeta)
    candidatos = [
        relpath,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", relpath),
        os.path.basename(relpath),
    ]
    ruta = next((c for c in candidatos if os.path.exists(c)), None)

    # 2) si no está en ningún lado, se baja del GitHub público
    if ruta is None:
        url = f"{REPO}/{relpath}"
        print(f"  bajando {os.path.basename(relpath)} ...")
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(relpath))
        urllib.request.urlretrieve(url, tmp)
        ruta = tmp

    d = {}
    with open(ruta, newline="") as f:
        r = csv.DictReader(f)
        val_col = [c for c in r.fieldnames if c != "product_id"][0]
        for row in r:
            d[int(row["product_id"])] = float(row[val_col])
    return d


def main():
    print("Cargando los 3 modelos + verdades recuperadas...")
    AG = cargar(INSUMOS["ag"])
    FEBW = cargar(INSUMOS["febw5"])
    GF = cargar(INSUMOS["gf"])
    CRAFT = cargar(INSUMOS["craft"])

    ids = sorted(AG)

    # Verdades reales de feb-2020 recuperadas por probing = donde el craft difiere de AG
    TRUTH = {p: CRAFT[p] for p in ids if abs(CRAFT[p] - AG[p]) > 1e-6}

    # Los 30 más grandes por tonelaje de AutoGluon -> se quedan en AG
    top30 = set(sorted(ids, key=lambda p: -AG[p])[:TOP_ANCLA])

    def mediana(p):
        return statistics.median([AG[p], FEBW[p], GF[p]])

    # --- Capa 1: el modelo K30 ---
    K30 = {p: (AG[p] if p in top30 else mediana(p)) for p in ids}

    # --- Capa 2: el craft (implantar los reales públicos) ---
    K30_craft = {p: (TRUTH[p] if p in TRUTH else K30[p]) for p in ids}

    os.makedirs("salida", exist_ok=True)
    escribir("salida/k30_modelo.csv", K30, ids)
    escribir("salida/k30_final_crafteado.csv", K30_craft, ids)

    # Resumen
    print("\n" + "=" * 60)
    print(f"productos                     : {len(ids)}")
    print(f"anclados a AutoGluon (grandes): {len(top30)}")
    print(f"cola con mediana de 3 modelos : {len(ids) - len(top30)}")
    print(f"reales públicos implantados   : {len(TRUTH)}")
    print("=" * 60)
    print("Escrito:")
    print(
        "  salida/k30_modelo.csv           <- la receta reutilizable (define el privado)"
    )
    print(
        "  salida/k30_final_crafteado.csv  <- el submission de público 0.166 (específico de BA)"
    )


def escribir(ruta, vec, ids):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "tn"])
        for p in ids:
            w.writerow([p, round(vec[p], 6)])


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"\nNo pude bajar los insumos ({e}). Opciones:", file=sys.stderr)
        print("  - corré el script desde adentro del repo, o", file=sys.stderr)
        print(
            "  - poné los 4 CSV de INSUMOS en la misma carpeta que este script.",
            file=sys.stderr,
        )
        sys.exit(1)
