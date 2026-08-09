# Candidatos para el private (para submitear cuando resetee el cupo)

IMPORTANTE: el private esta OCULTO hasta el 16-ago. Estos son apuestas informadas,
validadas en las 25 respuestas reales de feb-2020 (proxy). El proxy PUEDE mentir para
el private (LR13 le gana al proxy pero pierde en private). Ancla segura = autogluon_pure
(private 0.2531 MEDIDO, 2do puesto real).

| candidato | TFE en 25 reales (proxy) | notas |
|---|---|---|
| autogluon_pure | 0.1999 | ANCLA: private 0.2531 medido, no apostar en contra sin razon |
| median_ag_lr_mg | 0.1714 | mejor proxy; mediana robusta corrige outliers de AG sin irse a LR13 |
| median_ag_lr_lgbm | 0.1839 | mediana con LGBM en vez de magicos |
| ag80_lr20 | 0.1926 | blend conservador, mayoria AutoGluon |
| ag70_lr30 | 0.1899 | blend un poco mas hacia LR13 |
| ag60_lr20_lgbm20 | 0.1905 | blend 3 vias anclado en AG |

## Candidatos LGBM (private NUNCA medido - apuesta genuinamente abierta)

El LGBM se corrio el 8-ago, despues del export, asi que su private esta 100% oculto.
En las 25 reales anda parecido a AutoGluon (no claramente mejor ni peor).

| candidato | TFE 25 reales |
|---|---|
| lgbm_febw5_pure | 0.1938 |
| lgbm_granofino_pro_pure | 0.2142 |
| median_ag_lgbm5_lgbmgf | 0.2027 |
| ag50_lgbmfebw5_50 | 0.1941 |

---

## HANDOFF: cómo continuar en la OTRA máquina

Contexto completo en `docs/bitacora-20260808.md` (leerlo primero). Estado actual: la
submission seleccionada como final (1/1) es el craft con **público 0.171 / private 0.2531**
(2º puesto medido). Competencia cierra **2026-08-16**, tope ~100 submits/día.

**Requisito (NO viaja por git, es secreto):** copiar el token de Kaggle a esta máquina en
`~/.kaggle/kaggle.json` (chmod 600). Está en el Desktop de la máquina original como `kaggle.json`.
Instalar el CLI: `pip install kaggle` (o `uv pip install kaggle`).

**Cuando resetee el cupo, submitear los 10 candidatos y leer sus PÚBLICOS:**

```bash
cd <repo>
for f in exp/private_candidates/*.csv; do
  kaggle competitions submit -c labo-iii-2026-ba -f "$f" -m "$(basename "$f" .csv)"
  sleep 3
done
# esperar ~1 min a que puntúen, luego:
kaggle competitions submissions -c labo-iii-2026-ba --csv --page-size 200
```

**Decisión final (marcar 1 sola en Kaggle > Submissions > checkbox "Select"):**
- SEGURO: dejar `autogluon_pure` / el craft actual → private **0.2531** (2º puesto medido).
- APUESTA: elegir el mejor candidato (p.ej. `median_ag_lr_mg`) → private OCULTO, quizás mejor,
  quizás peor. Elegirlo = resignar el 0.2531 medido. El público de los candidatos NO confirma
  el private (ya demostrado: LR13 gana público, pierde private).

El público que saquen los candidatos es validación sobre el split público entero (~390 reales),
más fuerte que las 25 — pero sigue sin garantizar el private. Sin datos nuevos del private,
la jugada racional es dejar el 0.2531.
