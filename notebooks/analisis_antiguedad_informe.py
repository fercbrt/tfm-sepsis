"""
Análisis de la antigüedad del informe de radiología seleccionado.

MOTIVACIÓN
----------
La ablación leave-one-out mostró que eliminar el texto cuesta -0,046 de AUROC
a 6h. Pero el informe seleccionado es "el más reciente anterior a t_ref", sin
límite de antigüedad: puede ser de hace una hora o de hace una semana. Si buena
parte de los informes son antiguos, la contribución medida es una COTA INFERIOR
de lo que aportaría la modalidad con una ventana acotada.

Este análisis responde dos preguntas:
  (1) ¿Cómo se distribuye la antigüedad de los informes seleccionados?
  (2) ¿Discrimina mejor el texto cuando el informe es reciente?

CÓMO USARLO
-----------
Se ejecuta como celda en `09_clinicalbert.ipynb`, DESPUÉS de construir
`notes_6h` / `notes_12h` y de disponer de las predicciones del clasificador
sobre test.

SUPUESTOS SOBRE EL CUADERNO
---------------------------
  - `notes_6h` NO tiene `charttime` ni `ref_time`. El `group_by("stay_id").agg()`
    de `assign_notes` produce: stay_id, note_text, note_charttime, sepsis, split.
    Por eso la columna de fecha del informe es `note_charttime` y `ref_time` se
    recupera de la cohorte con `con_ref_time()` (misma fórmula que el cuaderno:
    intime + ref_hour horas).
  - Las predicciones de test no viven en ninguna variable: se reconstruyen desde
    los artefactos guardados (`cls_{h}_test.npy`, `stay_ids_{h}_test.npy`,
    `labels_{h}_test.npy` y `mlp_{h}.pt`) con `predicciones_test()`. El PASO 7 del
    cuaderno los exporta ordenados por stay_id y perfectamente alineados, así que
    no hace falta volver a pasar BERT.

CONFUSOR QUE CONDICIONA LA LECTURA
----------------------------------
Los pacientes más graves reciben imagen con más frecuencia, de modo que un
informe reciente puede ser en sí mismo un marcador de gravedad. Si el AUROC
sube en los tramos recientes, hay que comprobar antes si la PREVALENCIA también
sube: en ese caso el efecto estaría confundido y no probaría que el texto
"envejece". Por eso la tabla reporta prevalencia por tramo junto al AUROC.
"""

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

# Tramos de antigüedad en horas.
BINS   = [0, 6, 12, 24, 72, 168, np.inf]
LABELS = ["≤6h", "6–12h", "12–24h", "1–3d", "3–7d", ">7d"]

# Mínimo de casos positivos por tramo para que el AUROC sea interpretable.
MIN_POS = 30


# ==========================================================================
# 1. Antigüedad del informe
# ==========================================================================
def con_ref_time(notes_df, cohort_h):
    """Recupera `ref_time` desde la cohorte y lo une por stay_id.

    `assign_notes` no propaga ref_time a través del group_by, así que se
    reconstruye con la misma fórmula que usa el cuaderno: intime + ref_hour.
    """
    ref = (
        cohort_h
        .with_columns(
            (pl.col("intime") + pl.duration(hours=pl.col("ref_hour"))).alias("ref_time")
        )
        .select(["stay_id", "ref_time"])
    )
    return notes_df.join(ref, on="stay_id", how="left")


def con_antiguedad(notes_df, col_ref="ref_time", col_chart="note_charttime"):
    """Añade la columna `edad_h`: horas entre el informe y el tiempo de referencia."""
    return notes_df.with_columns(
        ((pl.col(col_ref) - pl.col(col_chart)).dt.total_seconds() / 3600.0)
        .alias("edad_h")
    )


def predicciones_test(horizon, out_dir, mlp_cls, device):
    """Predicciones del clasificador de BERT sobre test, por stay_id.

    Usa los artefactos del PASO 7/8 del cuaderno (embeddings [CLS] ordenados por
    stay_id y pesos del MLP), de modo que no hay que reejecutar BERT.
    """
    sids = np.load(out_dir / f"stay_ids_{horizon}_test.npy")
    embs = np.load(out_dir / f"cls_{horizon}_test.npy")
    lbls = np.load(out_dir / f"labels_{horizon}_test.npy")
    assert len(sids) == len(embs) == len(lbls), "desalineación stay_id/emb/label"

    mlp = mlp_cls().to(device)
    mlp.load_state_dict(torch.load(out_dir / f"mlp_{horizon}.pt", map_location=device))
    mlp.eval()
    with torch.no_grad():
        probs = torch.sigmoid(mlp(torch.from_numpy(embs).to(device))).cpu().numpy()

    return pl.DataFrame({
        "stay_id": sids.astype(np.int64),
        "y_true":  lbls.astype(np.int64),
        "y_prob":  probs.astype(np.float64),
    })


def describir(notes_df, etiqueta=""):
    """Percentiles y reparto por tramo. Devuelve (df_con_edad, percentiles, reparto)."""
    d = con_antiguedad(notes_df)
    e = d["edad_h"].drop_nulls().to_numpy()

    print(f"\n=== Antigüedad del informe {etiqueta} (n={len(e):,}) ===")
    pcts = {}
    for q in (10, 25, 50, 75, 90, 95):
        pcts[f"p{q}"] = float(np.percentile(e, q))
        print(f"  p{q:<3} {pcts[f'p{q}']:8.1f} h")
    print(f"  media {e.mean():7.1f} h   máx {e.max():.1f} h")
    pcts["media"] = float(e.mean())
    pcts["max"]   = float(e.max())

    print("\n  Distribución por tramo:")
    idx = np.digitize(e, BINS[1:-1], right=True)
    reparto = {}
    for i, lab in enumerate(LABELS):
        n = int((idx == i).sum())
        reparto[lab] = n
        print(f"    {lab:<7} {n:7,}  ({100*n/len(e):5.1f} %)")
    return d, pcts, reparto


# ==========================================================================
# 2. Rendimiento estratificado por antigüedad
# ==========================================================================
def auroc_por_tramo(edad_h, y_true, y_prob, etiqueta=""):
    """AUROC y prevalencia por tramo de antigüedad.

    La prevalencia es imprescindible: si sube en los tramos recientes, el
    efecto sobre el AUROC puede estar confundido por gravedad.
    """
    edad_h = np.asarray(edad_h, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    idx = np.digitize(edad_h, BINS[1:-1], right=True)

    print(f"\n=== AUROC por antigüedad del informe {etiqueta} ===")
    print(f"  {'Tramo':<8} {'n':>7} {'pos':>6} {'neg':>7} {'prev':>7} {'AUROC':>8}")
    print("  " + "-" * 48)

    filas = []
    for i, lab in enumerate(LABELS):
        m = idx == i
        n, pos = int(m.sum()), int(y_true[m].sum())
        if n == 0:
            continue
        neg  = n - pos
        prev = pos / n
        if pos >= MIN_POS and neg >= MIN_POS:
            auc = roc_auc_score(y_true[m], y_prob[m])
            auc_s = f"{auc:.4f}"
        else:
            auc, auc_s = np.nan, "  n/d"
        print(f"  {lab:<8} {n:7,} {pos:6,} {neg:7,} {prev:6.1%} {auc_s:>8}")
        filas.append({"tramo": lab, "n": n, "pos": pos, "neg": neg,
                      "prevalencia": round(prev, 4),
                      "auroc": None if np.isnan(auc) else round(auc, 4),
                      "interpretable": bool(pos >= MIN_POS and neg >= MIN_POS)})

    print(f"\n  n/d = menos de {MIN_POS} positivos en el tramo; AUROC no interpretable.")
    print("  ⚠️ Compara la columna de prevalencia entre tramos antes de atribuir")
    print("     al texto cualquier tendencia del AUROC.")
    return filas


def analizar(horizon, notes_df, cohort_h, out_dir, mlp_cls, device):
    """Análisis completo de un horizonte. Devuelve el DataFrame de resultados."""
    import pandas as pd

    d, pcts, reparto = describir(con_ref_time(notes_df, cohort_h), f"[{horizon}]")

    preds = predicciones_test(horizon, out_dir, mlp_cls, device)
    j = d.join(preds, on="stay_id", how="inner")

    # El join debe cubrir exactamente el split de test que exportó el cuaderno.
    n_test_notas = d.filter(pl.col("split") == "test").height
    assert len(j) == len(preds) == n_test_notas, (
        f"desajuste de cobertura: join={len(j)} preds={len(preds)} test={n_test_notas}")
    assert (j["y_true"] == j["sepsis"]).all(), "las etiquetas guardadas no cuadran"

    # Comprobación global: debe reproducir el AUROC de test de results.json.
    auroc_global = roc_auc_score(j["y_true"].to_numpy(), j["y_prob"].to_numpy())
    print(f"\n  AUROC global de test reconstruido {horizon}: {auroc_global:.4f}")

    filas = auroc_por_tramo(j["edad_h"], j["y_true"], j["y_prob"], f"[BERT {horizon}]")

    # Una sola lista de filas: así los conteos conservan dtype entero en el CSV.
    n_all = sum(reparto.values())
    columnas = ["horizonte", "seccion", "auroc_global_test", "tramo",
                "n_test", "pos_test", "neg_test", "prevalencia", "auroc",
                "interpretable", "n_todos_splits", "pct_todos_splits",
                "pct_test", "valor_h"]
    rows = []
    for f in filas:
        rows.append({
            "horizonte": horizon, "seccion": "tramo",
            "auroc_global_test": round(auroc_global, 4), "tramo": f["tramo"],
            "n_test": f["n"], "pos_test": f["pos"], "neg_test": f["neg"],
            "prevalencia": f["prevalencia"], "auroc": f["auroc"],
            "interpretable": f["interpretable"],
            "n_todos_splits": reparto[f["tramo"]],
            "pct_todos_splits": round(100 * reparto[f["tramo"]] / n_all, 2),
            "pct_test": round(100 * f["n"] / len(j), 2),
            "valor_h": None,
        })
    for k, v in pcts.items():
        rows.append({
            "horizonte": horizon, "seccion": "percentil",
            "auroc_global_test": round(auroc_global, 4), "tramo": k,
            "valor_h": round(v, 2),
        })
    df = pd.DataFrame(rows, columns=columnas)
    # Enteros anulables: las filas de percentil dejan NA en estas columnas y sin
    # esto pandas las convertiría a float ("1141.0" en el CSV).
    return df.astype({c: "Int64" for c in
                      ("n_test", "pos_test", "neg_test", "n_todos_splits")})


# --------------------------------------------------------------------------
# EJECUTAR  (nombres reales del cuaderno 09_clinicalbert)
# --------------------------------------------------------------------------
# df_ant_6h  = analizar("6h",  notes_6h,  cohort_6h,  OUT_DIR,
#                       SepsisMLPClassifier, DEVICE)
# df_ant_12h = analizar("12h", notes_12h, cohort_12h, OUT_DIR,
#                       SepsisMLPClassifier, DEVICE)
# df_ant_6h.to_csv(OUT_DIR / "antiguedad_informe_6h.csv", index=False)
# df_ant_12h.to_csv(OUT_DIR / "antiguedad_informe_12h.csv", index=False)
