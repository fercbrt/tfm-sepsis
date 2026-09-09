"""
Métricas de evaluación compartidas por todos los notebooks del hito 2.
======================================================================

Centraliza el cálculo de las cinco métricas reportadas en la memoria para que
todos los modelos usen exactamente la misma implementación:

  1. AUROC            — discriminación global, insensible a la prevalencia.
  2. AUPRC            — discriminación sobre la clase minoritaria.
  3. Sens@Esp90       — sensibilidad en el punto de operación clínico
                        (especificidad fija del 90 %).
  4. Brier score      — error cuadrático medio entre probabilidad y etiqueta.
                        Regla de puntuación propia: por la descomposición de
                        Murphy mezcla fiabilidad, resolución e incertidumbre.
  5. ECE              — Expected Calibration Error: aísla el término de
                        fiabilidad, |confianza - frecuencia observada|.

El ECE se calcula con bins por CUANTILES (igual masa) y no equiespaciados.
Con ~8.000 muestras de prueba y una prevalencia del 8,6 %, los bins
equiespaciados dejan casi vacíos los tramos de probabilidad alta y el
resultado se vuelve inestable; los de igual masa reparten las muestras y
hacen la cifra reproducible.

Además implementa la recalibración de Platt que los cuadernos aplican como
etapa final del sistema (véase más abajo, sección «Recalibración posterior»).

Uso desde un notebook (mismo directorio). La vía normal es la calibrada, que
evalúa las dos particiones de una vez y persiste las probabilidades:

    from metricas import metricas_con_calibracion

    r = metricas_con_calibracion(y_val, p_val, y_test, p_test,
                                 etiqueta="fusión 6h",
                                 out_dir=OUT_DIR, horizonte="6h")
    r["val"], r["test"]   # cinco métricas + *_sin_calibrar + platt_a/platt_b

Para evaluar un único vector de probabilidades sin recalibrar (por ejemplo en
un estudio de ablación) siguen disponibles las piezas sueltas:

    from metricas import metricas_completas, guardar_probs
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

N_BINS_ECE = 10  # bins por cuantiles; declarado explícitamente en la memoria

# Descripción que los cuadernos guardan dentro de su results.json, para que el
# artefacto explique por sí solo qué son las columnas nuevas.
NOTA_METRICAS = (
    "Brier y ECE (10 bins por cuantiles) del sistema TRAS el reescalado de Platt, "
    "cuyos coeficientes (platt_a, platt_b) se ajustan solo con la partición de "
    "validación. Las claves *_sin_calibrar conservan los valores previos. "
    "AUROC/AUPRC/Sens@Esp90 son invariantes por ser la recalibración monótona."
)


def sens_at_spec90(y, p):
    """Sensibilidad máxima con especificidad >= 90 % (es decir, FPR <= 0,10)."""
    fpr, tpr, _ = roc_curve(y, p)
    mask = fpr <= 0.10
    return float(tpr[mask].max()) if mask.any() else 0.0


def brier(y, p):
    """Brier score: media de (p - y)^2. Menor es mejor."""
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def brier_skill_score(y, p):
    """
    Brier normalizado contra la línea base de predecir siempre la prevalencia.
    BSS = 1 - Brier / Brier_base. No se reporta en la tabla principal, pero
    sirve para contrastar modelos evaluados sobre subpoblaciones con distinta
    prevalencia (caso de Bio_ClinicalBERT).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    base = np.full_like(y, y.mean())
    b_base = float(np.mean((base - y) ** 2))
    return float(1.0 - brier(y, p) / b_base) if b_base > 0 else float("nan")


def curva_fiabilidad(y, p, n_bins=N_BINS_ECE):
    """
    Datos del diagrama de fiabilidad con bins por cuantiles.

    Devuelve (conf, obs, pesos, n_por_bin):
      conf      — probabilidad media predicha en cada bin
      obs       — frecuencia observada de la clase positiva en cada bin
      pesos     — proporción de muestras en cada bin
      n_por_bin — recuento absoluto por bin
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    # Bordes por cuantiles. np.unique colapsa bordes repetidos, que aparecen
    # cuando muchas predicciones comparten valor (colas saturadas).
    bordes = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if bordes.size < 2:  # todas las predicciones son idénticas
        return (np.array([p.mean()]), np.array([y.mean()]),
                np.array([1.0]), np.array([y.size]))

    # np.digitize con right=True deja el mínimo fuera; se corrige con clip.
    idx = np.clip(np.digitize(p, bordes[1:-1], right=True), 0, bordes.size - 2)

    conf, obs, pesos, n_bin = [], [], [], []
    for b in range(bordes.size - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        conf.append(p[m].mean())
        obs.append(y[m].mean())
        pesos.append(n / y.size)
        n_bin.append(n)

    return (np.array(conf), np.array(obs),
            np.array(pesos), np.array(n_bin))


def ece(y, p, n_bins=N_BINS_ECE):
    """
    Expected Calibration Error con bins por cuantiles.
    ECE = sum_b (n_b / N) * |conf_b - obs_b|. Menor es mejor; 0 = calibración
    perfecta a la resolución del binning.
    """
    conf, obs, pesos, _ = curva_fiabilidad(y, p, n_bins)
    return float(np.sum(pesos * np.abs(conf - obs)))


def metricas_completas(y, p, etiqueta="", verbose=True, n_bins=N_BINS_ECE):
    """
    Las cinco métricas de la memoria sobre un par (etiquetas, probabilidades).

    Devuelve un dict con las claves que consume scripts/collect_results.py:
    auroc, auprc, sens_at_spec90, brier, ece.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    m = {
        "auroc":          round(float(roc_auc_score(y, p)), 4),
        "auprc":          round(float(average_precision_score(y, p)), 4),
        "sens_at_spec90": round(sens_at_spec90(y, p), 4),
        "brier":          round(brier(y, p), 4),
        "ece":            round(ece(y, p, n_bins), 4),
    }

    if verbose:
        print(f"  {etiqueta:<34}  AUROC={m['auroc']:.4f}  "
              f"AUPRC={m['auprc']:.4f}  Sens@Spec90={m['sens_at_spec90']:.4f}  "
              f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}")

    return m


def guardar_probs(out_dir, tag, y, p):
    """
    Persiste etiquetas y probabilidades del conjunto evaluado.

    Sin esto, cualquier métrica nueva obliga a re-ejecutar el entrenamiento
    entero. Con el .npz guardado, añadir una métrica es releer el fichero.

    tag identifica horizonte y partición, p. ej. "6h_test".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / f"probs_{tag}.npz"
    np.savez_compressed(
        destino,
        y=np.asarray(y, dtype=np.float32).ravel(),
        p=np.asarray(p, dtype=np.float32).ravel(),
    )
    return destino


def cargar_probs(out_dir, tag):
    """Lee un .npz guardado por guardar_probs y devuelve (y, p)."""
    d = np.load(Path(out_dir) / f"probs_{tag}.npz")
    return d["y"], d["p"]


# ---------------------------------------------------------------------------
# Recalibración posterior
# ---------------------------------------------------------------------------
# Todos los modelos se entrenan con la clase positiva reponderada
# (pos_weight ~ 11,3), lo que desplaza las probabilidades muy por encima de la
# prevalencia real. El reescalado de Platt corrige ese desplazamiento
# ajustando una regresión logística de una variable sobre el logit del modelo,
# SIEMPRE con datos de validación. Al ser estrictamente monótono no altera el
# orden de los pacientes, de modo que AUROC, AUPRC y Sens@Esp90 quedan
# intactos y solo cambian el Brier y el ECE.

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class CalibradorPlatt:
    """
    Reescalado de Platt: sigmoide(a * logit(p) + b), con (a, b) ajustados por
    máxima verosimilitud sobre la partición de validación.

    Se implementa a mano (descenso de Newton sobre dos parámetros) para no
    añadir scikit-learn como dependencia de este módulo y para dejar el
    procedimiento explícito en el código de la memoria.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.ajustado = False

    def fit(self, y, p, n_iter=100, tol=1e-10):
        y = np.asarray(y, dtype=np.float64).ravel()
        z = _logit(p)
        a, b = 1.0, 0.0

        for _ in range(n_iter):
            q = 1.0 / (1.0 + np.exp(-(a * z + b)))
            w = np.clip(q * (1.0 - q), 1e-12, None)
            r = q - y
            # Gradiente y hessiano de la log-verosimilitud binomial en (a, b)
            g = np.array([np.sum(r * z), np.sum(r)])
            H = np.array([[np.sum(w * z * z), np.sum(w * z)],
                          [np.sum(w * z),     np.sum(w)]])
            try:
                paso = np.linalg.solve(H + 1e-10 * np.eye(2), g)
            except np.linalg.LinAlgError:
                break
            a, b = a - paso[0], b - paso[1]
            if np.max(np.abs(paso)) < tol:
                break

        self.a, self.b = float(a), float(b)
        self.ajustado = True
        return self

    def __call__(self, p):
        if not self.ajustado:
            raise RuntimeError("El calibrador no se ha ajustado todavía.")
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(p) + self.b)))


def metricas_con_calibracion(y_val, p_val, y_test, p_test, etiqueta="",
                             out_dir=None, horizonte=None, verbose=True):
    """
    Evalúa validación y prueba aplicando recalibración de Platt.

    El calibrador se ajusta EXCLUSIVAMENTE con la partición de validación y se
    aplica después a ambas particiones; el conjunto de prueba no interviene en
    el ajuste. Cada partición devuelve las cinco métricas del sistema calibrado
    más el Brier y el ECE sin calibrar, que la memoria usa para cuantificar el
    efecto de la corrección.

    Si se indican out_dir y horizonte, persiste las probabilidades crudas
    (probs_<h>_<split>.npz) y las calibradas (probs_cal_<h>_<split>.npz).
    """
    cal = CalibradorPlatt().fit(y_val, p_val)

    salida = {}
    for split, y, p in (("val", y_val, p_val), ("test", y_test, p_test)):
        p_cal = cal(p)
        m = metricas_completas(y, p_cal, etiqueta=f"{etiqueta} {split} [cal]",
                               verbose=verbose)
        m["brier_sin_calibrar"] = round(brier(y, p), 4)
        m["ece_sin_calibrar"] = round(ece(y, p), 4)
        # Los coeficientes viajan dentro de cada partición para que queden
        # registrados en el results.json que guardan los cuadernos.
        m["platt_a"] = round(cal.a, 6)
        m["platt_b"] = round(cal.b, 6)
        salida[split] = m

        if out_dir is not None and horizonte is not None:
            guardar_probs(out_dir, f"{horizonte}_{split}", y, p)
            guardar_probs(out_dir, f"cal_{horizonte}_{split}", y, p_cal)

    salida["platt"] = {"a": round(cal.a, 6), "b": round(cal.b, 6)}

    if verbose:
        print(f"      Platt (ajustado en validación): a={cal.a:.4f}  b={cal.b:.4f}"
              f"  |  Brier prueba {salida['test']['brier_sin_calibrar']:.4f}"
              f" -> {salida['test']['brier']:.4f}"
              f"  |  ECE {salida['test']['ece_sin_calibrar']:.4f}"
              f" -> {salida['test']['ece']:.4f}")

    return salida
