# Predicción temprana de sepsis en UCI — Notebooks

Cuadernos de análisis y modelado del Trabajo de Fin de Máster **«Predicción Temprana de
Sepsis en UCI mediante Aprendizaje Profundo Multimodal con Fusión de Datos Clínicos
Heterogéneos»** (Máster en Inteligencia Artificial, Universidad Internacional de Valencia).

El objetivo es **predecir la aparición de sepsis entre 6 y 12 horas antes de su presentación
clínica**, fusionando tres modalidades heterogéneas de MIMIC-IV v3.0: series temporales de
constantes vitales, resultados de laboratorio e informes clínicos de texto libre.

Este repositorio es un **espejo público de solo lectura** de los notebooks, publicado para que
la comunidad pueda consultar los hallazgos y el proceso metodológico. El código de producción
(pipelines, entrenamiento, evaluación) reside en un repositorio privado aparte.

---

## ⚠️ Aviso sobre los datos (importante)

Este proyecto usa **MIMIC-IV v3.0** (Beth Israel Deaconess Medical Center), un dataset de acceso
restringido distribuido por [PhysioNet](https://physionet.org/content/mimiciv/) bajo un
*Data Use Agreement* (DUA) que **prohíbe la redistribución de los datos**.

Por ese motivo:

- **No se incluye ningún dato de MIMIC-IV** en este repositorio.
- Las salidas de los notebooks que mostraban registros de pacientes individuales
  (`.head()` de tablas con `subject_id`/`hadm_id`/`stay_id` y timestamps) han sido
  **redactadas**; en su lugar aparece la marca
  *«Vista previa de registros individuales omitida — DUA de PhysioNet»*.
- Todas las salidas conservadas son **agregados estadísticos, métricas o gráficas** que no
  identifican a ningún paciente.

Para ejecutar los notebooks necesitas tu propio acceso acreditado a MIMIC-IV (certificación CITI
+ credencial de PhysioNet) y descargar los datos localmente en `~/mimic-iv-3.0/`.

---

## Pipeline y notebooks

El flujo va de la exploración del dato crudo a la red de fusión multimodal. La numeración es
continua; el paso 00 (acceso y descarga de MIMIC-IV) es manual y no se documenta aquí por el DUA.

| # | Notebook | Modalidad / rol | Qué hace |
|---|----------|-----------------|----------|
| 01 | [`01_dataset_overview`](notebooks/01_dataset_overview.ipynb) | — | Exploración agregada de MIMIC-IV: estancias, demografía, *item IDs* de vitales y labs, cobertura por modalidad |
| 02 | [`02_cohort_sepsis3`](notebooks/02_cohort_sepsis3.ipynb) | — | Construcción de la cohorte **Sepsis-3** (infección sospechada por criterio de Angus + SOFA ≥ 2) |
| 03 | [`03_extract_vitals`](notebooks/03_extract_vitals.ipynb) | Vitales | Extracción y remuestreo horario de constantes vitales (`chartevents`) |
| 04 | [`04_extract_labs`](notebooks/04_extract_labs.ipynb) | Labs | Extracción de resultados de laboratorio (`labevents`) con *carry-forward* + indicadores de faltantes |
| 05 | [`05_splits`](notebooks/05_splits.ipynb) | — | Partición estratificada por paciente en *train/val/test* |
| 06 | [`06_xgboost_baseline`](notebooks/06_xgboost_baseline.ipynb) | Características | **Baseline XGBoost** sobre características clínicas (qSOFA, SOFA, SIRS) |
| 07 | [`07_tabnet`](notebooks/07_tabnet.ipynb) | Labs (tabular) | **TabNet** con atención interpretable sobre el vector de laboratorio |
| 08 | [`08_tft`](notebooks/08_tft.ipynb) | Vitales (serie) | **Temporal Fusion Transformer** sobre las series de vitales |
| 09 | [`09_clinicalbert`](notebooks/09_clinicalbert.ipynb) | Texto | **Bio\_ClinicalBERT** congelado: extracción de *embeddings* [CLS] de informes |
| 10 | [`10_fusion`](notebooks/10_fusion.ipynb) | Todas | **Fusión multimodal** con *cross-modal attention* de los tres codificadores (TabNet + TFT + Bio\_ClinicalBERT), con la ablación de la rama tabular |

---

## Resultados principales

Rendimiento sobre el conjunto de prueba (8.016 muestras, 690 positivas). En **negrita**, el mejor
valor de cada columna dentro de cada horizonte.

| Modelo | Horizonte | AUROC | AUPRC | Sens@Esp90 |
|--------|:---------:|:-----:|:-----:|:----------:|
| XGBoost (base clínica) | 6h | 0,8670 | 0,5609 | 0,6275 |
| XGBoost (base clínica) | 12h | 0,8510 | 0,5276 | 0,5905 |
| TabNet (labs) | 6h | 0,7655 | 0,3829 | 0,4348 |
| TabNet (labs) | 12h | 0,7695 | 0,3768 | 0,4460 |
| TFT (vitales) | 6h | 0,8603 | 0,6418 | 0,6536 |
| TFT (vitales) | 12h | 0,8679 | 0,6668 | 0,6667 |
| Bio\_ClinicalBERT (texto) | 6h | 0,7998 | 0,1791 | 0,3224 |
| Bio\_ClinicalBERT (texto) | 12h | 0,8040 | 0,2034 | 0,3495 |
| **Fusión multimodal (3 modelos)** | **6h** | **0,9122** | **0,7018** | **0,7290** |
| **Fusión multimodal (3 modelos)** | **12h** | **0,9165** | **0,7246** | **0,7587** |

### Hallazgos clave

- **La fusión multimodal supera ampliamente a todo modelo individual**: ≈ +0,05 puntos de
  AUROC sobre el mejor unimodal (TFT), muy por encima de la mejora habitual en fusión tardía.
- **El TFT es el mejor encoder unimodal** en las métricas sensibles al desbalanceo, con una
  ventaja decisiva en AUPRC sobre XGBoost (+0,08 a 6h, +0,14 a 12h).
- **La modelización explícita de la dinámica temporal conserva mejor la señal** al alejarse del
  *onset*: las representaciones estáticas de ventana (XGBoost) se degradan más al pasar de 6h a 12h.
- **El texto aporta señal predictiva independiente**: Bio\_ClinicalBERT alcanza AUROC ≈ 0,80 con
  un extractor congelado; su baja AUPRC refleja la baja prevalencia en la subpoblación con informe,
  no un mal rendimiento.
- **La atención inter-modal se reparte de forma equilibrada** (todos los pesos entre 0,43 y 0,57):
  ninguna modalidad domina ni resulta prescindible para las otras, lo que respalda la decisión de
  integrar las tres.
- **La ventaja frente a la línea de base clínica crece con el horizonte**: de +0,045 a +0,066 puntos
  de AUROC y de +0,102 a +0,168 de sensibilidad clínica al pasar de 6h a 12h, porque XGBoost se
  degrada al alejarse del *onset* y la fusión no.

---

## Reproducibilidad

```bash
# 1. Acceso acreditado a MIMIC-IV v3.0 y descarga local en ~/mimic-iv-3.0/
# 2. Entorno
conda create -n sepsis-tfm python=3.10 && conda activate sepsis-tfm
pip install polars pandas numpy scikit-learn xgboost pytorch-tabnet \
            torch pytorch-forecasting transformers matplotlib jupyter
# 3. Ejecutar los notebooks en orden (01 → 10)
jupyter notebook notebooks/
```

Los notebooks asumen los datos crudos en `~/mimic-iv-3.0/{hosp,icu}/` y escriben artefactos
intermedios en `data/processed/` (no incluido en este repositorio).

---

## Metodología (resumen)

- **Etiquetado**: criterios Sepsis-3 (SOFA ≥ 2 con infección sospechada); ventana de etiqueta
  desde *onset* − 6 h hasta *onset* − 12 h.
- **Desbalanceo**: prevalencia de sepsis ≈ 23 % en la cohorte; pérdida ponderada y AUROC/AUPRC
  como métricas primarias (nunca *accuracy*).
- **Alineamiento temporal**: todas las modalidades alineadas al ingreso en UCI (`intime`),
  remuestreadas a *buckets* de 1 h para vitales.
- **Datos faltantes**: vitales con *forward-fill* hasta 4 h; labs con *carry-forward* +
  indicadores de faltantes.

---

## Cita

Si utilizas este trabajo, cítalo como:

> Calvino Balonero, F. (2026). *Predicción Temprana de Sepsis en UCI mediante Aprendizaje
> Profundo Multimodal con Fusión de Datos Clínicos Heterogéneos* [Trabajo de Fin de Máster,
> Máster en Inteligencia Artificial, Universidad Internacional de Valencia (VIU)].

El dataset debe citarse según las indicaciones de PhysioNet para MIMIC-IV v3.0.

## Licencia

El **código** de los notebooks se publica bajo licencia MIT (ver [`LICENSE`](LICENSE)). Los
**datos** de MIMIC-IV **no** están cubiertos por esta licencia y no se incluyen ni redistribuyen.
