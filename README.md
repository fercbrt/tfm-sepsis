# Predicción temprana de sepsis en UCI — Código

Código de análisis y modelado del Trabajo de Fin de Máster **«Predicción Temprana de Sepsis en
UCI mediante Aprendizaje Profundo Multimodal con Fusión de Datos Clínicos Heterogéneos»**
(Máster en Inteligencia Artificial, Universidad Internacional de Valencia).

El objetivo es **predecir la aparición de sepsis con 6 y 12 horas de antelación** respecto a su
presentación clínica, fusionando tres modalidades heterogéneas de MIMIC-IV v3.0: series
temporales de signos vitales, un vector tabular de laboratorio y signos vitales agregados, e
informes de radiología en texto libre.

Este repositorio contiene **la totalidad del código de análisis y modelado** del trabajo: los
cuadernos reproducen el *pipeline* completo, desde la exploración del dato de partida hasta la
red de fusión y sus estudios de ablación.

---

## ⚠️ Aviso sobre los datos (importante)

Este proyecto usa **MIMIC-IV v3.0** (Beth Israel Deaconess Medical Center), un dataset de acceso
restringido distribuido por [PhysioNet](https://physionet.org/content/mimiciv/) bajo un
*Data Use Agreement* (DUA) que **prohíbe la redistribución de los datos**.

Por ese motivo:

- **No se incluye ningún dato de MIMIC-IV** en este repositorio.
- Las salidas de los cuadernos que mostraban registros de pacientes individuales
  (`.head()` de tablas con `subject_id`/`hadm_id`/`stay_id` y timestamps) han sido
  **redactadas**; en su lugar aparece la marca
  *«Vista previa de registros individuales omitida — DUA de PhysioNet»*.
- Todas las salidas conservadas son **agregados estadísticos, métricas o gráficas** que no
  identifican a ningún paciente.

Para ejecutar los cuadernos necesitas tu propio acceso acreditado a MIMIC-IV (certificación CITI
+ credencial de PhysioNet) y descargar los datos localmente en `~/mimic-iv-3.0/`.

---

## Pipeline y cuadernos

El flujo va de la exploración del dato crudo a la red de fusión multimodal. La numeración es
continua; el paso 00 (acceso y descarga de MIMIC-IV) es manual y no se documenta aquí por el DUA.

| # | Cuaderno | Modalidad / rol | Qué hace |
|---|----------|-----------------|----------|
| 01 | [`01_dataset_overview`](notebooks/01_dataset_overview.ipynb) | — | Exploración agregada de MIMIC-IV: estancias, demografía, *item IDs* de vitales y labs, cobertura por modalidad |
| 02 | [`02_cohort_sepsis3`](notebooks/02_cohort_sepsis3.ipynb) | — | Construcción de la cohorte **Sepsis-3**: infección sospechada según el criterio de Angus y ΔSOFA ≥ 2 |
| 03 | [`03_extract_vitals`](notebooks/03_extract_vitals.ipynb) | Vitales | Extracción y remuestreo horario de los signos vitales (`chartevents`) |
| 04 | [`04_extract_labs`](notebooks/04_extract_labs.ipynb) | Labs | Extracción de resultados de laboratorio (`labevents`) con *carry-forward* e indicadores de faltantes |
| 05 | [`05_splits`](notebooks/05_splits.ipynb) | — | Partición estratificada **a nivel de paciente** en *train/val/test* |
| 06 | [`06_xgboost_baseline`](notebooks/06_xgboost_baseline.ipynb) | Tabular | **Línea de base clínica XGBoost** sobre el vector de 99 características: estadísticos de ventana de vitales y laboratorio, qSOFA y SIRS. Las puntuaciones SOFA se excluyen (ver *Metodología*) |
| 07 | [`07_tabnet`](notebooks/07_tabnet.ipynb) | Tabular | **TabNet** sobre el mismo vector de 99 características, como línea de base tabular interpretable |
| 08 | [`08_tft`](notebooks/08_tft.ipynb) | Vitales (serie) | **Temporal Fusion Transformer** sobre las series de vitales; codificador temporal de la fusión (64 dim) |
| 09 | [`09_clinicalbert`](notebooks/09_clinicalbert.ipynb) | Texto | **Bio\_ClinicalBERT** congelado: extracción de *embeddings* [CLS] de los informes de radiología (768 dim) |
| 10 | [`10_fusion`](notebooks/10_fusion.ipynb) | Todas | **Fusión multimodal** con *cross-modal attention*: TFT + Bio\_ClinicalBERT + vector tabular incorporado **por proyección lineal directa**, sin codificador intermedio. Incluye la ablación del codificador tabular |
| — | [`metricas.py`](notebooks/metricas.py) | — | Módulo compartido: las cinco métricas, la recalibración de Platt y la persistencia de probabilidades. Lo importan los cuadernos 06–10 |
| — | [`ablacion_leave_one_out.py`](notebooks/ablacion_leave_one_out.py) | — | Ablación por eliminación de modalidad, invocada desde `10_fusion` |
| — | [`analisis_antiguedad_informe.py`](notebooks/analisis_antiguedad_informe.py) | — | Análisis **exploratorio** de la antigüedad de los informes seleccionados. **No se llegó a ejecutar**: sus resultados no forman parte del trabajo |

> **TabNet no forma parte de la red de fusión.** Opera únicamente como línea de base. Su
> representación de 32 dimensiones se evaluó como rama tabular alternativa y se descartó: ver la
> ablación del codificador tabular en *Hallazgos clave*.

---

## Resultados principales

Rendimiento sobre el conjunto de prueba (8.016 muestras, 690 positivas). En **negrita**, el mejor
valor de cada columna dentro de cada horizonte; en Brier y ECE el mejor valor es el **menor**.

Las tres primeras columnas miden **discriminación** (ordenar pacientes por riesgo); las dos
últimas, **calidad probabilística** (que la probabilidad anunciada corresponda a la frecuencia
real). El Brier y el ECE son los del sistema tras la recalibración de Platt descrita abajo.

| Modelo | Horizonte | AUROC | AUPRC | Sens@Esp90 | Brier | ECE |
|--------|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| XGBoost (base clínica) | 6h | 0,8670 | 0,5609 | 0,6275 | 0,0540 | 0,0097 |
| XGBoost (base clínica) | 12h | 0,8510 | 0,5276 | 0,5905 | 0,0578 | 0,0095 |
| TabNet | 6h | 0,7655 | 0,3829 | 0,4348 | 0,0643 | 0,0108 |
| TabNet | 12h | 0,7695 | 0,3768 | 0,4460 | 0,0662 | 0,0132 |
| TFT (vitales) | 6h | 0,8603 | 0,6418 | 0,6536 | 0,0452 | **0,0069** |
| TFT (vitales) | 12h | 0,8679 | 0,6668 | 0,6667 | 0,0442 | 0,0101 |
| Bio\_ClinicalBERT (radiología) | 6h | 0,7998 | 0,1791 | 0,3224 | 0,0603 | 0,0086 |
| Bio\_ClinicalBERT (radiología) | 12h | 0,8040 | 0,2034 | 0,3495 | 0,0622 | 0,0127 |
| **Fusión multimodal** | **6h** | **0,9307** | **0,7306** | **0,7681** | **0,0401** | 0,0075 |
| **Fusión multimodal** | **12h** | **0,9277** | **0,7340** | **0,7619** | **0,0403** | **0,0054** |

> La AUPRC y el Brier dependen de la prevalencia, por lo que las cifras de Bio\_ClinicalBERT
> —evaluado sobre el subconjunto de estancias con informe, de prevalencia 7,0 %— no son
> directamente comparables con las del resto en esas dos columnas.
>
> Las cifras de la fusión son las de la ejecución cuyos pesos se conservan. Re-entrenar la misma
> configuración da AUROC 0,9275–0,9311 a 6h: el orden en que se registran los módulos altera el
> sorteo de inicialización y desplaza el resultado en milésimas, sin afectar a ninguna conclusión.

### Hallazgos clave

- **Las escalas clínicas apenas superan el azar como predictores anticipados.** Evaluadas en la
  hora de referencia, sin entrenamiento ni ajuste de umbral, sobre el mismo conjunto de prueba:
  qSOFA obtiene AUROC 0,5433 y SIRS 0,5172, con AUPRC en torno a la prevalencia. Son escalas
  reactivas, diseñadas para confirmar un deterioro ya presente.
- **La fusión multimodal supera a todo modelo individual**: +0,070 puntos de AUROC sobre el TFT
  y +0,064 sobre XGBoost a 6h, con intervalos de confianza por *bootstrap* emparejado (1.000
  réplicas) que no se acercan al cero. En sensibilidad clínica la ganancia es aún mayor: +0,115
  y +0,141.
- **Las tres modalidades contribuyen y ninguna sobra.** La ablación por eliminación de modalidad
  degrada el sistema en las tres métricas y en los dos horizontes, sin excepción: quitar el texto
  cuesta −0,046 de AUROC a 6h; quitar los vitales, −0,026 de AUROC pero −0,139 de AUPRC, la mayor
  pérdida de las tres; quitar la rama tabular, −0,019.
- **Comprimir la rama tabular anula su aportación.** Sin rama tabular, la fusión obtiene AUROC
  0,9118 a 6h; con la representación de 32 dimensiones de TabNet, 0,9145; con el vector de 99
  características sin comprimir, 0,9311. Un codificador subóptimo puede anular la contribución de
  una modalidad válida sin que la métrica global lo refleje.
- **El TFT es el mejor codificador unimodal en las métricas sensibles al desbalanceo.** Frente a
  XGBoost, el AUROC no separa —el intervalo de la diferencia cruza el cero en los dos
  horizontes—, pero la AUPRC sí, con claridad (+0,081 a 6h, +0,139 a 12h): la ventaja de
  modelizar la dinámica temporal es una ventaja sobre la clase minoritaria.
- **La ventaja frente a la línea de base clínica crece con el horizonte**: de +0,064 a +0,077
  puntos de AUROC y de +0,141 a +0,171 de sensibilidad clínica al pasar de 6h a 12h, porque
  XGBoost se degrada al alejarse del *onset* y la fusión no.
- **La atención inter-modal se reparte de forma equilibrada** (los seis pesos entre 0,42 y 0,59):
  ninguna modalidad domina. Esa observación es solo sugerente —los pesos describen cómo pondera
  el modelo, no qué aporta cada fuente—; es la ablación la que la convierte en evidencia.
- **Sin recalibrar, ningún modelo emite probabilidades utilizables**: los cinco sobreestiman el
  riesgo por efecto de la ponderación de clase, hasta el punto de que su Brier queda *por detrás*
  del de un predictor que siempre anunciara la prevalencia. Un reescalado de Platt ajustado sobre
  validación lo corrige —ECE de la fusión 0,1661 → 0,0075 a 6h— sin mover ni una diezmilésima de
  AUROC, AUPRC o Sens@Esp90, por ser una transformación monótona.

---

## Reproducibilidad

```bash
# 1. Acceso acreditado a MIMIC-IV v3.0 y descarga local en ~/mimic-iv-3.0/
# 2. Entorno
conda create -n sepsis-tfm python=3.10 && conda activate sepsis-tfm
pip install polars pandas numpy scikit-learn xgboost pytorch-tabnet \
            torch pytorch-forecasting transformers matplotlib jupyter
# 3. Ejecutar los cuadernos en orden (01 → 10)
jupyter notebook notebooks/
```

Los cuadernos asumen los datos crudos en `~/mimic-iv-3.0/{hosp,icu}/` y escriben artefactos
intermedios en `data/processed/` (no incluido en este repositorio).

---

## Metodología (resumen)

- **Etiquetado**: criterios Sepsis-3 —infección sospechada según Angus (antibiótico y cultivo en
  una ventana de 24 h) y ΔSOFA ≥ 2—. El inicio se fija en el instante más temprano en que se
  cumplen ambas condiciones.
- **Muestras de predicción**: una por estancia. Para los casos sépticos, el tiempo de referencia
  es *t*<sub>onset</sub> − *H*, con *H* ∈ {6, 12} h; para los controles, un instante aleatorio de
  la estancia. Se exigen al menos 12 h de historia previa, lo que restringe el sistema a la sepsis
  de aparición no inmediata.
- **Desbalanceo**: la prevalencia es del 22,97 % en la cohorte, pero ≈ 8 % en las muestras de
  predicción (8,3 % a 6h), que es la cifra relevante para interpretar la AUPRC. Pérdida ponderada
  (`pos_weight` ≈ 11) y AUROC/AUPRC como métricas primarias (nunca *accuracy*).
- **Exclusión de SOFA**: como SOFA ≥ 2 es parte del criterio de etiquetado, **las puntuaciones SOFA
  —la agregada y las seis por sistema orgánico— se excluyen del vector de características** para
  no codificar la propia etiqueta. Se conservan las mediciones fisiológicas y analíticas a partir
  de las cuales se calculan.
- **Particiones**: a nivel de paciente, de modo que ningún reingreso se reparte entre
  entrenamiento y evaluación.
- **Calibración**: la ponderación de clase infla las probabilidades, así que los cinco modelos
  incorporan como etapa final un reescalado de Platt ajustado **solo sobre validación**. El
  conjunto de prueba no interviene ni en esa etapa ni en ninguna otra decisión.
- **Ventana de entrada**: 24 h anteriores al tiempo de referencia, remuestreadas a 1 h para los
  vitales. El informe de radiología es el más reciente anterior al tiempo de referencia.
- **Datos faltantes**: vitales con *forward-fill* hasta 4 h; labs con *carry-forward* e
  indicadores de faltantes; estancias sin informe, con *embedding* nulo e indicador `has_note`.

---

## Cita

Si utilizas este trabajo, cítalo como:

> Calvino Balonero, F. (2026). *Predicción Temprana de Sepsis en UCI mediante Aprendizaje
> Profundo Multimodal con Fusión de Datos Clínicos Heterogéneos* [Trabajo de Fin de Máster,
> Máster en Inteligencia Artificial, Universidad Internacional de Valencia (VIU)].

El dataset debe citarse según las indicaciones de PhysioNet para MIMIC-IV v3.0.

## Licencia

El **código** de los cuadernos se publica bajo licencia MIT (ver [`LICENSE`](LICENSE)). Los
**datos** de MIMIC-IV **no** están cubiertos por esta licencia y no se incluyen ni redistribuyen.
