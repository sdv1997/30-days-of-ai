# Contexto del proyecto — léeme primero

> Este fichero te pone al día rápido si arrancas una sesión nueva (ej. desde un pod remoto). El estado real está en los `day_XX/README.md`, pero esto te da los 30 segundos de contexto.

## Quién soy yo (usuario)

- Físico con base sólida en ML/AI. Quiere profundizar, no aprender 101.
- Comunicación en **castellano**.
- Intereses laterales: física, historia, economía (pueden aparecer en días futuros del challenge).

## Qué es este proyecto

**10 días de IA** — challenge público en GitHub + X. Cada día = algo distinto relacionado con ML / IA, idealmente cerrable en una sesión.

Repo: https://github.com/sdv1997/30-days-of-ai

## Día 1 — cerrado

**Richter's Predictor** (DrivenData, predicción de daño en edificios tras terremoto Nepal 2015). Detalle completo en [day_01/README.md](day_01/README.md).

Resumen:
- **Resultado final: F1 0.7534 público, rank 55 / ~8.800 (top 0.6%).**
- 5 iteraciones documentadas (1 negativa con target encoding, el resto positivas).
- Pipeline ganador: ensemble simple (LightGBM + CatBoost beefier) con 56 features (38 originales + 18 derivadas: agregaciones por aldea y deltas edificio-vs-aldea).
- Todo corrió en CPU local (Day 1 fue tabular).

## Hardware — situación actual

- Hasta Día 1: solo CPU local.
- A partir de Día 2: **runpod community cloud con RTX A5000** (24 GB VRAM). Pod montado, VS Code Remote-SSH conectado. Pagas por hora (~$0.27/h), apagar cuando no se usa.
- Cuando una sesión nueva arranque dentro del pod, los ficheros viven en `/workspace/30-days-of-ai/`.

## Día 2 — cerrado

**Conser-vision** (DrivenData, clasificación de imágenes de cámaras trampa). Detalle completo en [day_02/README.md](day_02/README.md).

Resumen:
- **Resultado final: log-loss 0.8990 público, rank #18.**
- Pipeline ganador: MegaDetectorV5 → crop al animal → ConvNeXt V2 Base fine-tuned.
- CV site-disjoint (StratifiedGroupKFold por site) obligatorio — sin él el modelo aprende atajos del entorno.
- MegaDetector fue la mejora más grande: bajó el LB de 1.2758 → 0.8990 (-0.377).
- CV media 0.9479 con alta varianza entre folds (0.80 – 1.13) → el reto es generalización cross-site.
- Corrió en RTX A5000: ~48 min detección + ~2h entrenamiento.

## Cómo trabajamos

- **Solo ejecuta.** Una vez alineados sobre el qué, lanza código sin pedir permiso ni hacer preguntas de confirmación. Asume defaults razonables. Solo pregunta si algo es genuinamente bloqueante.
- **No subir CSVs** (ni dataset original ni predicciones) al repo — `.gitignore` ya los excluye. Datos crudos se descargan manualmente de la plataforma.
- **Honestidad sobre métricas:** experimentos negativos se cuentan (parte de la narrativa).
- **Cada día deja artefactos en GitHub:** script reproducible + README + plot/output. Otros casi nada.
- **CV ↔ leaderboard:** verificar que el gap CV→público sea pequeño y consistente. Si se ensancha, alerta de overfit al CV.

## Convenciones del repo

- `day_XX/` por día. Dentro: `README.md` + scripts mínimos + plots.
- `data/` gitignored (CSVs / imágenes raw).
- Submissions también gitignored (regenerables corriendo el pipeline).
- Permisos de Claude Code en `bypassPermissions` en este proyecto (`.claude/settings.local.json`, también gitignored).

## Stack

Python 3.12, pandas, NumPy, scikit-learn, LightGBM, CatBoost, PyTorch (en pod cuando toque NN/CV), matplotlib. Sin frameworks pesados.

## Día 3 — cerrado

**What's Up, Docs?** (DrivenData, generación de abstracts de papers de ciencias sociales). Detalle completo en [day_03/README.md](day_03/README.md).

Resumen:
- **Resultado final: ROUGE-2 0.1398 público, rank #9 / 495.**
- Modelo: Qwen2.5-7B-Instruct-AWQ (4-bit, ~4.4 GB), inferencia via vLLM en A5000.
- La mejora clave fue aumentar el contexto de 6000 → 24.000 chars (+0.014 ROUGE-2), no el cambio de modelo.
- `HF_HOME=/workspace/.cache/huggingface` obligatorio para no llenar el overlay raíz de 20 GB.

## Día 4 — EN PROGRESO

**BirdCLEF+ 2026** (Kaggle, identificar 234 especies en ventanas de 5s de soundscapes). Detalle completo y plan en [day_04/README.md](day_04/README.md).

Estado:
- Pipeline two-stage funcionando: teacher EfficientNet-B0 en clips → pseudo-label de soundscapes → fine-tune student → export ONNX (CPU, Kaggle 90 min).
- **Phase 1 (clips): AUC 0.9804. Phase 2 (soundscapes): AUC 0.7757.** Aún sin submission. Top del LB ~0.94.
- Métrica: ROC-AUC macro. Solo corrido fold 0.
- **Cuello de botella = brecha de dominio focal→soundscape**: el teacher clava clips limpios pero cae en soundscapes ruidosos. Phase 2 hace plateau en ep5 y sobreajusta después.
- **Plan próximo día (decidido): combinar (1) background-noise augmentation en Phase 1 + (2) backbone más fuerte `eca_nfnet_l0`.** Antes de relanzar: borrar `pseudo_fold0.npz` (pseudo-labels del teacher B0, hay que regenerarlos) y respaldar/borrar los `.pt`/`.onnx` (resume incompatible con arquitectura nueva).

## Lo que aprendí en Día 3 (lecciones transferibles)

- **En summarización, el contexto importa más que el tamaño del modelo.** Pasar de 6000 a 24.000 chars (+0.014) superó con creces cambiar de 3B a 7B (+0.0004). Truncar la conclusión del paper es perder la mitad de la información relevante.
- **AWQ 4-bit** reduce pesos de ~15 GB a ~4.4 GB con pérdida mínima en tareas generativas. Útil cuando el almacenamiento es el cuello de botella, no la calidad.
- **vLLM** con batching: 345 papers en ~30s en A5000. La latencia real es el warmup/compilación (~2 min), no la inferencia.
- **Gestión de caché HuggingFace en RunPod:** siempre redirigir con `HF_HOME=/workspace/.cache/huggingface`. El overlay raíz tiene solo 20 GB y `pip install vllm` ya consume ~12 GB de eso.

## Lo que aprendí en Día 2 (lecciones transferibles)

- **En camera trap classification, MegaDetector primero siempre.** Recortar al animal elimina el fondo site-específico y es la mejora más grande posible. Sin él el modelo aprende el entorno, no el animal.
- **CV site-disjoint es no negociable** cuando train y test tienen sites distintos. CV con sites mezclados es trampa.
- Pipeline two-stage (detector → clasificador) es el estándar en wildlife ML en producción. MegaDetector es open source y genérico — sirve para cualquier dataset de cámaras trampa.
- Alta varianza entre folds en CV site-disjoint es información real, no ruido: refleja que algunos entornos son más difíciles de generalizar que otros.
- Lanzar entrenamientos largos con `tmux`, no como background task de Claude Code (muere si se desconecta SSH).

## Lo que aprendí en Día 1 (lecciones transferibles)

- En tabular con cardinalidad alta: cast `geo_level_*_id` a `category` antes de tunear nada — diferencia +0.03 F1.
- LightGBM categórico nativo ya hace algo equivalente a target encoding internamente. TE explícito no aporta en esta familia de datos.
- Stacking con 2 modelos correlacionados no aporta sobre el promedio simple — necesita más diversidad (XGBoost + LGBM + CatBoost) para brillar.
- CV calibrada con 5-fold StratifiedKFold seed=42 y comparar todos los experimentos contra los mismos folds. Gap CV→público de -0.0008 (consistente).
