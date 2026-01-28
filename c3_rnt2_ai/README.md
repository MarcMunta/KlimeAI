# C3 + RNT-2 Local Stack

Este repo implementa un prototipo modular para una “IA 120B-like” local combinando:
- **C3**: pesos paginados y comprimidos con caché coherente.
- **RNT-2**: tokenización neuronal reversible con canal de escape exacto.
- **KV híbrido**: ventana exacta + KV cuantizado + memoria latente.
- **Agente**: herramientas, memoria persistente y aprendizaje continuo.

## Instalación
```bash
python -m venv .venv
. .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -e .
```

## CLI (Fase 0)
```bash
python -m c3rnt2.cli doctor
python -m c3rnt2.cli demo_tokenizer
python -m c3rnt2.cli demo_agent
python -m c3rnt2.cli demo_core_generate
```

## Tokenizador RNT-2 (Fase 1)
Entrenamiento MVP (codebook por frecuencia de bloques):
```bash
python -m c3rnt2.tokenizer.rnt2_train --corpus data/corpora --output data/runs/rnt2_dev.pt
```

## Core Transformer + KV híbrido (Fase 2)
El core usa un transformer pequeño con tokens byte-level para el demo. El KV híbrido registra ventana exacta y memoria latente.

## C3 Runtime (Fase 3)
El runtime usa tiles comprimidos y caché con LRU + estabilidad. En el MVP, la descompresión ocurre en CPU/torch.

## Agente (Fase 5)
El demo crea un repo mini, abre docs (best effort), edita un bug y ejecuta tests.
```bash
python -m c3rnt2.cli demo_agent
```

## Benchmarks
```bash
python -m c3rnt2.training.eval
```

## Configuración
`config/settings.yaml` define perfiles (`dev_small`, `core_only`, `c3_paged`, `agent`).

## Notas
- Este MVP prioriza arquitectura, métricas y exactitud. Rendimiento se optimiza en fases siguientes.
- El repo está diseñado para enchufar un backend HF más grande en el futuro.
