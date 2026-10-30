# Evaluation strategy

The deterministic evaluation layer runs without an LLM judge and is suitable for CI.

## Metrics

- Retrieval: Precision@k, Recall@k, reciprocal rank, nDCG@k
- Citations: precision, completeness, exact document/location accuracy
- Generation: abstention accuracy and factual-sentence citation coverage
- Guardrails: attack block rate and benign false-positive rate

## Initial quality gates

- Recall@k >= 0.80
- MRR >= 0.75
- Citation precision = 1.00
- Citation completeness >= 0.90
- Abstention accuracy >= 0.90
- Attack block rate >= 0.95
- Guardrail false-positive rate <= 0.05

The example JSONL file is only a schema illustration. It is not ground truth. Real cases must be created by reviewing extracted source pages and recording the exact relevant chunk IDs.

RAGAS or another model-graded layer can be added after the deterministic gates pass. Model-graded scores must not replace citation-integrity checks.

