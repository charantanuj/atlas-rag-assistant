from __future__ import annotations

import json
from pathlib import Path

from rag_assistant.evaluation.models import EvaluationReport


def write_json_report(report: EvaluationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path

