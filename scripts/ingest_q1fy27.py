"""CLI: ingest the three Q1 FY27 seed documents and print the manifest."""

from __future__ import annotations

import json
import logging
import sys

sys.path.insert(0, ".")

from app.config import get_settings
from app.ingestion.pipeline import ingest_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    records = ingest_all()
    settings = get_settings()
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    print(f"Ingested {len(records)} documents. Registry: {settings.registry_path}")
    print(f"Manifest: {settings.manifest_path}")
    for entry in manifest:
        print(
            f"  - {entry['filename']} | {entry['document_type']} | "
            f"{entry['period']} | {entry['reporting_basis']} | "
            f"{entry['file_size']} bytes | sha256:{entry['sha256'][:16]}..."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
