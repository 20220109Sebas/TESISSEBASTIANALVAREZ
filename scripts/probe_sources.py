"""Comprobación de disponibilidad; no descarga datasets masivos."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCES = [
    "https://contratacionesabiertas.oece.gob.pe/api",
    "https://contratacionesabiertas.oece.gob.pe/api/v1/files?page=1&paginateBy=1&format=json",
    "https://www.gob.pe/52005-acceder-al-portal-de-contrataciones-abiertas-de-la-compra-publica-ocds",
]


def main() -> None:
    results = []
    for url in SOURCES:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tesis-recommender/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                results.append({"url": url, "status": response.status, "content_type": response.headers.get("Content-Type")})
        except (urllib.error.URLError, TimeoutError) as error:
            results.append({"url": url, "status": None, "error": str(error)})
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "sources": results}
    output = Path("artifacts/source_probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
