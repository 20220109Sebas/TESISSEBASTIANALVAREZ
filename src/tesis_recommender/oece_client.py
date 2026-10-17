from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = "https://contratacionesabiertas.oece.gob.pe/api/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def list_snapshots(source: str | None = None, year: int | None = None, paginate_by: int = 100) -> dict:
    params = {"page": 1, "paginateBy": paginate_by, "format": "json"}
    if source:
        params["source"] = source
    if year:
        params["year"] = year
    url = f"{API_ROOT}/files?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "tesis-recommender/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tesis-recommender/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
        return int(value) if value else None


def catalog_range(source: str, years: range, workers: int = 6) -> dict:
    snapshots = []
    for year in years:
        payload = list_snapshots(source, year)
        snapshots.extend(payload["results"])
    snapshots.sort(key=lambda item: (item["year"], item["month"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_content_length, item["files"]["json"]): item for item in snapshots}
        for future in as_completed(futures):
            item = futures[future]
            try:
                item["json_zip_bytes"] = future.result()
                item["head_error"] = None
            except Exception as error:  # catálogo parcial sigue siendo útil
                item["json_zip_bytes"] = None
                item["head_error"] = str(error)
    return {
        "cataloged_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "years": list(years),
        "snapshot_count": len(snapshots),
        "known_total_json_zip_bytes": sum(item["json_zip_bytes"] or 0 for item in snapshots),
        "snapshots": snapshots,
    }


def download_json_snapshot(source: str, year: int, month: int, output_dir: str | Path) -> dict[str, str | int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{API_ROOT}/file/{source}"
    archive_url = f"{base}/json/{year}/{month:02d}/"
    sha_url = f"{base}/sha/{year}/{month:02d}/"
    archive_path = output_dir / f"{year}-{month:02d}_{source}_json.zip"
    request = urllib.request.Request(sha_url, headers={"User-Agent": "tesis-recommender/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        expected_sha = response.read().decode("utf-8").strip().split()[0]
    existing_json = output_dir / f"{year}-{month:02d}_{source}.json"
    if existing_json.exists() and _sha256(existing_json) == expected_sha:
        return {
            "source": source, "year": year, "month": month, "archive_url": archive_url,
            "archive_path": str(archive_path), "json_path": str(existing_json), "sha256": expected_sha, "status": "cached",
            "archive_bytes": archive_path.stat().st_size if archive_path.exists() else None, "json_bytes": existing_json.stat().st_size,
        }
    request = urllib.request.Request(archive_url, headers={"User-Agent": "tesis-recommender/0.1"})
    temporary_path = archive_path.with_suffix(".zip.part")
    with urllib.request.urlopen(request, timeout=180) as response, temporary_path.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    temporary_path.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if len(members) != 1:
            raise ValueError(f"Se esperaba un JSON en el ZIP; encontrados: {members}")
        json_path = output_dir / Path(members[0]).name
        with archive.open(members[0]) as source_handle, json_path.open("wb") as target:
            shutil.copyfileobj(source_handle, target, length=1024 * 1024)
    actual_sha = _sha256(json_path)
    if actual_sha != expected_sha:
        raise ValueError(f"Checksum inválido: esperado {expected_sha}, obtenido {actual_sha}")
    return {
        "source": source, "year": year, "month": month, "archive_url": archive_url,
        "archive_path": str(archive_path), "json_path": str(json_path), "sha256": actual_sha, "status": "downloaded",
        "archive_bytes": archive_path.stat().st_size, "json_bytes": json_path.stat().st_size,
    }


def download_range(source: str, years: range, root_dir: str | Path, workers: int = 4) -> dict:
    root_dir = Path(root_dir)
    tasks = [(year, month) for year in years for month in range(1, 13)]
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_json_snapshot, source, year, month, root_dir / source / str(year) / f"{month:02d}"): (year, month)
            for year, month in tasks
        }
        for future in as_completed(futures):
            year, month = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"year": year, "month": month, "error": str(error)})
    results.sort(key=lambda item: (item["year"], item["month"]))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source, "years": list(years),
        "completed": len(results), "failed": len(errors), "snapshots": results, "errors": errors,
    }
    manifest_path = root_dir / f"manifest_{source}_{min(years)}_{max(years)}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
