"""Register a bounded, identity-checked public-SEACE browser recovery batch.

This is an offline importer, not a browser driver or a change to extraction rules.
Observation records are automatic source reviews, never human or expert annotations.
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_observation(observation, selected, tender, limits):
    if observation["opportunity_id"] != selected["opportunity_id"]:
        raise ValueError("Different process")
    if observation["convocation_id"] != str(tender["id"]) or observation["nomenclature"] != tender["title"]:
        raise ValueError("Public ficha differs from OCDS process identity")
    expected_code = parse_qs(urlparse(selected["url"]).query)["fileCode"][0]
    if observation["file_code"] != expected_code:
        raise ValueError("Public download control points to a different document")
    ficha = urlparse(observation["ficha_url"])
    params = parse_qs(ficha.query)
    if (ficha.scheme != "https" or ficha.netloc != "prod2.seace.gob.pe"
        or ficha.path != "/seacebus-uiwd-pub/fichaSeleccion/fichaSeleccion.xhtml"
        or set(params) != {"idHistorial", "ptoRetorno"}
        or params["ptoRetorno"] != ["listaHistorialContrataciones"]
        or not re.fullmatch(r"[a-f0-9-]{36}", params["idHistorial"][0])):
        raise ValueError("Unexpected public ficha URL")
    if observation["document_title_displayed"] != selected["title"]:
        raise ValueError("Different document role")
    declared = datetime.fromisoformat(selected["date_published_declared"]).strftime("%d/%m/%Y %H:%M")
    if observation["document_date_displayed"] != declared:
        raise ValueError("Document date differs at UI minute precision")
    status = observation["status"]
    over = observation["size_kb_displayed"] * 1024 > limits["max_bytes_per_document"]
    if status not in {"browser_download_received", "not_downloaded_displayed_size_over_limit", "browser_download_event_timeout"}:
        raise ValueError("Unsupported observation status")
    if over != (status == "not_downloaded_displayed_size_over_limit"):
        raise ValueError("Observation violates the frozen size limit")
    if (status == "browser_download_received") != bool(observation.get("download_path")):
        raise ValueError("Download evidence and saved path disagree")


def protected_outputs():
    paths = {}
    annual_dir = ROOT / "artifacts/experiment_annual_v1"
    annual = json.loads((annual_dir / "manifest.json").read_text())
    for field in ("ranking_sha256", "component_sha256"):
        paths.update({str(annual_dir / p): h for p, h in annual[field].items()})
    paths[str(annual_dir / "candidate_order.json")] = annual["candidate_order_sha256"]
    paths[str(annual_dir / "results.json")] = annual["results_sha256"]
    directory = ROOT / "artifacts/extraction_commitment_v1"
    extraction = json.loads((directory / "manifest.json").read_text())
    paths.update({str(directory / p): h for p, h in extraction["output_sha256"].items()})
    return paths


def verify_hashes(paths):
    for path, expected in paths.items():
        if sha(path) != expected:
            raise ValueError("Protected input changed: " + str(path))


def register(observations_path, output):
    if output.exists():
        raise ValueError("Use a new access directory")
    plan_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    manifest_path = plan_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_hashes({str(plan_path.parent / p): h for p, h in manifest["output_sha256"].items()})
    plan = json.loads(plan_path.read_text())
    check_code(plan)
    batch = plan["batches"][0]
    verify_hashes({batch["path"]: batch["sha256"]})
    fixed_batch = json.loads(Path(batch["path"]).read_text())
    limits = fixed_batch["limits"]
    observations = json.loads(observations_path.read_text())["observations"]
    selected = plan["pending_sources"][:batch["sources"]]
    if [o["opportunity_id"] for o in observations] != [s["opportunity_id"] for s in selected]:
        raise ValueError("Observations must follow the complete fixed first batch")
    inputs = {str(observations_path): sha(observations_path), str(plan_path): sha(plan_path),
              str(manifest_path): sha(manifest_path), batch["path"]: batch["sha256"]}
    inputs.update(plan["frozen_rule_and_runner_sha256"])
    inputs.update(protected_outputs())
    ready = []
    for o, s in zip(observations, selected):
        verify_hashes({s["current_record_file_path"]: s["current_record_sha256"]})
        inputs[s["current_record_file_path"]] = s["current_record_sha256"]
        record = json.loads(Path(s["current_record_file_path"]).read_text())["records"][0]
        if record["compiledRelease"]["ocid"] != s["opportunity_id"]:
            raise ValueError("Saved record has different OCID")
        validate_observation(o, s, record["compiledRelease"]["tender"], limits)
        row = dict(s, browser_observation=o, retrieval_transport="public_seace_browser_download",
                   original_document_replaced=False, human_annotation=False, http_status=None,
                   historical_content_proven=False, ranking_eligible=False,
                   status=o["status"], absence_of_requirements_proven=False)
        if o["status"] == "browser_download_received":
            path = Path(o["download_path"])
            size = path.stat().st_size
            if size > limits["max_bytes_per_document"]:
                raise ValueError("Downloaded bytes exceed frozen limit")
            with path.open("rb") as stream:
                if not stream.read(5) == b"%PDF-":
                    raise ValueError("Downloaded file is not a PDF")
            digest = sha(path)
            inputs[str(path)] = digest
            row.update(status="downloaded_pdf", sha256=digest, bytes=size,
                       file_path=str(output / "pdfs" / (digest + ".pdf")))
        ready.append(row)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    (output / "pdfs").mkdir()
    for r in ready:
        if r["status"] == "downloaded_pdf":
            shutil.copyfile(r["browser_observation"]["download_path"], r["file_path"])
            verify_hashes({r["file_path"]: r["sha256"]})
    access_plan = dict(limits=limits, pending_sources=selected,
                       purpose="separate_substitute_development_browser_recovery", ranking_eligible=False)
    dump(output / "access_plan.json", access_plan)
    dump(output / "access_results.json", ready)
    dump(output / "manifest.json", {
        "access_plan_path": str(output / "access_plan.json"),
        "access_plan_sha256": sha(output / "access_plan.json"),
        "input_sha256": inputs, "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    check_code(plan)
    verify_hashes(inputs)
    return {"fichas_verified": len(ready), "pdfs_saved": sum(r["status"] == "downloaded_pdf" for r in ready)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.observations.resolve(), args.output.resolve())))
