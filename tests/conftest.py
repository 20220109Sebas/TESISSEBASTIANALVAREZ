"""Separate self-contained tests from checks of archived research executions."""

import json
from pathlib import Path

import pytest


_REQUIREMENTS = Path(__file__).with_name("integration_requirements.json")
_INTEGRATION = json.loads(_REQUIREMENTS.read_text(encoding="utf-8"))
_INTEGRATION_MODULES = set(_INTEGRATION["files"])
_INTEGRATION_TESTS = set(_INTEGRATION["tests"])


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="Collect archived-execution checks; requires the original datasets and artifacts.",
    )


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--run-integration"):
        if collection_path.name in _INTEGRATION_MODULES:
            return True
    return None


def pytest_report_header(config):
    if config.getoption("--run-integration"):
        return "Including archived-execution checks (external research evidence required)."
    return (
        "Self-contained suite; archived-execution modules are not collected. "
        "See tests/README.md and --run-integration."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    requires_archive = pytest.mark.skip(reason="Requires archived research execution; use --run-integration.")
    for item in items:
        if item.nodeid.removeprefix("tests/") in _INTEGRATION_TESTS:
            item.add_marker(requires_archive)
