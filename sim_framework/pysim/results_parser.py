"""Parse cocotb results.xml + the framework's sim_metrics.json sideband file."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestOutcome:
    name: str
    passed: bool
    failure_message: Optional[str] = None


def parse_results_xml(path: Path) -> list[TestOutcome]:
    """Parse a cocotb-emitted JUnit-style results.xml into a flat outcome list."""
    if not path.exists():
        return []
    tree = ET.parse(path)
    out: list[TestOutcome] = []
    for ts in tree.findall(".//testsuite"):
        for tc in ts.findall("testcase"):
            failure = tc.find("failure")
            out.append(TestOutcome(
                name=f"{ts.get('name', '?')}::{tc.get('name', '?')}",
                passed=(failure is None),
                failure_message=(failure.text if failure is not None else None),
            ))
    return out


def parse_metrics_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)
