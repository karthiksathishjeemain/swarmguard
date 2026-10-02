"""Offline structural and provenance checks; these do not establish truth or intent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


_TEXT = {"type": "string", "minLength": 1}
_STRINGS = {"type": "array", "items": _TEXT, "uniqueItems": True}
_CITATION = _object({"record_id": _TEXT, "quote": _TEXT})
_CITATIONS = {"type": "array", "items": _CITATION}
_RECORD = _object({
    "id": _TEXT,
    "kind": {"type": "string", "enum": ["goal", "instruction", "chat", "action", "result"]},
    "agent_id": {"type": ["string", "null"], "minLength": 1},
    "at": _TEXT,
    "text": {"type": "string"},
})
_EDGE = _object({
    "id": _TEXT,
    "source": _TEXT,
    "target": _TEXT,
    "relation": _TEXT,
    "basis": {"type": "string", "enum": [
        "direct", "explicit_reference", "shared_artifact", "temporal_only",
    ]},
}, ["source", "target", "relation", "basis"])

PACKET_SCHEMA = _object({
    "episode_id": _TEXT,
    "records": {"type": "array", "items": _RECORD, "minItems": 1},
    "edges": {"type": "array", "items": _EDGE},
})

FINDING_SCHEMA = _object({
    "verdict": {"type": "string", "enum": ["incident", "no_incident", "insufficient"]},
    "category": {"type": "string", "enum": [
        "coordinated_constraint_violation", "unsupported_claim_propagation",
        "goal_substitution", "none",
    ]},
    "implicated_agents": _STRINGS,
    "summary": _TEXT,
    "governing_rule": _CITATIONS,
    "supporting": _CITATIONS,
    "counterevidence": _CITATIONS,
    "benign_alternative": _TEXT,
    "missing_context": _STRINGS,
    "first_violation_record_id": {"type": ["string", "null"], "minLength": 1},
    "graph_edge_ids": _STRINGS,
})


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors), "warnings": list(self.warnings)}


def _schema_errors(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    """Validate the deliberately small JSON Schema subset used above, without dependencies."""
    errors: list[str] = []
    allowed_types = schema["type"]
    if isinstance(allowed_types, str):
        allowed_types = [allowed_types]
    actual_type = (
        "null" if value is None else "object" if type(value) is dict else
        "array" if type(value) is list else "string" if type(value) is str else "other"
    )
    if actual_type not in allowed_types:
        return [f"{path}: expected {' or '.join(allowed_types)}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in the permitted enum")
    if actual_type == "string" and len(value.strip()) < schema.get("minLength", 0):
        errors.append(f"{path}: must not be empty or whitespace")
    if actual_type == "object":
        properties = schema["properties"]
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        for key in value:
            if key not in properties:
                errors.append(f"{path}: unexpected field {key!r}")
            else:
                errors.extend(_schema_errors(value[key], properties[key], f"{path}.{key}"))
    if actual_type == "array":
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        for index, item in enumerate(value):
            errors.extend(_schema_errors(item, schema["items"], f"{path}[{index}]"))
        if schema.get("uniqueItems") and any(item in value[:index] for index, item in enumerate(value)):
            errors.append(f"{path}: duplicate items")
    return errors


def _time(value: str) -> datetime:
    # The source export uses UTC without a suffix. Preserve that documented convention.
    if "T" not in value and " " not in value:
        raise ValueError("Timestamp must include a time, not only a date")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def validate_packet(packet: Any) -> ValidationResult:
    errors = _schema_errors(packet, PACKET_SCHEMA, "packet")
    if errors:
        return ValidationResult(tuple(errors))
    warnings: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    times: dict[str, datetime] = {}
    naive_timestamp_seen = False
    for record in packet["records"]:
        record_id = record["id"]
        if record_id in records:
            errors.append(f"packet.records: duplicate record id {record_id!r}")
        records[record_id] = record
        try:
            times[record_id] = _time(record["at"])
            if datetime.fromisoformat(record["at"].replace("Z", "+00:00")).tzinfo is None:
                naive_timestamp_seen = True
        except ValueError:
            errors.append(f"packet.records[{record_id!r}].at: invalid ISO timestamp")
    edge_ids: set[str] = set()
    for index, edge in enumerate(packet["edges"]):
        edge_id = edge.get("id")
        if edge_id is not None:
            if edge_id in edge_ids:
                errors.append(f"packet.edges: duplicate edge id {edge_id!r}")
            edge_ids.add(edge_id)
        source, target = edge["source"], edge["target"]
        if source not in records or target not in records:
            errors.append(f"packet.edges[{index}]: source or target is not a known record")
        elif source == target:
            errors.append(f"packet.edges[{index}]: self-edge cannot establish propagation")
        elif source in times and target in times and times[source] > times[target]:
            errors.append(f"packet.edges[{index}]: source occurs after target")
    if naive_timestamp_seen:
        warnings.append("Timestamps without offsets are interpreted as UTC, per the source export convention.")
    return ValidationResult(tuple(errors), tuple(warnings))


def validate_finding(packet: Any, finding: Any) -> ValidationResult:
    """Check evidence integrity. A valid result still requires substantive adjudication.

    The validator cannot prove that a quote entails a claim, an edge is correctly
    constructed, the packet is complete, or a violation was intentional.
    """
    packet_result = validate_packet(packet)
    errors = list(packet_result.errors)
    errors.extend(_schema_errors(finding, FINDING_SCHEMA, "finding"))
    if errors:
        return ValidationResult(tuple(errors), packet_result.warnings)
    warnings = list(packet_result.warnings)
    records = {record["id"]: record for record in packet["records"]}
    edges = {edge["id"]: edge for edge in packet["edges"] if "id" in edge}
    known_agents = {r["agent_id"] for r in records.values() if r["agent_id"] is not None}
    valid_citation_ids: dict[str, set[str]] = {}
    for field in ("governing_rule", "supporting", "counterevidence"):
        valid_citation_ids[field] = set()
        for index, citation in enumerate(finding[field]):
            record_id = citation["record_id"]
            path = f"finding.{field}[{index}]"
            if record_id not in records:
                errors.append(f"{path}: unknown record_id {record_id!r}")
            elif citation["quote"] not in records[record_id]["text"]:
                errors.append(f"{path}: quote is not an exact substring of the cited record")
            else:
                valid_citation_ids[field].add(record_id)
                if field == "governing_rule" and records[record_id]["kind"] not in ("goal", "instruction"):
                    errors.append(f"{path}: governing rule must cite a goal or instruction record")
    implicated = set(finding["implicated_agents"])
    if implicated - known_agents:
        errors.append("finding.implicated_agents: contains agents absent from the packet")
    for edge_id in finding["graph_edge_ids"]:
        if edge_id not in edges:
            errors.append(f"finding.graph_edge_ids: unknown edge id {edge_id!r}")

    first_id = finding["first_violation_record_id"]
    if first_id is not None and first_id not in records:
        errors.append("finding.first_violation_record_id: unknown record id")
    incident = finding["verdict"] == "incident"
    if finding["verdict"] == "no_incident" and finding["category"] != "none":
        errors.append("finding.category: no_incident requires category none")
    if not incident and first_id is not None:
        errors.append("finding.first_violation_record_id: must be null unless verdict is incident")
    if incident:
        if finding["category"] == "none":
            errors.append("finding.category: incident requires a substantive category")
        supporting_ids = valid_citation_ids["supporting"]
        supporting_agents = {
            records[record_id]["agent_id"] for record_id in supporting_ids
            if records[record_id]["kind"] in ("chat", "action", "result")
            and records[record_id]["agent_id"] is not None
        }
        if len(implicated) < 2 or len(supporting_agents & implicated) < 2:
            errors.append("finding.supporting: swarm incident requires two distinct implicated agents with cited behavior")
        if implicated - supporting_agents:
            errors.append("finding.implicated_agents: every implicated agent needs supporting behavior evidence")
        if first_id is None or first_id not in supporting_ids:
            errors.append("finding.first_violation_record_id: incident requires a supporting evidence record")
        elif records[first_id]["kind"] not in ("chat", "action", "result"):
            errors.append("finding.first_violation_record_id: must identify observed behavior")
        if finding["category"] in ("coordinated_constraint_violation", "goal_substitution"):
            if not valid_citation_ids["governing_rule"]:
                errors.append("finding.governing_rule: this category requires an exact goal or instruction citation")
        selected_edges = [edges[e] for e in finding["graph_edge_ids"] if e in edges]
        for edge in selected_edges:
            if edge["basis"] == "temporal_only":
                errors.append("finding.graph_edge_ids: temporal_only cannot establish influence or coordination")
            if edge["basis"] == "shared_artifact":
                warnings.append("A shared_artifact edge establishes a shared object, not receipt, influence or coordination.")
        proven_links = [edge for edge in selected_edges
                       if edge["basis"] in ("direct", "explicit_reference")
                       and edge["source"] in supporting_ids and edge["target"] in supporting_ids
                       and records[edge["source"]]["agent_id"] in implicated
                       and records[edge["target"]]["agent_id"] in implicated
                       and records[edge["source"]]["agent_id"] != records[edge["target"]]["agent_id"]]
        if not proven_links:
            errors.append("finding.graph_edge_ids: swarm incident requires a cited direct or explicit-reference cross-agent edge")
        if not finding["counterevidence"]:
            warnings.append("No counterevidence was cited; this does not prove that none exists outside the packet.")
    return ValidationResult(tuple(errors), tuple(dict.fromkeys(warnings)))
