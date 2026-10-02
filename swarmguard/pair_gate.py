"""Fail-closed provenance gate for supplied result-to-claim NLI pair mappings.

Mappings must come from a reviewer or instrumented provenance, never from a model
guess. This module checks their internal consistency; it cannot authenticate the
reviewer, discover omitted records, prove completeness, or establish truth.

The history inventory is an explicit attestation about the same actor, artifact
and version through the claim. Every known observation and retry in that scope
must be supplied. Unknown completeness or unresolved retries makes the pair
ineligible. The gate does not infer this inventory from text.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any

from .schema import validate_packet


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_TEXT = {"type": "string", "minLength": 1}
_MAPPING = _object({
    "record_id": _TEXT, "quote": _TEXT, "subject_actor_id": _TEXT,
    "artifact_id": _TEXT, "artifact_version": _TEXT,
})
PAIR_DESCRIPTOR_SCHEMA = _object({
    "schema_version": {"type": "integer", "enum": [1]},
    "pair_id": _TEXT,
    "episode_id": _TEXT,
    "packet_sha256": _TEXT,
    "verification": _object({
        "method": {"type": "string", "enum": ["reviewer_verified", "instrumented"]},
        "source_reference": _TEXT,
        "verified_at": _TEXT,
    }),
    "result": _MAPPING,
    "claim": _MAPPING,
    "history": _object({
        "complete_through_claim": {"type": "boolean", "enum": [True]},
        "latest_observation_record_id": _TEXT,
        "observations": {"type": "array", "items": _MAPPING, "minItems": 1},
        "retries": {"type": "array", "items": _object({
            "action": _MAPPING,
            "resolved_by_record_id": {"type": ["string", "null"], "minLength": 1},
        })},
        "unresolved_retry": {"type": "boolean", "enum": [False]},
    }),
})


def packet_sha256(packet):
    """Canonical hash shared with the API runner; array order is significant."""
    raw = json.dumps(packet, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _descriptor_errors(value, schema, path="pair"):
    kinds = {dict: "object", list: "array", str: "string", int: "integer",
             bool: "boolean", type(None): "null"}
    actual = kinds.get(type(value))
    expected = schema["type"]
    expected = [expected] if isinstance(expected, str) else expected
    if actual not in expected:
        return [path + ": invalid_type"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(path + ": unsupported_or_uncertain_value")
    if actual == "string" and len(value.strip()) < schema.get("minLength", 0):
        errors.append(path + ": empty_value")
    if actual == "object":
        props = schema["properties"]
        for key in schema["required"]:
            if key not in value:
                errors.append(path + "." + key + ": missing_field")
        for key in value:
            if key not in props:
                errors.append(path + ": unexpected_field")
            else:
                errors.extend(_descriptor_errors(value[key], props[key], path + "." + key))
    elif actual == "array":
        if len(value) < schema.get("minItems", 0):
            errors.append(path + ": incomplete_inventory")
        for index, item in enumerate(value):
            errors.extend(_descriptor_errors(item, schema["items"], f"{path}[{index}]"))
    return errors


def _zoned(value):
    if "T" not in value and " " not in value:
        raise ValueError("timestamp_requires_time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp_requires_timezone")
    return parsed


@dataclass(frozen=True)
class PairGateResult:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    pair: dict[str, Any] | None = None

    @property
    def eligible(self):
        return not self.errors and self.pair is not None

    @property
    def status(self):
        return "eligible_for_nli" if self.eligible else "insufficient"

    def to_dict(self, *, include_text=False):
        pair = dict(self.pair) if self.pair is not None else None
        if pair is not None and not include_text:
            pair.pop("premise", None)
            pair.pop("hypothesis", None)
        return {"status": self.status, "errors": list(self.errors),
                "warnings": list(self.warnings), "pair": pair}


def gate_pair(packet, descriptor):
    """Return NLI eligibility only; a passing pair is not an incident verdict.

    `result`, `claim`, and each history mapping contain exact source quotations
    plus verified subject/artifact/version metadata. No semantic score is accepted
    as an input, and even a passing NLI contradiction needs separate adjudication.
    """
    packet_validation = validate_packet(packet)
    errors = []
    if not packet_validation.valid:
        errors.append("invalid_packet_schema_or_provenance")
    errors.extend(_descriptor_errors(descriptor, PAIR_DESCRIPTOR_SCHEMA))
    if errors:
        return PairGateResult(tuple(errors))
    if descriptor["episode_id"] != packet["episode_id"]:
        errors.append("episode_id_mismatch")
    digest = packet_sha256(packet)
    if descriptor["packet_sha256"] != digest:
        errors.append("packet_hash_mismatch")
    records = {record["id"]: record for record in packet["records"]}
    times = {}
    # Require explicit offsets even though the general source schema permits its
    # documented UTC convention. Pair descriptors cannot silently assume zones.
    for record in packet["records"]:
        try:
            times[record["id"]] = _zoned(record["at"])
        except (ValueError, TypeError):
            errors.append("packet_timestamp_requires_explicit_timezone")
    try:
        verified_at = _zoned(descriptor["verification"]["verified_at"])
    except (ValueError, TypeError):
        errors.append("verification_timestamp_requires_explicit_timezone")
        verified_at = None

    claim = descriptor["claim"]
    result = descriptor["result"]
    identity = tuple(claim[key] for key in ("subject_actor_id", "artifact_id", "artifact_version"))

    def check_mapping(mapping, kind, path):
        for field in ("subject_actor_id", "artifact_id", "artifact_version"):
            if mapping[field].strip().casefold() in {
                    "unknown", "unspecified", "uncertain", "n/a", "na", "none", "null", "?", "*"}:
                errors.append(path + ": unresolved_identity_metadata")
        if mapping["artifact_version"].strip().casefold() in {"latest", "current"}:
            errors.append(path + ": artifact_version_is_not_immutable")
        record = records.get(mapping["record_id"])
        if record is None:
            errors.append(path + ": unknown_record")
            return None
        if record["kind"] != kind:
            errors.append(path + ": incorrect_record_kind")
        if mapping["quote"] not in record["text"]:
            errors.append(path + ": quote_not_exact_substring")
        if record["agent_id"] is None or record["agent_id"] != mapping["subject_actor_id"]:
            errors.append(path + ": source_actor_subject_mismatch")
        mapped_identity = tuple(mapping[key] for key in ("subject_actor_id", "artifact_id", "artifact_version"))
        if mapped_identity != identity:
            errors.append(path + ": actor_artifact_or_version_mismatch")
        return record

    claim_record = check_mapping(claim, "chat", "claim")
    result_record = check_mapping(result, "result", "result")
    if (claim_record is not None and result_record is not None
            and result_record["agent_id"] != claim_record["agent_id"]):
        errors.append("result_and_claim_actor_mismatch")
    claim_time = times.get(claim["record_id"])
    result_time = times.get(result["record_id"])
    if claim_time is not None and result_time is not None and result_time > claim_time:
        errors.append("result_occurs_after_claim")
    if claim_time is not None and verified_at is not None and verified_at < claim_time:
        errors.append("verification_predates_claim")

    history = descriptor["history"]
    observations = {}
    for index, observation in enumerate(history["observations"]):
        check_mapping(observation, "result", f"observation[{index}]")
        record_id = observation["record_id"]
        if record_id in observations:
            errors.append("duplicate_observation_record")
        observations[record_id] = observation
        if claim_time is not None and record_id in times and times[record_id] > claim_time:
            errors.append("observation_occurs_after_claim")
    if result != observations.get(result["record_id"]):
        errors.append("selected_result_missing_or_inconsistent_in_history")
    if history["latest_observation_record_id"] != result["record_id"]:
        errors.append("selected_result_is_not_declared_latest_observation")
    known_observation_times = {record_id: times[record_id] for record_id in observations if record_id in times}
    if len(known_observation_times) == len(observations) and known_observation_times:
        latest_time = max(known_observation_times.values())
        latest_ids = {record_id for record_id, at in known_observation_times.items() if at == latest_time}
        if latest_ids != {result["record_id"]}:
            errors.append("selected_result_is_not_unique_latest_observation")

    retry_ids = set()
    for index, retry in enumerate(history["retries"]):
        action = retry["action"]
        check_mapping(action, "action", f"retry[{index}]")
        action_id = action["record_id"]
        if action_id in retry_ids:
            errors.append("duplicate_retry_record")
        retry_ids.add(action_id)
        resolution_id = retry["resolved_by_record_id"]
        if resolution_id is None:
            errors.append("unresolved_retry")
        elif resolution_id not in observations:
            errors.append("retry_resolution_not_in_observation_inventory")
        elif resolution_id in times and action_id in times and times[resolution_id] <= times[action_id]:
            errors.append("retry_resolution_not_strictly_after_action")
        if claim_time is not None and action_id in times and times[action_id] > claim_time:
            errors.append("retry_occurs_after_claim")
    if errors:
        return PairGateResult(tuple(dict.fromkeys(errors)))
    warnings = (
        "Supplied reviewer/instrumentation attestations are checked for consistency, not authenticated; omitted evidence cannot be detected here.",
        "Eligibility is not evidence of truth, contradiction, misalignment, or cross-agent reliance.",
    )
    pair = {
        "pair_id": descriptor["pair_id"], "episode_id": packet["episode_id"],
        "packet_sha256": digest, "premise_id": result["record_id"],
        "hypothesis_id": claim["record_id"], "premise": result["quote"],
        "hypothesis": claim["quote"], "subject_actor_id": identity[0],
        "artifact_id": identity[1], "artifact_version": identity[2],
        "verification_method": descriptor["verification"]["method"],
        "source_reference": descriptor["verification"]["source_reference"],
    }
    return PairGateResult(warnings=warnings, pair=pair)
