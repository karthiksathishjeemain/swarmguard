"""Deterministic routing of local diagnostics, never proof of swarm misconduct."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from .local_guard import MODEL_ID, MODEL_REVISION, PROMPT_VERSION, digest
from .schema import validate_packet

REFERENCE_RELATIONS = {
    "quoted_prior_message", "explicit_source_message_id_reference",
    "explicit_request_accepted", "explicit_result_acknowledgment",
    "explicit_reliance", "explicit_rejection", "explicit_goal_change_agreement",
    "explicit_reference", "explicit_claim_adopted", "explicit_claim_rejected",
    "explicit_claim_adoption", "explicit_response_to_claim",
    "explicit_goal_replacement_agreement", "explicit_result_acceptance",
}


def route_episode(packet, signal, model, coverage=None, *, synthetic=False):
    """Keep a broad unsafe label separate from evidence of multi-agent causation.

    A cross-agent edge provides review context; it does not attribute the model's
    unsafe label to either endpoint. Even a complete packet yields a candidate,
    never an incident verdict or a declaration of alignment.
    """
    result = {
        "episode_id": packet.get("episode_id"), "packet_sha256": digest(packet),
        "route": "insufficient", "safety_signal": None, "reasons": [],
        "reference_edge_ids": [], "coverage_gaps": [],
        "implicated_agents": [], "human_validated": False,
        "swarm_incident_established": False,
        "next_action": "inspect_missing_evidence",
    }
    if not validate_packet(packet).valid:
        result["reasons"].append("invalid_packet")
        return result
    if (model.get("model_id") != MODEL_ID or model.get("revision") != MODEL_REVISION or
            model.get("revision_verification") != "fixed_pin_and_all_download_manifest_hashes_verified"):
        result["reasons"].append("unverified_model_checkpoint")
    if signal is None:
        result["reasons"].append("model_not_run")
    elif (signal.get("episode_id") != packet["episode_id"] or
          signal.get("packet_sha256") != digest(packet) or
          signal.get("prompt_version") != PROMPT_VERSION or
          signal.get("model_id") != MODEL_ID):
        result["reasons"].append("stale_or_unbound_model_result")
    elif (signal.get("status") != "classified" or
          signal.get("safety") not in {"safe", "unsafe"} or
          signal.get("input_truncated") or signal.get("generation_incomplete") or
          signal.get("errors")):
        result["reasons"].append("model_abstained_or_incomplete")
    if result["reasons"]:
        return result

    if not synthetic:
        result["coverage_gaps"].append("retrieved_window_is_not_a_complete_history")
        # The retrieval builder predates the sorted-key guard hash convention.
        builder_hash = hashlib.sha256(json.dumps(packet, ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()
        if not coverage or coverage.get("packet_sha256") != builder_hash:
            result["reasons"].append("missing_or_stale_retrieval_manifest")
            return result
        if coverage.get("source_window_truncated") or coverage.get("target_source_window_truncated"):
            result["coverage_gaps"].append("source_action_window_truncated")
        record_meta = coverage.get("records", {})
        if any(record_meta.get(r["id"], {}).get(flag) for r in packet["records"]
               for flag in ("builder_truncated", "packet_bound_truncated", "existing_truncation_marker")):
            result["coverage_gaps"].append("record_text_excerpted_or_truncated")
        if not any(r["kind"] == "instruction" for r in packet["records"]):
            result["coverage_gaps"].append("governing_authorization_not_established")

    records = {r["id"]: r for r in packet["records"]}
    edges = [e for e in packet["edges"] if
             e["basis"] in {"direct", "explicit_reference"} and
             e["relation"] in REFERENCE_RELATIONS and
             records[e["source"]]["agent_id"] and records[e["target"]]["agent_id"] and
             records[e["source"]]["agent_id"] != records[e["target"]]["agent_id"]]
    result["reference_edge_ids"] = [e["id"] for e in edges if e.get("id")]
    result["safety_signal"] = signal["safety"]
    if signal["safety"] == "unsafe":
        if edges:
            result["route"] = "needs_context_review" if result["coverage_gaps"] else "needs_review"
            result["reasons"] = ["unsafe_signal_with_cross_agent_reference_context",
                                 "reference_does_not_establish_adoption_or_joint_misconduct"]
            result["next_action"] = "review_source_actions_and_reference_meaning"
        else:
            result["route"] = "unattributed_safety_signal"
            result["reasons"] = ["no_verified_cross_agent_reference_for_swarm_review"]
            result["next_action"] = "review_actor_attribution_before_swarm_classification"
    elif result["coverage_gaps"]:
        result["reasons"] = ["no_unsafe_signal_in_incomplete_packet"]
    else:
        result["route"] = "no_signal"
        result["reasons"] = ["model_returned_safe_for_this_packet_only"]
        result["next_action"] = "retain_for_blinded_negative_control_review"
    return result


def summarize_routes(rows):
    return {"episodes": len(rows), "route_counts": dict(Counter(r["route"] for r in rows)),
            "human_validated_incidents": 0, "external_api_calls": 0,
            "episodes_with_coverage_gaps": sum(bool(r["coverage_gaps"]) for r in rows)}
