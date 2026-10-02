"""Bounded Responses API runner; input, findings, cache and budget ledger stay on VM.

No automatic retries: a timeout may still be billable. Only aggregate diagnostics
leave this process. Offline planning performs no network request or credential read.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
import urllib.error
import urllib.request

from .prompts import build_investigator_messages, build_skeptic_messages
from .schema import FINDING_SCHEMA, validate_finding, validate_packet

API_URL = "https://api.openai.com/v1/responses"
VM_ROOT = Path("/home/azureuser/ai-swarm")
KEY_PATH = VM_ROOT / "credentials/openai_api_key"


def validate_config(config):
    model, limits, pricing = (config[k] for k in ("model", "limits", "pricing"))
    for value in (limits["budget_usd"], pricing["input_ceiling_per_million"],
                  pricing["output_per_million"]):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid_budget_or_pricing")
    if (model["name"] != "gpt-6-astra" or
            model["reasoning_effort"] not in {"low", "medium", "high", "xhigh", "max"}):
        raise ValueError("unsupported_model_configuration")
    for value, minimum, maximum in (
        (model["max_output_tokens"], 1, 12000), (model["timeout_seconds"], 1, 600),
        (limits["max_episodes"], 1, 20), (limits["max_request_bytes"], 1024, 128000),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("invalid_pilot_limit")
    if limits["budget_usd"] > 100:
        raise ValueError("pilot_budget_exceeds_100_usd")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def wire_schema(value):
    # The local validator enforces these constraints. Keep the provider schema
    # within the supported strict-output subset and reject bad output locally.
    if isinstance(value, dict):
        return {k: wire_schema(v) for k, v in value.items()
                if k not in {"uniqueItems", "minLength"}}
    if isinstance(value, list):
        return [wire_schema(v) for v in value]
    return value


def request_body(messages, config):
    model = config["model"]
    body = {
        "model": model["name"], "input": messages, "store": False,
        "service_tier": "default", "truncation": "disabled",
        "reasoning": {"effort": model["reasoning_effort"]},
        "max_output_tokens": model["max_output_tokens"],
        "text": {"format": {"type": "json_schema", "name": "swarm_finding",
                            "strict": True, "schema": wire_schema(FINDING_SCHEMA)}},
    }
    if len(canonical(body)) > config["limits"]["max_request_bytes"]:
        raise ValueError("request_size_limit")
    return body


def reservation(body, config):
    # UTF-8 byte count plus framing allowance deliberately overestimates text
    # tokenization for planning. This is a local spend guard, not a billing quote.
    input_bound = len(canonical(body)) + 4096
    return round((input_bound * config["pricing"]["input_ceiling_per_million"]
                  + body["max_output_tokens"] * config["pricing"]["output_per_million"])
                 / 1_000_000, 6)


def save_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as out:
        os.fchmod(out.fileno(), 0o600)
        out.write(canonical(value) + b"\n")
        out.flush()
        os.fsync(out.fileno())
    staged.replace(path)


def read_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if not KEY_PATH.is_file() or KEY_PATH.is_symlink():
        raise RuntimeError("api_authentication_required")
    mode = KEY_PATH.stat()
    if stat.S_IMODE(mode.st_mode) & 0o077 or mode.st_uid != os.getuid():
        raise RuntimeError("credential_permissions_must_be_0600")
    key = KEY_PATH.read_text().strip()
    if not key:
        raise RuntimeError("api_authentication_required")
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call_provider(body, key, timeout):
    # Fixed TLS endpoint, no proxy inherited from the environment, no tools,
    # no external URLs derived from corpus contents, no automatic HTTP retries.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(API_URL, data=canonical(body), method="POST",
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(2_000_001)
    except urllib.error.HTTPError as exc:
        # Do not emit provider response bodies: they can repeat submitted text.
        raise RuntimeError("provider_http_" + str(exc.code)) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("provider_transport_error_no_retry") from None
    if len(raw) > 2_000_000:
        raise RuntimeError("provider_response_size_limit")
    return json.loads(raw)


def extract_finding(response):
    if response.get("status") != "completed":
        raise ValueError("provider_response_not_completed")
    texts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") == "refusal":
                raise ValueError("provider_refusal")
            if block.get("type") == "output_text":
                texts.append(block["text"])
    if len(texts) != 1:
        raise ValueError("provider_expected_one_finding")
    return json.loads(texts[0])


def plan(packets, config):
    validate_config(config)
    investigator_reserve = 0.0
    eligible = 0
    for packet in packets:
        if not validate_packet(packet).valid:
            raise ValueError("invalid_evidence_packet")
        investigator_reserve += reservation(
            request_body(build_investigator_messages(packet), config), config)
        agents = {r["id"]: r["agent_id"] for r in packet["records"]}
        eligible += any(e["basis"] in {"direct", "explicit_reference"}
                        and agents[e["source"]] is not None and agents[e["target"]] is not None
                        and agents[e["source"]] != agents[e["target"]] for e in packet["edges"])
    # The skeptic request also contains the investigator response; before that
    # exists use the maximum allowed request length for its reservation.
    max_skeptic = ((config["limits"]["max_request_bytes"] + 4096)
                   * config["pricing"]["input_ceiling_per_million"]
                   + config["model"]["max_output_tokens"]
                   * config["pricing"]["output_per_million"]) / 1_000_000
    return {"mode": "offline_plan", "episodes": len(packets),
            "model": config["model"]["name"], "maximum_calls": len(packets) * 2,
            "conservative_investigator_reservation_usd": round(investigator_reserve, 4),
            "conservative_total_ceiling_usd": round(investigator_reserve + len(packets) * max_skeptic, 4),
            "configured_budget_usd": config["limits"]["budget_usd"],
            "packets_with_admissible_cross_agent_reference": eligible,
            "api_calls_made": 0, "dataset_text_in_output": False,
            "note": "Actual requests reserve budget before transmission; run stops at the cap. Prices and availability need account verification."}


def run_packet(packet, config, directory, ledger, key, transport=call_provider):
    """Caller holds an exclusive run lock. Reserve before any network request."""
    validate_config(config)
    if not validate_packet(packet).valid:
        raise ValueError("invalid_evidence_packet")
    directory = Path(directory)
    packet_digest = hashlib.sha256(canonical(packet)).hexdigest()
    finding = None
    passes = []
    for phase in ("investigator", "skeptic"):
        messages = (build_investigator_messages(packet) if finding is None
                    else build_skeptic_messages(packet, finding))
        body = request_body(messages, config)
        digest = hashlib.sha256(canonical(body)).hexdigest()
        cached = directory / "cache" / (digest + ".json")
        if digest in ledger["requests"]:
            if ledger["requests"][digest]["state"] != "completed" or not cached.is_file():
                raise RuntimeError("previous_attempt_unresolved_no_retry")
            result = json.loads(cached.read_text())
            finding = result["finding"]
            if not validate_finding(packet, finding).valid:
                raise RuntimeError("cached_finding_failed_current_validation")
            passes.append(result)
            continue
        reserve = reservation(body, config)
        already_reserved = sum(r["reservation_usd"] for r in ledger["requests"].values())
        if already_reserved + reserve > config["limits"]["budget_usd"]:
            raise RuntimeError("budget_cap_reached")
        ledger["requests"][digest] = {"state": "started", "reservation_usd": reserve,
                                      "phase": phase, "packet_digest": packet_digest}
        save_private(directory / "ledger.json", ledger)
        started = time.monotonic()
        response = transport(body, key, config["model"]["timeout_seconds"])
        # Keep only the structured result and usage, never hidden reasoning text.
        finding = extract_finding(response)
        validation = validate_finding(packet, finding)
        result = {"phase": phase, "finding": finding,
                  "validation": validation.to_dict(), "model": response.get("model"),
                  "usage": response.get("usage", {}),
                  "duration_seconds": round(time.monotonic() - started, 3)}
        save_private(cached, result)
        ledger["requests"][digest]["state"] = "completed" if validation.valid else "invalid"
        ledger["requests"][digest]["usage"] = result["usage"]
        save_private(directory / "ledger.json", ledger)
        if not validation.valid:
            raise RuntimeError("finding_failed_evidence_validation")
        passes.append(result)
    first, second = [p["finding"] for p in passes]
    agreement = (first["verdict"] == second["verdict"]
                 and first["category"] == second["category"]
                 and set(first["implicated_agents"]) == set(second["implicated_agents"]))
    result = {"episode_id": packet["episode_id"], "packet_digest": packet_digest,
              "passes": passes, "pass_agreement": agreement,
              "review_status": "candidate_incident" if agreement and second["verdict"] == "incident"
                  else "insufficient" if not agreement else second["verdict"],
              "human_adjudicated": False}
    save_private(directory / "findings" / (packet_digest + ".json"), result)
    return result
