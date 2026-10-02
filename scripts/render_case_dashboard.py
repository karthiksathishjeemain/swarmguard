#!/usr/bin/env python3
"""Render a self-contained, private evidence dashboard; no server or external assets.

Real-data inputs and the generated HTML stay under the Azure VM's restricted/
directory. --synthetic uses only the repository's authored fixtures, so that
demo can be inspected locally. Only parsed diagnostic labels are rendered;
raw model output, reasoning and API responses are never embedded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VM_RESTRICTED = Path("/home/azureuser/ai-swarm/restricted")
MAX_INPUT_BYTES = 32_000_000
sys.path.insert(0, str(PROJECT_ROOT))
from swarmguard.schema import validate_packet
from swarmguard.local_guard import DIAGNOSTIC_FIELDS
from swarmguard.decisions import route_episode


CSS = """
:root{color-scheme:light dark;--bg:#f5f7f9;--panel:#fff;--text:#17242d;--muted:#536570;--line:#d8e1e6;--accent:#076b62;--accent-soft:#e2f3ef;--warn:#855300;--warn-bg:#fff4db;--danger:#a52b3d;--danger-bg:#ffedf0;--shadow:0 8px 32px #162c3810}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}a{color:var(--accent);text-underline-offset:3px}button,a,input,select,summary{outline-offset:4px}button{cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid var(--accent)}[hidden]{display:none!important}.skip{position:absolute;left:16px;top:-80px;z-index:20;padding:12px;background:var(--panel)}.skip:focus{top:12px}.shell{max-width:1480px;margin:auto;padding:28px 32px 56px}.eyebrow{font:700 11px/1.4 ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}.masthead{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin:0 0 20px}.masthead h1{margin:6px 0 8px;font-size:clamp(30px,4vw,44px);letter-spacing:-.04em;line-height:1.12}.masthead p{margin:0;color:var(--muted);max-width:740px}.privacy{border:1px solid var(--line);border-radius:999px;padding:7px 12px;white-space:nowrap;font-size:12px;background:var(--panel)}.notice{padding:13px 18px;border:1px solid #cfa757;border-radius:10px;background:var(--warn-bg);color:var(--warn);margin:20px 0}.notice strong{display:block;font-size:15px}.notice p{margin:4px 0 0;font-size:13px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px}.stat b{display:block;font-size:25px;line-height:1.2;letter-spacing:-.03em}.stat span{font-size:12px;color:var(--muted)}.layout{display:grid;grid-template-columns:250px minmax(0,1fr);gap:24px;align-items:start}.sidebar{position:sticky;top:18px}.sidebar h2{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin:0 0 12px}.case-tabs{display:grid;gap:8px}.case-tab{display:block;text-align:left;width:100%;padding:13px 14px;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;transition:border-color .15s}.case-tab[aria-selected=true]{border-color:var(--accent);background:var(--accent-soft);box-shadow:inset 3px 0 var(--accent)}.case-tab .case-number{font-size:11px;color:var(--muted);display:block}.case-tab .case-id{display:block;font-weight:650;overflow-wrap:anywhere;margin:2px 0 7px}.sidebar-note{font-size:12px;color:var(--muted);margin-top:18px}.pill{display:inline-block;font-size:11px;font-weight:650;border-radius:999px;padding:3px 9px;background:var(--bg);border:1px solid var(--line);color:var(--muted);vertical-align:middle;line-height:1.45}.pill.unsafe{background:var(--danger-bg);border-color:var(--danger);color:var(--danger)}.pill.safe{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}.pill.warn{background:var(--warn-bg);border-color:var(--warn);color:var(--warn)}.episode{min-width:0}.episode-header h2{margin:0;font-size:24px;letter-spacing:-.03em;overflow-wrap:anywhere}.episode-header p{margin:5px 0 18px;color:var(--muted);font-size:13px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px;box-shadow:var(--shadow)}.panel h3{font-size:15px;margin:0 0 12px}.panel p{margin:8px 0;color:var(--muted);font-size:13px}.signal-top{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.signal-top h3{margin:0}.signal-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0 4px}.signal-grid dt{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}.signal-grid dd{font-size:13px;font-weight:600;margin:3px 0 0;overflow-wrap:anywhere}.small{font-size:12px;color:var(--muted)}.coverage-list{margin:8px 0;padding-left:20px;font-size:13px;color:var(--muted)}.coverage-list li+li{margin-top:4px}.filters{display:grid;grid-template-columns:minmax(120px,2fr) minmax(100px,1fr) minmax(120px,1fr);gap:10px;margin:12px 0}.filters label{font-size:11px;color:var(--muted);font-weight:650}.filters input,.filters select{display:block;width:100%;margin-top:4px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--text);padding:8px;font-size:13px;min-width:0}.timeline-title{display:flex;justify-content:space-between;align-items:baseline;gap:12px}.timeline-title h3{margin-bottom:0}.match-count{font-size:12px;color:var(--muted)}.timeline{margin:20px 0 0;padding:0;list-style:none}.record{position:relative;border-left:2px solid var(--line);padding:0 0 22px 22px;margin-left:7px;scroll-margin-top:18px}.record:last-child{padding-bottom:0;border-left-color:transparent}.record::before{content:"";position:absolute;left:-6px;top:5px;width:10px;height:10px;background:var(--accent);border:2px solid var(--panel);border-radius:50%;box-shadow:0 0 0 1px var(--accent)}.record[data-kind=goal]::before,.record[data-kind=instruction]::before{border-radius:2px;background:var(--warn);box-shadow:0 0 0 1px var(--warn)}.record-head{display:flex;align-items:center;flex-wrap:wrap;gap:8px;font-size:12px}.record-agent{font-weight:700}.record-kind{text-transform:uppercase;letter-spacing:.08em;font-size:10px;color:var(--muted)}.record time{color:var(--muted);margin-left:auto;font:11px ui-monospace,monospace}.record pre{font:13px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:9px 0;padding:12px 14px;border:1px solid var(--line);border-radius:8px;background:var(--bg);max-height:310px;overflow:auto;tab-size:2}.record-id{font:11px ui-monospace,monospace;color:var(--muted);overflow-wrap:anywhere}.record:target pre{outline:2px solid var(--accent)}.edge-list{margin:0;padding:0;list-style:none}.edge-list li{padding:11px 0;border-bottom:1px solid var(--line)}.edge-list li:last-child{border-bottom:0;padding-bottom:0}.edge-line{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font:12px ui-monospace,monospace;overflow-wrap:anywhere}.edge-list p{font-size:12px;margin:5px 0 0}.edge-basis{font-family:system-ui,sans-serif}.edge-arrow{color:var(--muted)}details{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}summary{font-size:13px;font-weight:600;cursor:pointer}.empty{color:var(--muted);padding:14px 0;font-size:13px}.footer{font-size:12px;color:var(--muted);margin:28px 0 0;border-top:1px solid var(--line);padding-top:16px}.expectation{border-left:3px solid var(--accent);padding-left:12px}.expectation strong{font-size:12px}code{overflow-wrap:anywhere}.noscript{padding:10px;border:1px solid var(--line)}
@media(prefers-color-scheme:dark){:root{--bg:#111b22;--panel:#17252e;--text:#e6edf1;--muted:#aec0cb;--line:#344954;--accent:#77dfcd;--accent-soft:#1b3b3b;--warn:#f4cb77;--warn-bg:#392f1c;--danger:#ff9bad;--danger-bg:#3f2530;--shadow:none}.notice{border-color:#8e733b}}
@media(max-width:860px){.shell{padding:20px 16px 36px}.layout{grid-template-columns:1fr}.sidebar{position:static}.case-tabs{grid-template-columns:repeat(2,minmax(0,1fr));max-height:290px;overflow:auto}.sidebar-note{margin-bottom:0}.stats{gap:8px}.stat{padding:12px}.masthead{display:block}.privacy{display:inline-block;margin-top:14px}.signal-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:480px){.stats{grid-template-columns:repeat(2,1fr)}.filters{grid-template-columns:1fr}.panel{padding:15px}.record time{margin-left:0;width:100%}.signal-grid{grid-template-columns:1fr}.signal-top{display:block}.signal-top .pill{margin-top:8px}}
@media print{body{background:white;color:black}.shell{max-width:none;padding:0}.sidebar,.filters,.skip,.privacy{display:none}.layout{display:block}.episode[hidden]{display:block!important}.panel{box-shadow:none;break-inside:avoid}.record pre{max-height:none}.episode{break-before:page}.stats{margin-bottom:12px}}
""".strip()


JS = """
(() => {
  'use strict';
  const tabs = Array.from(document.querySelectorAll('.case-tab'));
  const panels = Array.from(document.querySelectorAll('.episode'));
  function choose(tab, moveFocus = false) {
    tabs.forEach(item => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    panels.forEach(panel => { panel.hidden = panel.id !== tab.getAttribute('aria-controls'); });
    if (moveFocus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => choose(tab));
    tab.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowDown' || event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault(); choose(tabs[next], true);
    });
  });
  panels.forEach(panel => {
    const search = panel.querySelector('[data-filter=search]');
    const kind = panel.querySelector('[data-filter=kind]');
    const agent = panel.querySelector('[data-filter=agent]');
    const records = Array.from(panel.querySelectorAll('.record'));
    const counter = panel.querySelector('.match-count');
    function filter() {
      const term = search.value.toLocaleLowerCase();
      let shown = 0;
      records.forEach(record => {
        const keep = (!kind.value || record.dataset.kind === kind.value) &&
          (!agent.value || record.dataset.agent === agent.value) &&
          (!term || record.textContent.toLocaleLowerCase().includes(term));
        record.hidden = !keep;
        if (keep) shown += 1;
      });
      counter.textContent = String(shown) + ' of ' + String(records.length) + ' records';
      panel.querySelector('.no-matches').hidden = shown !== 0;
    }
    [search, kind, agent].forEach(control => control.addEventListener('input', filter));
    panel.querySelectorAll('.edge-line a, .judge-citations a').forEach(link => {
      link.addEventListener('click', () => { search.value = ''; kind.value = ''; agent.value = ''; filter(); });
    });
    filter();
  });
  if (tabs.length) choose(tabs[0]);
})();
""".strip()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def element_id(episode_id: str, record_id: str) -> str:
    """Never put corpus IDs into HTML selectors, fragments or JavaScript."""
    return "r-" + hashlib.sha256((episode_id + "\0" + record_id).encode()).hexdigest()[:24]


def edge_element_id(episode_id: str, edge_id: str) -> str:
    return "e-" + hashlib.sha256((episode_id + "\0" + edge_id).encode()).hexdigest()[:24]


def read_json(path: Path, jsonl: bool = False):
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("input_size_limit")
    raw = path.read_text(encoding="utf-8")
    if jsonl and not raw.lstrip().startswith("["):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def is_restricted(path: Path, root: Path = VM_RESTRICTED) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def validate_real_paths(inputs: list[Path], output: Path) -> None:
    if sys.platform != "linux" or not Path("/mnt/ai-swarm-scratch").is_mount():
        raise ValueError("real_data_requires_azure_vm")
    if not all(is_restricted(path) for path in inputs + [output]):
        raise ValueError("real_data_requires_restricted_paths")


def write_private(path: Path, document: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".dashboard-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _bounded_label(value: object, fallback: str = "Not provided") -> str:
    # Diagnostic fields are parsed labels, never free-form rationale. Model
    # output containing XML/analysis, multiline text or long prose is omitted.
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _./:()+,-]{0,119}", value):
        return value
    return fallback


def diagnostic_index(report: object) -> tuple[dict[str, dict], str]:
    if report is None:
        return {}, "Not run"
    model = "Not provided"
    if isinstance(report, dict):
        if isinstance(report.get("model"), dict):
            model = _bounded_label(report["model"].get("model_id"))
        rows = report.get("results", [])
    elif isinstance(report, list):
        rows = report
    else:
        raise ValueError("invalid_diagnostic_report")
    if not isinstance(rows, list):
        raise ValueError("invalid_diagnostic_results")
    result: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("episode_id"), str):
            raise ValueError("invalid_diagnostic_row")
        if row["episode_id"] in result:
            raise ValueError("duplicate_diagnostic_episode")
        result[row["episode_id"]] = row
    return result, model


def diagnostic_label(row: dict | None) -> tuple[str, str]:
    if row is None:
        return "Not run", ""
    if row.get("_labels_suppressed"):
        return "Guard: insufficient", "warn"
    if row.get("input_truncated") is True or row.get("generation_incomplete") is True:
        return "Guard: insufficient", "warn"
    if row.get("status") == "classified" and row.get("safety") in ("safe", "unsafe"):
        safety = row["safety"]
        return f"Guard: {safety}", safety
    if row.get("status") == "insufficient":
        return "Guard: insufficient", "warn"
    return "Unrecognized result", "warn"


def prepare_display_diagnostics(packets: list[dict], diagnostics: object = None,
                                manifest: dict | None = None, *, synthetic: bool = False):
    """Use the router's full validity/binding checks for every visible label.

    Both HTML and CLI aggregates use these rows. A raw classifier label is not
    displayable merely because an episode ID matches. Coverage-only gaps may
    retain a properly bound safety signal while the overall route abstains.
    """
    indexed, model_name = diagnostic_index(diagnostics)
    model = diagnostics.get("model", {}) if isinstance(diagnostics, dict) else {}
    model = model if isinstance(model, dict) else {}
    coverage = manifest if isinstance(manifest, dict) else {}
    fatal_reasons = {
        "invalid_packet", "unverified_model_checkpoint", "model_not_run",
        "stale_or_unbound_model_result", "model_abstained_or_incomplete",
        "missing_or_stale_retrieval_manifest",
    }
    prepared = {}
    for packet in packets:
        episode_id = packet["episode_id"]
        row = indexed.get(episode_id)
        metadata = coverage.get(episode_id, {})
        metadata = metadata if isinstance(metadata, dict) else {}
        route = route_episode(packet, row, model, metadata, synthetic=synthetic)
        display_row = None
        if row is not None:
            bound = (route.get("safety_signal") in ("safe", "unsafe")
                     and not fatal_reasons.intersection(route.get("reasons", [])))
            display_row = {**row, "_reported_status": row.get("status"), "_labels_suppressed": not bound}
            if not bound:
                display_row.update(status="insufficient", safety=None, diagnostics=None)
        prepared[episode_id] = (display_row, route)
    return prepared, model_name


def prepare_judge_diagnostics(packets: list[dict], diagnostics: object = None):
    """Bind contextual findings independently of the broad safety baseline."""
    indexed, model_name = diagnostic_index(diagnostics)
    if diagnostics is None:
        return {packet["episode_id"]: None for packet in packets}, model_name
    # Lazy import keeps baseline-only reports usable without installing models
    # or loading the contextual inference adapter.
    from swarmguard.local_judge import bind_judge_result, validate_judgment
    model = diagnostics.get("model", {}) if isinstance(diagnostics, dict) else {}
    model = model if isinstance(model, dict) else {}
    prepared = {}
    for packet in packets:
        row = indexed.get(packet["episode_id"])
        if row is None:
            prepared[packet["episode_id"]] = None
            continue
        try:
            errors = list(bind_judge_result(packet, row, model))
            if not errors:
                errors.extend(validate_judgment(packet, row.get("finding")))
        except (KeyError, TypeError, ValueError):
            errors = ["invalid_judge_result"]
        shown = {**row, "_reported_status": row.get("status"), "_binding_errors": errors}
        if errors:
            shown.update(status="insufficient", finding=None)
        prepared[packet["episode_id"]] = shown
    return prepared, model_name


def judge_label(row: dict | None) -> tuple[str, str]:
    if row is None:
        return "Judge: not run", ""
    finding = row.get("finding") if isinstance(row.get("finding"), dict) else {}
    names = {
        "incident_candidate": ("Judge: candidate", "warn"),
        "no_finding": ("Judge: no finding", ""),
        "insufficient": ("Judge: insufficient", "warn"),
    }
    return names.get(finding.get("verdict"), ("Judge: insufficient", "warn"))


def render_judge_signal(packet: dict, row: dict | None, model_name: str, *, baseline_available: bool = True) -> str:
    label, color = judge_label(row)
    top = ('<section class="panel" aria-label="Contextual judgment"><div class="signal-top">'
           '<h3>Contextual judgment</h3>' + f'<span class="pill {color}">{esc(label)}</span></div>')
    if row is None:
        note = ("The broad guard assessment below is a separate baseline." if baseline_available
                else "No contextual verdict is available for this episode.")
        return top + f'<p>No contextual judge result is attached. {note}</p></section>'
    finding = row.get("finding") if isinstance(row.get("finding"), dict) else {}
    inference_mode = ("Reasoning enabled" if row.get("thinking_enabled", False) else "Fast mode") if finding else "Unverified or unavailable"
    fields = [
        ("Model", model_name), ("Actual run status", _bounded_label(row.get("_reported_status", row.get("status")))),
        ("Latency", metric(row.get("latency_seconds"), " s")),
        ("Category", finding.get("category", "Suppressed or unavailable")),
        ("Reason code", finding.get("reason_code", "Suppressed or unavailable")),
        ("Input / output tokens", metric(row.get("input_tokens")) + " / " + metric(row.get("output_tokens"))),
        ("Verified inference mode", inference_mode),
    ]
    grid = '<dl class="signal-grid">' + ''.join(
        f'<div><dt>{esc(name)}</dt><dd>{esc(value)}</dd></div>' for name, value in fields) + '</dl>'
    if not finding:
        return top + grid + ('<p>The contextual finding is suppressed because checkpoint, packet, prompt or output-validity '
                             'checks did not pass, or the run did not complete. No candidate label or evidence attribution is shown.</p></section>')
    records = {record["id"]: record for record in packet["records"]}
    edges = {edge["id"]: edge for edge in packet["edges"] if edge.get("id")}
    groups = []
    for field, title in (
        ("rule_record_ids", "Governing records"), ("source_record_ids", "Source records"),
        ("peer_record_ids", "Peer records"), ("action_record_ids", "Actions"),
        ("result_record_ids", "Results"), ("counterevidence_record_ids", "Counterevidence"),
    ):
        ids = finding.get(field, [])
        # The adapter validation already rejects unknown IDs. Keep the link
        # renderer independently fail-closed on missing or unexpected values.
        links = [f'<a href="#{element_id(packet["episode_id"], record_id)}">{esc(record_id)}</a>'
                 for record_id in ids if record_id in records]
        groups.append(f'<p><strong>{title}:</strong> ' + (' · '.join(links) if links else 'None cited') + '</p>')
    edge_links = [f'<a href="#{edge_element_id(packet["episode_id"], edge_id)}">{esc(edge_id)}</a>'
                  for edge_id in finding.get("graph_edge_ids", []) if edge_id in edges]
    groups.append('<p><strong>Reference edges:</strong> ' + (' · '.join(edge_links) if edge_links else 'None cited') + '</p>')
    return (top + grid + '<details open><summary>Model-selected evidence references</summary><div class="judge-citations">' +
            ''.join(groups) + '</div></details><p>These links identify exact existing records, not verified conclusions. '
            'A contextual candidate is human-unvalidated; a no-finding label is limited to the supplied packet. '
            'Reference checks do not establish that the evidence supports the model’s interpretation.</p></section>')


def metric(value: object, suffix: str = "") -> str:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return "Not provided"
    return f"{value:,.2f}{suffix}" if isinstance(value, float) else f"{value:,}{suffix}"


def render_signal(row: dict | None, model: str) -> str:
    label, color = diagnostic_label(row)
    heading = ('<div class="signal-top"><h3>Local model assessment</h3>'
               f'<span class="pill {color}">{esc(label)}</span></div>')
    if row is None:
        return ('<section class="panel" aria-label="Model assessment">' + heading +
                '<p>No model result is attached to this episode. The timeline below is evidence, '
                'not a predicted or human-validated incident.</p></section>')
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    fields = [
        ("Model", model), ("Actual run status", _bounded_label(row.get("_reported_status", row.get("status")))),
        ("Latency", metric(row.get("latency_seconds"), " s")),
        *[(name, diagnostics.get(key) if diagnostics.get(key) in labels else "Not provided")
          for name, (key, labels) in DIAGNOSTIC_FIELDS.items()],
        ("Input tokens", metric(row.get("input_tokens"))),
        ("Output tokens", metric(row.get("output_tokens"))),
        ("Human adjudication", "Not represented in this view"),
    ]
    grid = '<dl class="signal-grid">' + ''.join(
        f'<div><dt>{esc(name)}</dt><dd>{esc(value)}</dd></div>' for name, value in fields) + '</dl>'
    limits = []
    if not row.get("packet_sha256"):
        limits.append("No packet digest is attached to this result; its association with these exact records is unverified.")
    if row.get("_labels_suppressed"):
        limits.append("Safety and diagnostic labels are suppressed because result binding or validity checks did not pass. See review routing.")
    if row.get("input_truncated") is True:
        limits.append("The model input was truncated; omitted evidence can change the assessment.")
    if row.get("generation_incomplete") is True:
        limits.append("Generation was incomplete; treat any available label as incomplete output.")
    errors = row.get("errors")
    if isinstance(errors, list):
        safe_errors = [_bounded_label(value, "") for value in errors]
        safe_errors = [value for value in safe_errors if value]
        if safe_errors:
            limits.append("Run diagnostics: " + ", ".join(safe_errors[:12]))
    limit_html = ('<ul class="coverage-list">' + ''.join(f'<li>{esc(v)}</li>' for v in limits) + '</ul>') if limits else ''
    return ('<section class="panel" aria-label="Model assessment">' + heading + grid + limit_html +
            '<p>A guard safety label assesses the supplied trajectory. It does not establish '
            'intent, collective involvement or a proven swarm violation. Raw model reasoning is not included.</p></section>')


def render_route(route: dict) -> str:
    """Explain deterministic routing without turning a model label into attribution."""
    names = {
        "insufficient": ("Insufficient evidence", "warn"),
        "needs_context_review": ("Review with missing context", "warn"),
        "needs_review": ("Needs human review", "warn"),
        "unattributed_safety_signal": ("Review actor attribution", "warn"),
        "no_signal": ("No unsafe signal in this packet", ""),
    }
    explanations = {
        "invalid_packet": "The evidence packet did not pass structural validation.",
        "unverified_model_checkpoint": "The model checkpoint is not bound to the verified revision and download manifest.",
        "model_not_run": "No model run is attached to this episode.",
        "stale_or_unbound_model_result": "The result is not bound to this exact packet, model and prompt version.",
        "model_abstained_or_incomplete": "The model abstained, returned incomplete output, or encountered an error.",
        "missing_or_stale_retrieval_manifest": "The retrieval manifest is missing or does not match the packet.",
        "unsafe_signal_with_cross_agent_reference_context": "An unsafe diagnostic and a cross-agent reference are available for review.",
        "reference_does_not_establish_adoption_or_joint_misconduct": "The reference does not establish adoption of a claim or joint misconduct.",
        "no_verified_cross_agent_reference_for_swarm_review": "No verified cross-agent reference connects the diagnostic to swarm-level review.",
        "no_unsafe_signal_in_incomplete_packet": "The model returned no unsafe signal, but the packet is incomplete.",
        "model_returned_safe_for_this_packet_only": "The model returned safe for this packet only; this does not establish broader alignment.",
        "retrieved_window_is_not_a_complete_history": "The retrieved window is not a complete history.",
        "source_action_window_truncated": "A source action window was truncated.",
        "record_text_excerpted_or_truncated": "Some record text was excerpted or truncated.",
        "governing_authorization_not_established": "The governing authorization is not established in the packet.",
    }
    actions = {
        "inspect_missing_evidence": "Inspect missing evidence before drawing a conclusion.",
        "review_source_actions_and_reference_meaning": "Review the source actions and the meaning of the reference.",
        "review_actor_attribution_before_swarm_classification": "Establish which actor the safety signal concerns before considering swarm classification.",
        "retain_for_blinded_negative_control_review": "Retain this episode for blinded review as a negative control.",
    }
    label, color = names.get(route.get("route"), ("Unrecognized route", "warn"))
    reason_codes = route.get("reasons", [])
    not_run = "model_not_run" in reason_codes
    if not_run:
        label, color = "Not run", ""
        reason_codes = ["model_not_run"]
    reasons = [explanations[code] for code in reason_codes[:12] if code in explanations]
    gaps = [explanations[code] for code in route.get("coverage_gaps", [])[:12] if code in explanations]
    reason_html = '<ul class="coverage-list">' + ''.join(f'<li>{esc(text)}</li>' for text in reasons) + '</ul>' if reasons else ''
    gap_html = ('<details><summary>Routing coverage gaps</summary><ul class="coverage-list">' +
                ''.join(f'<li>{esc(text)}</li>' for text in gaps) + '</ul></details>') if gaps else ''
    action = actions.get(route.get("next_action"), "Review the available evidence.")
    if not_run:
        action = "Attach a verified local model result to assess this episode."
    return ('<section class="panel" aria-label="Deterministic review routing"><div class="signal-top">'
            f'<h3>Review routing</h3><span class="pill {color}">{esc(label)}</span></div>' + reason_html +
            f'<p><strong>Next step:</strong> {esc(action)}</p>' + gap_html +
            '<p>Routing prioritizes investigation. It does not identify implicated agents, establish a swarm incident '
            'or provide model-generated citations. Cross-agent references are context, not joint attribution.</p></section>')


def _time_key(record: dict) -> datetime:
    parsed = datetime.fromisoformat(record["at"].replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def render_coverage(packet: dict, metadata: dict, validation_warnings: tuple[str, ...]) -> str:
    records = metadata.get("records") if isinstance(metadata.get("records"), dict) else {}
    truncated = sum(any(info.get(key) is True for key in (
        "builder_truncated", "packet_bound_truncated", "existing_truncation_marker"))
        for info in records.values() if isinstance(info, dict))
    details = []
    if not metadata:
        details.append("No source-coverage manifest is attached; completeness and upstream truncation are unknown.")
    else:
        details.append(f"{truncated} records are marked as excerpted or truncated in the attached manifest.")
        if metadata.get("source_window_truncated") is True or metadata.get("target_source_window_truncated") is True:
            details.append("At least one source action window was already truncated before episode construction.")
        if metadata.get("context_is_retrospective") is True:
            details.append("This is a retrospective evidence selection, not a real-time detection replay.")
        for value in metadata.get("limitations", []) if isinstance(metadata.get("limitations"), list) else []:
            if isinstance(value, str):
                details.append(value)
    if not any(r["kind"] in ("goal", "instruction") for r in packet["records"]):
        details.append("No governing goal or instruction is present; a goal or authorization conflict cannot be established here.")
    if not any(e["basis"] in ("direct", "explicit_reference") for e in packet["edges"]):
        details.append("No direct or explicit-reference relationship is included.")
    details.extend(validation_warnings)
    return ('<details><summary>Evidence coverage and limitations</summary><ul class="coverage-list">' +
            ''.join(f'<li>{esc(value)}</li>' for value in details) + '</ul></details>')


def render_timeline(packet: dict, metadata: dict) -> str:
    agents = sorted({r["agent_id"] for r in packet["records"] if r["agent_id"] is not None})
    kinds = [kind for kind in ("goal", "instruction", "chat", "action", "result")
             if any(r["kind"] == kind for r in packet["records"])]
    filters = ('<div class="filters"><label>Find in episode<input data-filter="search" '
               'type="search" placeholder="Search evidence…" autocomplete="off"></label>'
               '<label>Record type<select data-filter="kind"><option value="">All types</option>' +
               ''.join(f'<option value="{esc(k)}">{esc(k.title())}</option>' for k in kinds) +
               '</select></label><label>Agent<select data-filter="agent"><option value="">All agents</option>' +
               '<option value="__governing__">Governing context</option>' +
               ''.join(f'<option value="{esc(a)}">{esc(a)}</option>' for a in agents) + '</select></label></div>')
    metadata_records = metadata.get("records") if isinstance(metadata.get("records"), dict) else {}
    rows = []
    for record in sorted(packet["records"], key=_time_key):
        agent = record["agent_id"]
        info = metadata_records.get(record["id"], {})
        marker = ''
        if isinstance(info, dict) and any(info.get(key) is True for key in (
                "builder_truncated", "packet_bound_truncated", "existing_truncation_marker")):
            marker = ' <span class="pill warn">Excerpted / truncated</span>'
        if isinstance(info, dict) and info.get("redactions"):
            marker += ' <span class="pill">Redacted</span>'
        rows.append(
            f'<li class="record" id="{element_id(packet["episode_id"], record["id"])}" '
            f'data-kind="{esc(record["kind"])}" data-agent="{esc(agent if agent is not None else "__governing__")}">'
            f'<div class="record-head"><span class="record-agent">{esc(agent or "Governing context")}</span>'
            f'<span class="record-kind">{esc(record["kind"])}</span>{marker}'
            f'<time datetime="{esc(record["at"])}">{esc(record["at"])}</time></div>'
            f'<pre>{esc(record["text"])}</pre><div class="record-id">Source record · {esc(record["id"])}</div></li>')
    return ('<section class="panel" aria-label="Evidence timeline"><div class="timeline-title">'
            '<h3>Evidence timeline</h3><span class="match-count" aria-live="polite"></span></div>' + filters +
            '<p>Messages are displayed as recorded evidence. Quoting an instruction or claim does not mean endorsing it.</p>'
            '<ol class="timeline">' + ''.join(rows) + '</ol><p class="no-matches empty" hidden>No matching records.</p></section>')


def render_edges(packet: dict) -> str:
    by_id = {record["id"]: record for record in packet["records"]}
    rows = []
    basis_notes = {
        "direct": "A direct source relationship; this alone does not prove agreement or a violation.",
        "explicit_reference": "An explicit reference; the target may endorse, challenge or merely quote the source.",
        "shared_artifact": "A shared artifact is context, not proof of influence or receipt.",
        "temporal_only": "Temporal proximity is context, not evidence of coordination.",
    }
    for edge in packet["edges"]:
        links = []
        for field in ("source", "target"):
            record = by_id[edge[field]]
            agent = record["agent_id"] or "Governing context"
            links.append(f'<a href="#{element_id(packet["episode_id"], record["id"])}">{esc(agent)} · {esc(record["id"])}</a>')
        target = f' id="{edge_element_id(packet["episode_id"], edge["id"])}"' if edge.get("id") else ''
        rows.append('<li' + target + '><div class="edge-line">' + links[0] + '<span class="edge-arrow" aria-label="to">→</span>' +
                    links[1] + '</div><p><span class="pill edge-basis">' + esc(edge["basis"].replace('_', ' ')) +
                    '</span> ' + esc(edge["relation"].replace('_', ' ')) + '</p><p>' + esc(basis_notes[edge["basis"]]) + '</p></li>')
    body = '<ul class="edge-list">' + ''.join(rows) + '</ul>' if rows else '<p>No relationships are attached to this episode.</p>'
    return '<section class="panel" aria-label="Agent relationships"><h3>Recorded relationships</h3>' + body + '</section>'


def render_dashboard(packets: list[dict], diagnostics: object = None, manifest: dict | None = None,
                     *, synthetic: bool = False, expectations: dict | None = None,
                     judge_diagnostics: object = None) -> str:
    if not isinstance(packets, list) or not 1 <= len(packets) <= 100:
        raise ValueError("dashboard_accepts_1_to_100_episodes")
    if len({packet.get("episode_id") for packet in packets if isinstance(packet, dict)}) != len(packets):
        raise ValueError("duplicate_or_invalid_episode")
    if sum(len(packet.get("records", [])) for packet in packets) > 10000:
        raise ValueError("record_count_limit")
    validations = [validate_packet(packet) for packet in packets]
    if any(not result.valid for result in validations):
        raise ValueError("invalid_evidence_packet")
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError("invalid_coverage_manifest")
    prepared, model = prepare_display_diagnostics(packets, diagnostics, manifest, synthetic=synthetic)
    prepared_judge, judge_model = prepare_judge_diagnostics(packets, judge_diagnostics)
    judge_only = judge_diagnostics is not None and diagnostics is None
    manifest = manifest or {}
    expectations = expectations or {}
    tabs, panels = [], []
    for index, (packet, validation) in enumerate(zip(packets, validations)):
        episode_id = packet["episode_id"]
        row, route = prepared[episode_id]
        label, color = diagnostic_label(row)
        judge_row = prepared_judge[episode_id]
        judge_status, judge_color = judge_label(judge_row)
        meta = manifest.get(episode_id, {})
        meta = meta if isinstance(meta, dict) else {}
        agents = {record["agent_id"] for record in packet["records"] if record["agent_id"] is not None}
        badges = ('' if judge_only else f'<span class="pill {color}">{esc(label)}</span> ')
        badges += f'<span class="pill {judge_color}">{esc(judge_status)}</span>'
        tabs.append(
            f'<button class="case-tab" id="tab-{index}" type="button" role="tab" aria-controls="episode-{index}" '
            f'aria-selected="{str(index == 0).lower()}"><span class="case-number">EPISODE {index + 1:02d}</span>'
            f'<span class="case-id">{esc(episode_id)}</span>{badges}</button>')
        coverage = render_coverage(packet, meta, validation.warnings)
        expectation = ''
        expected = expectations.get(episode_id)
        if synthetic and isinstance(expected, dict):
            expectation = ('<div class="expectation"><strong>Authored evaluation expectation — not model output</strong>'
                           f'<p>{esc(expected.get("expected_verdict", "Unknown"))} · '
                           f'{esc(expected.get("expected_category", "Unknown"))}</p>'
                           f'<p>{esc(expected.get("explanation", ""))}</p></div>')
        panels.append(
            f'<article class="episode" id="episode-{index}" role="tabpanel" aria-labelledby="tab-{index}" tabindex="0">'
            f'<header class="episode-header"><h2>{esc(episode_id)}</h2><p>{len(agents)} agents · '
            f'{len(packet["records"])} evidence records · {len(packet["edges"])} relationships</p></header>' +
            render_judge_signal(packet, judge_row, judge_model, baseline_available=not judge_only) +
            ('' if judge_only else render_signal(row, model) + render_route(route)) +
            '<section class="panel" aria-label="Case scope"><h3>Case scope</h3>' +
            ('<p>This episode is an authored fictional fixture. Expected labels are test targets, not measured accuracy.</p>'
             if synthetic else '<p>This is a bounded selection of source records. Evidence outside the selection may change the interpretation.</p>') +
            expectation + coverage + '</section>' + render_edges(packet) + render_timeline(packet, meta) + '</article>')
    unique_agents = {r["agent_id"] for p in packets for r in p["records"] if r["agent_id"] is not None}
    guard_count = sum(prepared[p["episode_id"]][0] is not None for p in packets)
    judge_count = sum(prepared_judge[p["episode_id"]] is not None for p in packets)
    attached_results = ((judge_count, "Attached judge results") if judge_only else
                        (guard_count + judge_count, "Attached results · guard + judge") if judge_diagnostics is not None else
                        (guard_count, "Attached guard results"))
    statistics = [
        (len(packets), "Episodes in this view"), (len(unique_agents), "Distinct agents"),
        (sum(len(p["records"]) for p in packets), "Source records"),
        attached_results,
    ]
    stat_html = ''.join(f'<div class="stat"><b>{n:,}</b><span>{esc(label)}</span></div>' for n, label in statistics)
    import base64
    style_hash = base64.b64encode(hashlib.sha256(CSS.encode()).digest()).decode()
    script_hash = base64.b64encode(hashlib.sha256(JS.encode()).digest()).decode()
    csp = (f"default-src 'none'; script-src 'sha256-{script_hash}'; style-src 'sha256-{style_hash}'; "
           "connect-src 'none'; img-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'")
    mode = "Authored synthetic demo" if synthetic else "Private Azure evidence view"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{esc(csp)}"><meta name="referrer" content="no-referrer">'
        '<title>SwarmGuard · Evidence investigation</title><style>' + CSS + '</style></head><body>'
        '<a class="skip" href="#case-content">Skip to evidence</a><main class="shell">'
        '<header class="masthead"><div><span class="eyebrow">SwarmGuard / Investigation workspace</span>'
        '<h1>Follow the evidence.</h1><p>Inspect the instructions, actions and cross-agent references behind a diagnostic signal.</p></div>'
        f'<span class="privacy">{esc(mode)} · Offline HTML</span></header>'
        '<aside class="notice" role="note"><strong>Diagnostic signal—not human-validated finding</strong>'
        '<p>Model labels and reference links are evidence for investigation. They do not establish intent, agreement or a proven swarm violation.</p></aside>'
        '<div class="stats">' + stat_html + '</div><noscript><p class="noscript">JavaScript is disabled. All episodes remain readable; filters require JavaScript.</p></noscript>'
        '<div class="layout"><nav class="sidebar" aria-label="Episode selection"><h2>Episodes</h2>'
        '<div class="case-tabs" role="tablist" aria-label="Select episode" aria-orientation="vertical">' + ''.join(tabs) + '</div>'
        '<p class="sidebar-note">Select an episode to inspect its model status, source coverage and evidence. Use arrow keys to move between episodes.</p></nav>'
        '<div id="case-content">' + ''.join(panels) + '</div></div>'
        '<footer class="footer">Self-contained report · No external assets, telemetry, network requests or execution of recorded commands. '
        'No annotations are saved by this view. Raw model reasoning is not embedded.</footer></main><script>' + JS + '</script></body></html>')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Use only the repository's authored fixtures")
    parser.add_argument("--packets", type=Path, help="JSON array or JSONL, on VM under restricted/")
    parser.add_argument("--results", type=Path, help="Optional local guard diagnostic report")
    parser.add_argument("--judge-results", type=Path, help="Optional local contextual judge report")
    parser.add_argument("--manifest", type=Path, help="Optional packet coverage manifest, on VM under restricted/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.synthetic:
        if args.packets or args.manifest:
            parser.error("--synthetic accepts only the fixed authored fixtures, not corpus packets or manifests")
        packets = read_json(PROJECT_ROOT / "fixtures/frontier_synthetic.json")
        answers = read_json(PROJECT_ROOT / "fixtures/frontier_synthetic_answers.json")
        expectations = {row["episode_id"]: row for row in answers}
        manifest = None
    else:
        if not args.packets:
            parser.error("--packets is required for real-data rendering")
        validate_real_paths([p for p in (args.packets, args.results, args.judge_results, args.manifest) if p], args.output)
        packets = read_json(args.packets, jsonl=True)
        expectations = None
        manifest = read_json(args.manifest) if args.manifest else None
    diagnostics = read_json(args.results) if args.results else None
    judge_diagnostics = read_json(args.judge_results) if args.judge_results else None
    document = render_dashboard(packets, diagnostics, manifest, synthetic=args.synthetic, expectations=expectations,
                                judge_diagnostics=judge_diagnostics)
    write_private(args.output, document)
    prepared, _ = prepare_display_diagnostics(packets, diagnostics, manifest, synthetic=args.synthetic)
    judge_only = judge_diagnostics is not None and diagnostics is None
    counts = Counter() if judge_only else Counter(diagnostic_label(prepared[p["episode_id"]][0])[0] for p in packets)
    prepared_judge, _ = prepare_judge_diagnostics(packets, judge_diagnostics)
    judge_counts = Counter(judge_label(prepared_judge[p["episode_id"]])[0] for p in packets)
    print(json.dumps({"status": "rendered", "mode": "synthetic" if args.synthetic else "private_vm",
                      "episodes": len(packets), "records": sum(len(p["records"]) for p in packets),
                      "model_status_counts": dict(counts), "judge_status_counts": dict(judge_counts),
                      "html_bytes": len(document.encode()),
                      "dataset_text_in_stdout": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError):
        # Error strings from files can contain corpus text; only a controlled
        # aggregate code is emitted at this boundary.
        print(json.dumps({"status": "invalid_input_or_access", "dataset_text_in_stdout": False}))
        raise SystemExit(2)
