#!/usr/bin/env python3
"""Read-only Azure VM diagnostics; JSON to stdout, no dataset access or installs.

Azure CLI may maintain its existing authentication cache during the auth check.
"""
import json
import sys

if sys.platform != "linux":
    print(json.dumps({"ok": False, "error": "NonLinuxRefused"}))
    raise SystemExit(2)

import argparse
from datetime import datetime, timezone
import os
import platform
import shutil
import subprocess
import urllib.request


def failure(exc):
    return {"ok": False, "error": type(exc).__name__}


def command(argv, parser=lambda value: value.strip(), timeout=10):
    """Capture errors privately; expose only their class and exit status."""
    environment = dict(os.environ, AZURE_CORE_COLLECT_TELEMETRY="0",
                       AZURE_LOGGING_ENABLE_LOG_FILE="false",
                       AZURE_CORE_ONLY_SHOW_ERRORS="true", LC_ALL="C")
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout, env=environment, check=False)
        if result.returncode:
            return {"ok": False, "error": "CommandFailed",
                    "exit_status": result.returncode}
        return {"ok": True, "value": parser(result.stdout)}
    except Exception as exc:
        return failure(exc)


def imds():
    request = urllib.request.Request(
        "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01",
        headers={"Metadata": "true"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:
        data = json.loads(response.read(65536))
    return {key: data.get(key) for key in ("name", "location", "vmSize", "osType")}


def cpu_details(text):
    allowed = {"Architecture:", "CPU(s):", "Model name:", "Thread(s) per core:",
               "Core(s) per socket:", "Socket(s):", "NUMA node(s):"}
    return {row["field"].rstrip(":"): row["data"]
            for row in json.loads(text)["lscpu"] if row["field"] in allowed}


def memory():
    with open("/proc/meminfo", encoding="utf-8") as source:
        return {line.split(":", 1)[0]: int(line.split()[1]) * 1024
                for line in source if line.startswith(("MemTotal:", "MemAvailable:"))}


def safe_account(text):
    data = json.loads(text)
    # Never include account, tenant, subscription, user, or token identifiers.
    return {"state": data.get("state") if data.get("state") in
            {"Enabled", "Warned", "PastDue", "Disabled", "Deleted"} else "Unknown",
            "is_default": data.get("isDefault") is True}


PROBE = r'''
import json, socket, sys, urllib.error, urllib.request
host = sys.argv[1]
report = {}
try:
    socket.getaddrinfo(host, 443)
    report["dns_ok"] = True
except Exception as exc:
    report.update(dns_ok=False, dns_error=type(exc).__name__)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request("https://" + host + "/", method="HEAD")
    with opener.open(request, timeout=4) as response:
        report["https_status"] = response.status
except urllib.error.HTTPError as exc:
    report["https_status"] = exc.code
except Exception as exc:
    report["https_error"] = type(exc).__name__
print(json.dumps(report))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-vm", default="hackathon")
    expected = parser.parse_args().expected_vm
    try:
        metadata = imds()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "AzureIdentityUnverified",
                          "detail": failure(exc)}))
        return 2
    if metadata["name"] != expected or metadata["osType"] != "Linux":
        print(json.dumps({"ok": False, "error": "UnexpectedAzureVM", "azure": metadata}))
        return 2

    report = {"ok": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "note": "Collector completed; inspect individual checks for failures.",
              "azure": metadata,
              "python": platform.python_version(), "kernel": platform.release(),
              "os": platform.freedesktop_os_release().get("PRETTY_NAME", "Linux"),
              "cpu_count": os.cpu_count(), "cpu_affinity_count": len(os.sched_getaffinity(0)),
              "load_average": os.getloadavg(), "memory_bytes": memory(),
              "lscpu": command(["lscpu", "--json"], cpu_details),
              "block_devices": command(["lsblk", "--json", "--bytes", "--output",
                                        "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS"], json.loads),
              "mounts": command(["findmnt", "--json", "--output", "TARGET,FSTYPE"], json.loads),
              "disk_bytes": command(["df", "-B1", "--output=target,size,used,avail,pcent"]),
              "disk_inodes": command(["df", "--output=target,itotal,iused,iavail,ipcent"])}
    report["azure_cli"] = {"installed": shutil.which("az") is not None}
    if report["azure_cli"]["installed"]:
        report["azure_cli"]["cached_account"] = command(
            ["az", "account", "show", "--query", "{state:state,isDefault:isDefault}",
             "--output", "json", "--only-show-errors"], safe_account)
        report["azure_cli"]["usable_auth"] = command(
            ["az", "account", "get-access-token", "--resource", "https://management.azure.com/",
             "--query", "expiresOn", "--output", "tsv", "--only-show-errors"],
            lambda text: bool(text.strip()))
    report["network"] = {host: command([sys.executable, "-c", PROBE, host], json.loads, timeout=5)
                         for host in ("huggingface.co", "pypi.org", "management.azure.com")}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(failure(exc)))
        raise SystemExit(1)
