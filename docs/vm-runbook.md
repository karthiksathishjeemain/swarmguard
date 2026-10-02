# Azure VM runbook

Diagnostics were executed on the Azure VM on 2026-10-02. Never print or copy private key contents or authentication tokens. Keep the 177 GB corpus, derived partitions, model weights, and benchmarks on the VM.

## Verified access and resources

The user's supplied SSH connection works with strict host-key checking. Azure instance metadata identifies `hackathon` in `eastus2` as `Standard_E64ads_v7`. Ubuntu 24.04.4 LTS provides Python 3.12.3 and 64 available vCPUs. RAM totals 540,821,135,360 bytes (about 504 GiB). The root volume has roughly 27–28 GB free. The VM has outbound HTTPS to Hugging Face and PyPI; Azure CLI is absent. The Mac's Azure CLI cache is present, but a management request failed with an Entra authentication error, so it is not a usable path for this work.

Four Azure local NVMe devices are 880 GiB each. `scripts/setup_scratch.sh` verified that `/dev/disk/azure/local/by-index/1` had no filesystem signatures, formatted that single device as ext4, and mounted it at `/mnt/ai-swarm-scratch` with 866 GiB free. The other three devices are untouched. Local temporary storage is disposable across VM lifecycle events. Keep code, source manifests, experiment logs, and final reports under `/home/azureuser/ai-swarm` on persistent storage. Use the scratch mount only for replaceable data caches and temporary output.

The VM Python environment is `/home/azureuser/ai-swarm/.venv`. `python3.12-venv`, Polars, DuckDB, PyArrow, psutil, and Laya 0.3.23 were installed on the VM. PyTorch 2.14.1+cpu came from the [official PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu); the other pinned package versions are in `requirements.lock.txt`. For a fresh Ubuntu image, install `python3.12-venv`, mount scratch, then run `scripts/bootstrap_vm.sh` from a project copy on the VM. `pip check` currently reports no dependency conflicts. The Laya English checkpoint is pinned to `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and cached under `/mnt/ai-swarm-scratch/hf-hub-cache`.

A source/documentation copy lives at `/home/azureuser/ai-swarm/project`. Local edits should be synchronized there; raw records and cache files must never be copied back to the Mac.

The bounded synthetic smoke command is:

```sh
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/scripts/laya_cpu_smoke.py \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --threads 8 --repeat 3
```

It completed in 4.865 seconds, including first-use model download/loading; the two warm calls together took 0.185 seconds, with peak process RSS 2,141,335,552 bytes. These synthetic timings are not an estimate of real episode throughput. Laya emitted a warning about invalid checkpoint temperature values, so confidence from affected entries is uncalibrated.

To repeat the VM diagnostic from this project directory, run:

```sh
ssh -i /path/to/your-private-key.pem azureuser@<VM_PUBLIC_IP> \
  'python3 - --expected-vm hackathon' < scripts/vm_preflight.py
```

The collector requires Linux and matching Azure instance metadata before hardware/auth/network diagnostics. It does not fetch data or credentials.

## Dataset access

No dataset copy was initially found under `/home/azureuser`, `/mnt`, `/data`, `/scratch`, or `/mnt/resource`. The gated Hugging Face repository is `aidigestorg/ai-village`, pinned for analysis to revision `838b4150303ca8228e8edb432d8b8ccae353d258`. Public metadata lists 390 files. The user completed interactive Hugging Face login on the VM; the pinned schema documents and named text tables now download successfully.

The VM-side login was completed privately; tokens remain in Hugging Face's credential store on the VM. Do not paste a token into chat or store it in this repository. `SCHEMA.md`, `CHANGELOG.md`, and `manifest.json` were fetched first. Named compressed tables were then downloaded to `/mnt/ai-swarm-scratch/raw` with SHA-256 checks against the published upstream hashes. `HF_HUB_CACHE` and `HF_XET_CACHE` point under scratch while the default credential location remains unchanged. Never run an unrestricted repository snapshot download, which would fetch daily screenshot archives.

`scripts/fetch_dataset_docs.py` and `scripts/fetch_text_files.py` pin the revision, allow explicit named files, check available scratch, and write a download manifest under persistent VM storage. The measured export details and chosen direction are in [dataset-audit.md](dataset-audit.md). The current pipeline projects chat, sessions, and turns to Parquet under scratch; all aggregate reports and progress logs remain under `/home/azureuser/ai-swarm`.

## Reproduce and review

The frozen experiment configuration is `config/experiment.toml`. From the VM, after scratch is mounted and the Python environment exists, run:

```sh
bash /home/azureuser/ai-swarm/project/scripts/run_pipeline.sh
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/investigate.py check
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/investigate.py progress
```

The pipeline is idempotent for existing derived files and takes an exclusive lock. On a fresh VM scratch disk it uses the existing VM-side Hugging Face login to fetch only the three text tables needed for the investigator. It projects them with PyArrow, joins sends to chat with DuckDB, builds 20-action windows, ranks candidates, freezes the blinded 300-case queue, runs Laya twice per case with reversed answer options, and writes aggregate metrics. It never requests screenshot TARs. Rebuilding after a changed configuration requires clearing the affected derived files on scratch; preserve the persistent manifests and annotations and do not compare results from different configurations as if identical.

In an **authorized interactive SSH terminal on the VM**, inspect a case or label the next unreviewed case:

```sh
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/investigate.py case 1
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/investigate.py review
```

These commands display raw corpus text; keep their output on Azure and do not screen-share or paste cases without separate review. The `review` view hides rank, sampling arm, and Laya result. Labels are append-only under `/home/azureuser/ai-swarm/restricted`, on persistent VM storage. `scripts/summarize_reviews.py` recomputes aggregate precision/yield only for actually reviewed cases. As of 2026-10-02, zero of 300 full cases have been adjudicated; no claim precision or contradiction yield is reported.

The 1.4 GiB derived Parquet directory can be checkpointed **on the VM** to its 30 GB persistent root disk before deallocation. This is separate from the 5.0 GiB raw text cache and does not move either to the Mac:

```sh
/home/azureuser/ai-swarm/.venv/bin/python /home/azureuser/ai-swarm/project/scripts/checkpoint_derived.py save
```

If the local NVMe scratch is lost, remount it safely with the scratch setup procedure, then run `checkpoint_derived.py restore` before `run_pipeline.sh`. The restore command verifies SHA-256 hashes and refuses to overwrite a differing scratch artifact. The persistent checkpoint includes derived message/evidence text, so treat it as restricted corpus data. Deallocate through the Azure portal or a working Azure management client when you are finished using the VM; a guest `shutdown` alone can leave compute charges running. Deallocated VMs avoid compute charges but still incur disk charges. [Microsoft Azure FinOps guidance](https://learn.microsoft.com/en-us/cloud-computing/finops/best-practices/compute)

Keep at least the greater of 50 GiB or 20% of scratch capacity free. Treat Laya as a frozen secondary component: its checkpoint warns about invalid temperature entries, and real-data pilots showed unstable labels and missed completed claims. Do not report confidence as calibrated or model outputs as misalignment ground truth. The [aggregate findings](aggregate-findings.md) contain the public-safe results so far.

## Completion evidence

Record a redacted JSON report with timestamps, VM identity, measured hardware and storage, Python/CLI status, network results, and persistent/scratch paths. Save dataset manifest, schema observations, quality counts, and benchmark summaries on Azure. No report should contain account credentials, SAS URLs, tokens, private keys, or unreviewed corpus excerpts.
