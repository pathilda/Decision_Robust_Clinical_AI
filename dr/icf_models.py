"""ICF profiles for one client file with several models, one model after another.

    python -m dr.icf_models --models qwen35-4b qwen35-27b medgemma-4b medgemma-27b \
        llama31-8b mistral-small-24b \
        --clients /home/pathilda/Anonymization/Labeled_Sample.xlsx \
        --out-dir /scratch/pathilda/icf_labelled --models-dir /scratch/pathilda/Models \
        --deidentified-confirmed

The client file needs an ID column and the service-plan and assessment text columns (names
set with --id-column, --sp-column, --assessment-column). Writes

    <out-dir>/data/chunks.xlsx, chunks_meta.json      (dr.chunks: which clients, which columns)
    <out-dir>/outputs/<model>/icf/icf.jsonl           (dr.icf: one record per client)

Each model runs in its own process (python -m dr.icf), so its GPU memory is released before
the next model loads. A model that fails does not stop the others. Rerunning the same command
resumes: clients with a successful profile from the current prompt are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import JsonlStore, Workspace, models_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, choices=sorted(models_config()["models"]))
    p.add_argument("--clients", type=Path, required=True, help="Excel file with the notes")
    p.add_argument("--sheet", default=None)
    p.add_argument("--id-column", default="Client AlayaCare Client ID")
    p.add_argument("--sp-column", default="SP text")
    p.add_argument("--assessment-column", default="assessment text")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--deidentified-confirmed", action="store_true",
                   help="Required: confirms the file is de-identified (notes go to local LLMs)")
    args = p.parse_args(argv)
    if not args.deidentified_confirmed:
        print("error: pass --deidentified-confirmed (the notes are sent to local LLMs)", file=sys.stderr)
        return 2

    missing = [m for m in args.models if not (args.models_dir / m / "config.json").is_file()]
    if missing:  # check every checkpoint up front, not hours into the run
        print(f"error: no checkpoint (config.json) in {args.models_dir} for: {', '.join(missing)}",
              file=sys.stderr)
        return 2

    data, outputs = args.out_dir / "data", args.out_dir / "outputs"
    chunk_cmd = [sys.executable, "-m", "dr.chunks", "--clients", str(args.clients),
                 "--id-column", args.id_column, "--sp-column", args.sp_column,
                 "--assessment-column", args.assessment_column,
                 "--data-dir", str(data), "--deidentified-confirmed"]
    if args.sheet is not None:
        chunk_cmd += ["--sheet", str(args.sheet)]
    print(">>> registering clients:", " ".join(chunk_cmd[2:]), flush=True)
    if subprocess.run(chunk_cmd, check=False).returncode != 0:
        print("error: dr.chunks failed; nothing was run", file=sys.stderr)
        return 1

    ws = Workspace(data, outputs)
    results = {}
    for model in args.models:
        cmd = [sys.executable, "-m", "dr.icf", "--model", model, "--models-dir", str(args.models_dir),
               "--data-dir", str(data), "--outputs-dir", str(outputs),
               "--batch-size", str(args.batch_size)]
        print(f"\n>>> ICF with {model}", flush=True)
        code = subprocess.run(cmd, check=False).returncode
        store = JsonlStore(ws.icf(model))
        ok = len(store.ok())
        results[model] = (code, ok, len(store.records))

    print("\nmodel                 exit  parsed/records")
    for model, (code, ok, total) in results.items():
        print(f"{model:<21} {code:>4}  {ok}/{total}")
    return 0 if all(code == 0 for code, _, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
