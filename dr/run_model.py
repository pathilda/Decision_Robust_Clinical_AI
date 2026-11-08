"""One SLURM job = one model loaded once; run every configuration whose inputs exist.

    python -m dr.run_model --model qwen35-27b --steps icf
    python -m dr.run_model --model qwen35-27b                       # every step, in order
    python -m dr.run_model --model qwen35-4b --chunk-ids 1 2        # pilot (step 8)

Order: icf -> rating -> allocation -> candidates -> tiebreak. Steps after icf need
outputs/excluded_clients.csv built with this model (python -m dr.exclusions); a step is skipped
with a message when its inputs do not exist yet.
"""

from __future__ import annotations

import argparse
import sys

from .common import (
    ALLOCATION_RUNGS,
    PROFILE_IDS,
    RATING_OUTPUTS,
    RUNGS,
    TIEBREAK_RUNGS,
    Workspace,
    add_gurobi_argument,
    add_models_dir_argument,
    add_workspace_argument,
    apply_gurobi_license,
    model_entry,
    model_revision,
)

STEP_ORDER = ("icf", "rating", "allocation", "candidates", "tiebreak")


def open_engine(ws, model, models_dir=None):
    from . import manifest
    from .engine import make_engine

    entry = model_entry(model, models_dir)
    if entry["path"] is None:
        raise ValueError("LLM steps need --models-dir")
    revision = model_revision(entry)
    if revision == "unknown":
        print(f"warning: {entry['path']}/REVISION missing; revision recorded as 'unknown'",
              file=sys.stderr)
    engine = make_engine(entry, revision)
    manifest.update(ws, model, engine.info())
    return engine, revision


class LazyEngine:
    def __init__(self, ws, model, models_dir):
        self.args = (ws, model, models_dir)
        self.engine = self.revision = None

    def get(self):
        if self.engine is None:
            self.engine, self.revision = open_engine(*self.args)
        return self.engine, self.revision

    def close(self):
        if self.engine is not None:
            self.engine.close()


def inputs_ready(ws, model):
    from .context import cohorts

    try:
        cohorts(ws, model)
        return True, None
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    add_models_dir_argument(p, required=False)  # required by every LLM step
    add_gurobi_argument(p)
    p.add_argument("--steps", nargs="+", choices=STEP_ORDER, default=list(STEP_ORDER))
    p.add_argument("--rungs", type=int, nargs="+", choices=RUNGS, default=list(RUNGS))
    p.add_argument("--outputs", nargs="+", choices=RATING_OUTPUTS, default=list(RATING_OUTPUTS))
    p.add_argument("--profiles", nargs="+", choices=PROFILE_IDS, default=list(PROFILE_IDS))
    p.add_argument("--chunk-ids", nargs="+", default=None, help="Pilot: restrict to these chunks")
    p.add_argument("--max-clients", type=int, default=None, help="ICF pilot: first N clients")
    p.add_argument("--batch-size", type=int, default=512)
    add_workspace_argument(p)
    args = p.parse_args(argv)
    apply_gurobi_license(args)
    model_entry(args.model, args.models_dir)
    ws = Workspace.from_args(args)
    lazy = LazyEngine(ws, args.model, args.models_dir)
    steps = [s for s in STEP_ORDER if s in args.steps]
    status = 0
    try:
        for step in steps:
            if step == "icf":
                from . import icf

                icf.run(ws, args.model, *lazy.get(), max_clients=args.max_clients,
                        batch_size=args.batch_size)
                continue
            ready, reason = inputs_ready(ws, args.model)
            if not ready:
                print(f"skip {step}: {reason}", file=sys.stderr)
                continue
            if step == "rating":
                from . import rate

                rate.run(ws, args.model, *lazy.get(), args.rungs, args.outputs, args.profiles,
                         args.chunk_ids, args.batch_size)
            elif step == "allocation":
                from . import allocate

                rungs = [r for r in args.rungs if r in ALLOCATION_RUNGS]
                if rungs:
                    allocate.run(ws, args.model, *lazy.get(), rungs, args.profiles,
                                 args.chunk_ids, args.batch_size)
            elif step == "candidates":
                from . import candidates

                # Pools for every requested rung: rungs 4-6 feed tie-breaks; all rungs feed
                # the OR evaluation of ratings (Section 10). No LLM is needed.
                counts = candidates.run(ws, args.model, args.rungs, args.outputs, args.profiles,
                                        args.chunk_ids)
                status = status or (1 if counts["failed"] else 0)
            elif step == "tiebreak":
                from . import tiebreak

                rungs = [r for r in args.rungs if r in TIEBREAK_RUNGS]
                if rungs:
                    tiebreak.run(ws, args.model, *lazy.get(), rungs, args.outputs, args.profiles,
                                 args.chunk_ids, args.batch_size)
    finally:
        lazy.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
