"""Paths, configuration, templates, hashing and resumable JSONL storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "config"
PROMPTS_DIR = REPO / "prompts"
SCHEMAS_DIR = REPO / "schemas"

MODEL_KEYS = ("qwen35-4b", "qwen35-27b", "medgemma-4b", "medgemma-27b")
PROFILE_IDS = ("balanced", "unbalanced", "highly_unbalanced")
RATING_OUTPUTS = ("label", "distribution", "score")
RUNGS = (1, 2, 3, 4, 5, 6)
ALLOCATION_RUNGS = (5, 6)
TIEBREAK_RUNGS = (4, 5, 6)
STEPS = ("icf", "rating", "allocation", "tiebreak")


# ----------------------------------------------------------------------------- workspace
class Workspace:
    """Explicit data / outputs / previews directories (always given on the command line).

    Config, prompts and schemas always come from the repository.
    """

    def __init__(self, data_dir=None, outputs_dir=None, previews_dir=None):
        self._dirs = {
            "data": Path(data_dir).resolve() if data_dir else None,
            "outputs": Path(outputs_dir).resolve() if outputs_dir else None,
            "previews": Path(previews_dir).resolve() if previews_dir else None,
        }

    @classmethod
    def under(cls, root):
        """All three directories below one root (tests and convenience)."""
        root = Path(root)
        return cls(root / "data", root / "outputs", root / "prompts_preview")

    @classmethod
    def from_args(cls, args):
        return cls(
            getattr(args, "data_dir", None),
            getattr(args, "outputs_dir", None),
            getattr(args, "previews_dir", None),
        )

    def _dir(self, name, flag):
        if self._dirs[name] is None:
            raise ValueError(f"This command needs {flag}")
        return self._dirs[name]

    @property
    def data(self):
        return self._dir("data", "--data-dir")

    @property
    def chunks_file(self):
        return self.data / "chunks.xlsx"

    @property
    def chunks_meta(self):
        return self.data / "chunks_meta.json"

    @property
    def outputs(self):
        return self._dir("outputs", "--outputs-dir")

    @property
    def excluded(self):
        return self.outputs / "excluded_clients.csv"

    @property
    def manifest(self):
        return self.outputs / "manifest.json"

    @property
    def previews(self):
        return self._dir("previews", "--previews-dir")

    def icf(self, model):
        return self.outputs / model / "icf" / "icf.jsonl"

    def rating(self, model, rung, output, profile=None):
        base = self.outputs / model / "rating" / f"rung{rung}" / output
        if rung >= 5:
            if profile is None:
                raise ValueError("Rungs 5 and 6 require a load profile")
            base = base / profile
        return base / "results.jsonl"

    def allocation(self, model, rung, profile):
        return self.outputs / model / "allocation" / f"rung{rung}" / profile / "results.jsonl"

    def candidates(self, model, rung, source, profile, chunk_id):
        return (
            self.outputs / model / "candidates" / f"rung{rung}" / source / profile
            / f"chunk{chunk_id}.json"
        )

    def tiebreak(self, model, rung, source, profile):
        return (
            self.outputs / model / "tiebreak" / f"rung{rung}" / source / profile / "results.jsonl"
        )


def add_workspace_argument(parser, data=True, outputs=True, previews=False):
    """Every directory a command reads or writes is an explicit, required argument."""
    if data:
        parser.add_argument("--data-dir", type=Path, required=True,
                            help="Holds chunks.xlsx and chunks_meta.json (written by dr.chunks)")
    if outputs:
        parser.add_argument("--outputs-dir", type=Path, required=True,
                            help="Holds every result: <model>/..., excluded_clients.csv, manifest.json")
    if previews:
        parser.add_argument("--previews-dir", type=Path, required=True,
                            help="Where rendered example prompts are written (contains clinical text)")


def add_models_dir_argument(parser, required=True):
    parser.add_argument("--models-dir", type=Path, required=required,
                        help="Holds one folder per model key (qwen35-4b, ...), each with REVISION")


def add_gurobi_argument(parser):
    parser.add_argument("--gurobi-license", type=Path, default=None,
                        help="gurobi.lic to use (sets GRB_LICENSE_FILE for this process)")


def apply_gurobi_license(args):
    path = getattr(args, "gurobi_license", None)
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Gurobi license not found: {path}")
        os.environ["GRB_LICENSE_FILE"] = str(Path(path).resolve())


# ----------------------------------------------------------------------------- config
def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


@cache
def classes_config():
    raw = read_yaml(CONFIG_DIR / "classes.yaml")
    classes = raw["classes"]
    ids = [int(c["id"]) for c in classes]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("classes.yaml: class ids must be unique and ascending")
    scores = [float(c["score"]) for c in classes]
    if scores != sorted(scores):
        raise ValueError("classes.yaml: class scores must increase with the class id")
    return {
        "ids": ids,
        "scores": {int(c["id"]): float(c["score"]) for c in classes},
        "definitions": {int(c["id"]): list(c["definition"]) for c in classes},
        "therapy_plans": {int(c["id"]): c.get("typical_therapy_plan") for c in classes},
        "min": min(scores),
        "max": max(scores),
        "sha256": file_sha256(CONFIG_DIR / "classes.yaml"),
    }


@cache
def models_config():
    return read_yaml(CONFIG_DIR / "models.yaml")


@cache
def study_config():
    return read_yaml(CONFIG_DIR / "study.yaml")


def model_entry(key, models_dir=None):
    """Model config; ``path`` = <models_dir>/<key> (None when no models_dir is given)."""
    models = models_config()["models"]
    if key not in models:
        raise ValueError(f"Unknown model {key!r}; choose from {sorted(models)}")
    entry = dict(models[key])
    entry["key"] = key
    entry["path"] = (Path(models_dir).expanduser().resolve() / key) if models_dir else None
    return entry


def model_revision(entry):
    path = Path(entry["path"]) / "REVISION"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unknown"


def format_score(value):
    """1.0 -> '1.0', 1.2 -> '1.2', 14 -> '14.0', 1.25 -> '1.25'."""
    text = f"{float(value):.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


# ----------------------------------------------------------------------------- templates
@cache
def template(name):
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")


def prompt_version():
    return (PROMPTS_DIR / "VERSION").read_text(encoding="utf-8").strip()


def prompt_hashes():
    return {p.name: file_sha256(p) for p in sorted(PROMPTS_DIR.glob("*.txt"))}


_PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


def fill(text, **values):
    """Replace {UPPER_CASE} placeholders only; JSON braces in templates are left intact."""
    found = set(_PLACEHOLDER.findall(text))
    missing = found - set(values)
    if missing:
        raise KeyError(f"Unfilled template placeholders: {sorted(missing)}")
    unused = set(values) - found
    if unused:
        raise KeyError(f"Values for placeholders absent from template: {sorted(unused)}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


# ----------------------------------------------------------------------------- hashing
def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def derived_seed(*parts):
    return int(hashlib.sha256(stable_json(list(parts)).encode()).hexdigest()[:16], 16)


def now():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- JSONL store
class JsonlStore:
    """Append-only JSONL. The last record per key wins; truncated trailing lines are ignored."""

    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # partial line from an interrupted write
                    self.records[record["key"]] = record

    def done(self, key, prompt_hash=None):
        """Resume rule: skip a key only if it already succeeded with the same prompt."""
        record = self.records.get(key)
        return bool(
            record
            and record.get("parse_ok")
            and (prompt_hash is None or record.get("prompt_hash") == prompt_hash)
        )

    def get(self, key):
        return self.records.get(key)

    def ok(self):
        return {k: r for k, r in self.records.items() if r.get("parse_ok")}

    def append(self, records):
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self.records[record["key"]] = record
            handle.flush()
            os.fsync(handle.fileno())


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)
