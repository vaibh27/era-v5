"""Canonical output locations. Everything the grader inspects lives under submission_artifacts/.

All of it is regenerated from frozen inputs (assets/tokenizer.json, assets/corpus.jsonl)
by run_demo.py, so the whole tree is reproducible.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
ART = ROOT / "submission_artifacts"

SHARDS = ART / "shards"          # internal token .bin shards (kept for replay/audit)
MANIFESTS = ART / "manifests"    # one JSON per shard
LEDGERS = ART / "ledgers"        # consumption + learning ledgers (jsonl)
CHECKPOINTS = ART / "checkpoints"

RUN_LOG = ART / "run.log"
EVIDENCE_JSON = ART / "evidence.json"
EVIDENCE_MD = ART / "evidence.md"
PERFORMANCE_JSON = ART / "performance.json"

CORPUS = Path(__file__).parent / "assets" / "corpus.jsonl"


def ensure_dirs():
    for d in (ART, SHARDS, MANIFESTS, LEDGERS, CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)


def use_artifacts_dir(path):
    """Rebind every output location under `path`. run_pipeline.py uses this so its
    resumable, kill-and-restart runs write to their own tree and never disturb the graded
    submission_artifacts/. All modules read these as `paths.<NAME>` at call time, so the
    rebind is picked up everywhere. run_demo.py keeps the default."""
    global ART, SHARDS, MANIFESTS, LEDGERS, CHECKPOINTS
    global RUN_LOG, EVIDENCE_JSON, EVIDENCE_MD, PERFORMANCE_JSON
    ART = Path(path)
    SHARDS, MANIFESTS, LEDGERS, CHECKPOINTS = (
        ART / "shards", ART / "manifests", ART / "ledgers", ART / "checkpoints")
    RUN_LOG = ART / "run.log"
    EVIDENCE_JSON, EVIDENCE_MD = ART / "evidence.json", ART / "evidence.md"
    PERFORMANCE_JSON = ART / "performance.json"
    ensure_dirs()
    return ART
