"""tdes — Training Data Execution System (V5).

A small, fully-deterministic, dependency-light (numpy-only) pipeline that proves a
training-data system is correct, reproducible, and auditable:

    documents -> tokenized shards -> manifests -> mixture schedule -> packing ->
    batches -> training -> consumption ledger -> learning ledger -> checkpoint ->
    crash -> resume -> replay -> audit

Design invariant: every batch is a pure function of (seed, mixture schedule, step),
and every piece of evidence is computed from real hashes/ledgers at runtime.
"""
