"""Torch-free job enumeration, sharding and the TEST-freeze guard shared by generate.py, verify_outputs.py and merge_samples.py."""
import json
from pathlib import Path

import config_schema


def sample_name(prompt_id, cfg, steps, seed):
    return f"{prompt_id}__cfg{cfg:g}__st{steps}__seed{seed}"


def enumerate_jobs(prompts, cfg_all, seeds=None):
    """Deterministic, ordered list of (prompt, cfg, steps, seed). Order = prompt-major, as in the pilot."""
    seeds = cfg_all["seeds"] if seeds is None else seeds
    return [(p, c, s, sd) for p in prompts for c in cfg_all["cfg_scales"] for s in cfg_all["steps"] for sd in seeds]


def parse_shard(text):
    i, n = (int(x) for x in text.split("/"))
    if not (0 <= i < n):
        raise ValueError("shard must be i/n with 0 <= i < n")
    return i, n


def shard_jobs(jobs, shard):
    """Round-robin by position in the full ordered list: shards are disjoint and their union is the full list."""
    if shard is None:
        return jobs
    i, n = shard
    return [j for k, j in enumerate(jobs) if k % n == i]


def guard_test_access(prompts, cfg_all):
    """D-06: any TEST prompt requires a frozen v2 config (pinned revision, split manifest, SHA-256 Addendum hash)."""
    if any(p.get("split") == "TEST" for p in prompts):
        config_schema.check_frozen(cfg_all)


def load_prompts(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
