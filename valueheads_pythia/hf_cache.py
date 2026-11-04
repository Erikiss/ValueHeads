"""Hugging Face cache housekeeping (DESIGN 3.4 / 6.4, review findings F2 / F3).

  cleanup_hf_revision   delete one cached revision of a model repo (sweep --cleanup_cache; protected refs are kept)
  cleanup_hf_repo       delete every cached revision of a repo (judge --cleanup_judges: the 13.5 GB TruthfulQA judges,
                        15 GB ArmoRM and 1.5 GB Qwen3Guard do not stay next to the features on /content)
Both only touch the hub cache below `cache_dir` (default HF_HUB_CACHE derived from HF_HOME) and never delete datasets.
"""


def cleanup_hf_revision(hf_id, revision, cache_dir=None, dry_run=False, protect=("main",)):
    """Deletes the cached snapshot of `revision` of `hf_id` (model repo) unless its commit is shared with a protected
    ref. Returns a dict: deleted (bool), reason / commit / freed bytes."""
    from huggingface_hub import scan_cache_dir
    try:
        info = scan_cache_dir(cache_dir)
    except Exception as e:
        return {"deleted": False, "reason": f"cache not scannable: {e}"}
    for repo in info.repos:
        if repo.repo_id != hf_id or repo.repo_type != "model":
            continue
        for rev in repo.revisions:
            if revision in rev.refs or rev.commit_hash == revision:
                others = set(rev.refs) - {revision}
                if others & set(protect):
                    return {"deleted": False, "reason": f"commit shared with protected ref(s) {sorted(others & set(protect))}", "commit": rev.commit_hash}
                strategy = info.delete_revisions(rev.commit_hash)
                if not dry_run:
                    strategy.execute()
                return {"deleted": not dry_run, "dry_run": dry_run, "commit": rev.commit_hash, "freed_bytes": int(strategy.expected_freed_size),
                        "also_refs": sorted(others)}
        return {"deleted": False, "reason": f"revision {revision!r} not in cache"}
    return {"deleted": False, "reason": f"{hf_id} not in cache"}


def cleanup_hf_repo(repo_id, cache_dir=None, dry_run=False, repo_type="model"):
    """Deletes every cached revision of `repo_id` (model repos only by default). Returns a dict with deleted / freed bytes."""
    from huggingface_hub import scan_cache_dir
    try:
        info = scan_cache_dir(cache_dir)
    except Exception as e:
        return {"deleted": False, "reason": f"cache not scannable: {e}"}
    for repo in info.repos:
        if repo.repo_id != repo_id or repo.repo_type != repo_type:
            continue
        hashes = [rev.commit_hash for rev in repo.revisions]
        if not hashes:
            return {"deleted": False, "reason": f"{repo_id} has no cached revisions"}
        strategy = info.delete_revisions(*hashes)
        if not dry_run:
            strategy.execute()
        return {"deleted": not dry_run, "dry_run": dry_run, "repo": repo_id, "n_revisions": len(hashes),
                "freed_bytes": int(strategy.expected_freed_size)}
    return {"deleted": False, "reason": f"{repo_id} not in cache"}
