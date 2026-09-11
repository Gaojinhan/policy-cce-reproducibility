# Reproducibility repository

- `cmfg_cce/` is a 105-file byte-exact frozen scientific copy. Never edit it to make a test, demo or output pass. `metadata/source-copy-manifest.json` checks every file.
- Implement reader/demo/package changes in `policy_cce_repro/` and test them separately.
- Original archived jobs, q, seeds, results, provenance and hashes must not change. Unknown physical executor provenance stays unknown.
- `paper`, `audits`, `outcomes`, `verify` and `unpack` are local-only. No cloud fallback. `demo` uses a separate bounded synthetic configuration and never replaces paper evidence.
- Require new output directories, preserve failures, and never overwrite release assets or published tags. Use a new version for changed content.
- Structural/architecture diagrams are outside the reproduction scope. Do not copy manuscript assets or review letters into this repository.
- The authors have approved public GitHub release under MIT. Preserve LICENSE, NOTICE.md and third-party notices. Do not publish unrelated research-tree contents or credentials.
- Pull before editing an existing clean checkout. Preserve concurrent changes; do not reset, force-push or silently stash user work.
