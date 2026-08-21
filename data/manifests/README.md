# Split manifests

Manifests are canonical JSON documents with a SHA-256 digest. A session or path
may occur in exactly one of train/validation/test, and an existing manifest is
never overwritten with different content.

Generate the deterministic interface-validation manifest with:

```bash
python scripts/build_synthetic_manifest.py
```

The same schema will be used for real files, but real entries must preserve
unverified metadata explicitly rather than guessing it.

Manifests are immutable JSON files listing source-file checksum, robot, date,
session/path group, role (`train`, `validation`, `test`, `external_case`), and
exclusion reason. Create the split before fitting preprocessing or simulation
priors. Changing a split creates a new manifest version.
