# Third-party resources

Do not copy the legacy `hxa` binary bundle here until ownership and redistribution
rights are confirmed. The legacy Linux `.so` currently depends on Qt 5.12 and
old ICU libraries that are not satisfied on the cluster, so it is not a portable
Phase-1 dependency.

If KUKA is selected, create a small adapter with:

- source/provenance and license note;
- exact supported robot IDs;
- expected angle/length units and frame convention;
- a subprocess or ctypes boundary;
- FK/IK round-trip and known-pose tests;
- a reproducible runtime container or dependency specification.

The UR5 DH implementation should be reimplemented and tested in `src/` rather
than importing the legacy module, which has unrelated side effects and binary
imports.

