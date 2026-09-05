# MicroPerco 1.0.0 release readiness

Assessment date: 2026-09-04. Scope: GitHub source publication and reproducible
release-artifact preparation. Overall project verdict: **READY FOR GITHUB
PUBLICATION**.

## Release gates

- [PASS] Source repository remained read-only to the MicroPerco rebuild. The
  deterministic inventory still contains 292 files; 291 are byte-identical to
  the baseline. The sole digest change is a separately user-authorized edit to
  one source-workspace documentation file, recorded below and in
  `docs/provenance/source_snapshot.md`.
- [PASS] Package builds. `python -m build --no-isolation` produced both the
  `microperco-1.0.0` sdist and universal wheel; Twine 7.0.0 accepted the
  metadata and rendered README for both artifacts.
- [PASS] Wheel installs. A newly created virtual environment resolved only the
  declared runtime dependencies and installed the built wheel successfully.
- [PASS] Tests pass. All 303 tests passed on CPython 3.10.12, 3.11.15, and
  3.12.14.
- [PASS] Lint passes. Ruff 0.16.1 reported no findings.
- [PASS] Type checks pass. mypy 1.20.2 strict mode passed all 43 package source
  files.
- [PASS] CLI works. The installed `microperco --version`, `--help`, and
  `validate` commands succeeded; the built-in validation returned
  `"passed": true`.
- [PASS] README example works. Both executable README tests passed, and the
  installed-wheel smoke example completed successfully.
- [PASS] License included. The complete Apache License 2.0 text, NOTICE, and
  package metadata are present; the wheel contains LICENSE and NOTICE.
- [PASS] Third-party license audit complete. No third-party source or restricted
  source attachments are redistributed.
- [PASS] No secrets. A final repository scan found no tokens, private keys,
  credential assignments, passwords, or personal email addresses.
- [PASS] No absolute local paths. Public source, documentation, configuration,
  and examples contain no machine-specific home-directory dependency.
- [PASS] Examples run. The smoke, mixed-particle, critical-loading,
  inverse-design, and data-free modeling case-study scripts all exited
  successfully.
- [PASS] CI configured. GitHub Actions tests Ubuntu and Windows on Python
  3.10–3.12, runs Ruff, mypy, pytest, and builds distributions. The tag workflow
  builds, checks, smoke-tests, and uploads release artifacts.

## Numerical and scientific validation

- Built-in deterministic validation: PASS.
- Flat-cylinder distance versus SciPy SLSQP: 24 cases, maximum absolute
  difference `6.731e-10`, with 120/120 fixed-threshold decisions agreeing.
- Flat-cylinder distance versus HPP-FCL 2.4.4: 24 cases, maximum absolute
  difference `2.000e-5`, with 120/120 fixed-threshold decisions agreeing.
- Cell-list versus brute-force search: 24/24 complete accepted-edge sets
  identical; 80 versus 5,672 exact distance evaluations in the recorded sparse
  systems.
- Union-Find versus BFS: 15/15 face-to-face and 9/9 periodic-wrapping decisions
  agreed.
- Final figures: all alignment and text-size gates passed. Two 3D fill-edge
  warnings were reviewed at final size and accepted because the labels remain
  unobstructed and inside the canvas.

## Source-workspace integrity note

The source workspace lacked usable Git metadata, so the release used a
deterministic SHA-256 manifest. Its baseline digest was
`485e9b42c260d525c7b3ff6cb26319a10d354c0d36cc329d9f08c29ba0c53676`; the
final digest is
`1ffbd72953efda321ddfcc3463d09c0f500b69c958b259a0f136d5ff03f8295f`.
The complete difference is one independently authorized documentation edit to
`project/docs/Q1_Q4_PSEUDOCODE.md`; replacing that one final file digest with
its baseline digest reproduces the aggregate baseline exactly. No source code
or data changed, and the MicroPerco rebuild did not write to the source
workspace. This is a documented WARN on whole-workspace equality, not a
release blocker.

## Accepted v1.0 limitations

- Geometry uses IEEE-754 binary64. A severely ill-conditioned valid support-map
  query can still raise `GeometryError` if neither GJK nor the certified convex
  fallback closes a global distance bracket to the configured tolerance. It
  will not silently return an uncertified decision.
- Defensive caps reject more than 1,000,000 lattice images and implicit loading
  grids larger than 100,000 entries before pathological allocation or
  iteration.
- The built-in generator permits particle overlap and is not an equilibrated
  hard-particle packing model.
- `CERTIFIED_OPTIMAL` is limited to the declared finite integer search space and
  stochastic model. Statistical coverage assumes the declared Bernoulli model
  and independent certification streams.
- The recorded benchmark is one sparse periodic sphere workload; its observed
  speedup is not a universal performance guarantee.
- v1.0 does not implement ellipsoids, resistor networks, effective
  conductivity, GPU kernels, or parallel Monte Carlo.
- PyPI project-name availability and Trusted Publishing are publication-time
  concerns. The GitHub release workflow intentionally prepares artifacts but
  does not publish to PyPI without a maintainer-approved protected job.

None of these accepted limitations blocks normal documented v1.0 workflows.
