# Source snapshot and integrity record

MicroPerco 1.0.0 was rebuilt beside a read-only source workspace. The source workspace was never used as a build destination and no command intentionally wrote into it.

The source directory did not contain a usable Git repository: its `.git` entry was empty, so neither a commit identifier nor `git status` could provide an integrity baseline. A deterministic substitute baseline was therefore recorded before implementation:

- Inventory: 292 regular files, excluding `.git`, `.agents`, and `.codex` control paths.
- Algorithm: sort repository-relative paths bytewise; hash each file with SHA-256; hash the resulting manifest with SHA-256.
- Baseline manifest digest: `485e9b42c260d525c7b3ff6cb26319a10d354c0d36cc329d9f08c29ba0c53676`.
- Final inventory: 292 regular files.
- Final manifest digest: `1ffbd72953efda321ddfcc3463d09c0f500b69c958b259a0f136d5ff03f8295f`.

The aggregate digest changed during the rebuild because one documentation file,
`project/docs/Q1_Q4_PSEUDOCODE.md`, was edited in a separate user-authorized
session. Its baseline file digest was
`cb43bf29cd633d686f1cf088a9e2f1309d6fb63614eac2505f6f53feddc6bb82`; its final
digest is `d2e737d59b8b57d3ae369e5d770ae242eefa1342b549def68367e17b28738eac`.
Substituting the baseline digest for that single manifest entry reproduces the
recorded aggregate baseline exactly. The other 291 files are byte-identical,
and no source code or data changed.

Source-repository integrity is therefore **PASS for rebuild isolation**, with a
**WARN for the independently authorized concurrent documentation edit**. The
MicroPerco rebuild did not write to the source workspace and did not attempt to
undo the unrelated change.

The source workspace contained a competition-oriented scientific project, tests, generated reports, an XLSX attachment, and a PDF problem statement. The attachment and PDF are not redistributed because their public redistribution rights could not be established.

No source code was copied. The original workspace did not expose a software license, so the new Apache-2.0 implementation is a clean-room, behavior-level reimplementation informed by algorithm and test-semantics audit only.
