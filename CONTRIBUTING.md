# Contributing to OrchidRec

Thank you for helping improve OrchidRec. The project values small,
well-explained changes that preserve its dependency-free and reproducible
design.

## Development setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

No third-party runtime or test package is required.

## Change guidelines

- Keep runtime code within the Python standard library.
- Add type annotations to public and internal functions.
- Validate at the public boundary and raise an exception from
  `orchidrec.errors` with an actionable message.
- Never rely on set/dict/hash iteration for user-visible ordering.
- Use local seeded random generators; do not mutate the process-global random
  state.
- Preserve the versioned JSON model envelope. A breaking state change requires
  a schema-version increment and migration notes.
- Add focused tests for success, failure, and determinism where applicable.
- Do not commit generated demo artifacts, virtual environments, or build
  outputs.

## Verification

Before requesting review, run:

```bash
python -m unittest discover -s tests -v
coverage run -m unittest discover -s tests && coverage report
python -m compileall -q src tests examples
python -m orchidrec demo --output-dir artifacts/contributor-smoke
ruff check src tests examples
mypy src
python -m build
```

If behavior or configuration changes, update the README and checked-in example
in the same change.

## Commit and review notes

Describe the user-visible outcome, the design decision, and the exact commands
you ran. Keep refactors separate from behavioral changes where practical. If
tools assisted with substantial drafting or implementation, disclose that in
the contribution and remain able to explain and maintain every changed line.

By contributing, you agree that your contribution may be distributed under
the project's MIT License.
