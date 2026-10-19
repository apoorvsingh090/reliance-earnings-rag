# Contributing

Thanks for helping improve this project.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

`requirements-dev.txt` is the full local set (includes `sentence-transformers`;
CPU torch is enough). CI installs the cloud-safe `requirements.txt` plus the
test deps with CPU-only torch.

## Golden rules (please preserve)

- Numbers live in the database; the vector store never holds authoritative
  financial figures.
- Calculations are Python, never LLM arithmetic.
- Standalone vs consolidated figures are never mixed.
- Every factual claim carries a citation with a real page / source location.

## Tests

- 67 tests, one file per milestone under `tests/`. New behaviour needs tests.
- Run `pytest` before opening a PR; CI runs the same suite.

## Pull requests

- Branch from `main`: `feature/<short-description>` or `fix/<short-description>`.
- Conventional commits (`feat:` / `fix:` / `docs:`), one logical change per commit.
- PR description: what changed, why, and how you verified it.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
