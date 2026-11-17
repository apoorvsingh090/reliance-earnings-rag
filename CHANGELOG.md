# Changelog

All notable changes to Reliance Earnings RAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Actions CI
- CodeQL code scanning (Python)
- Community files: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CHANGELOG.md`
- Issue forms (bug report, feature request) and pull request template
- `dependabot.yml` (weekly grouped GitHub Actions + pip updates)
- README status badges and contributing/security sections

### Fixed

- Registry `local_path` values are stored with POSIX separators and
  resolved defensively, so re-extraction/re-indexing works on Linux
  hosts (Streamlit Cloud) instead of only Windows
- Uploaded files are capped at 50 MB and the temp upload file is removed
  even when validation fails; an upload colliding with a different
  already-registered filename is refused instead of silently ignored
- `/search` and `/ask` clamp `k` to 1–50
- Dashboard YoY deltas use the calculation engine's growth-vs-magnitude
  convention, so a narrowing loss reads positive as in `/calc` (equal
  to the displayed decimal)
