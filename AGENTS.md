# autodock — Agent skills

## Agent skills

### Issue tracker

GitHub Issues in this repo (uses `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.

### Git strategy

**Root repo (this one)**
- Minimal effective commits — each commit is a coherent unit of work
- Regular `git push origin main` to keep CI green
- `git add autodock/` always — never `git add -A` (benchmark data dirs like `benchmark_20target_final/`, `min_test_failures/` are hundreds of MB and cause HTTP 400 on push)

**`local_knowledge/` (independent git)**
- Has its own `.git` — not a submodule, not tracked by root repo
- `.gitignore` at root excludes `local_knowledge/` entirely
- **Do NOT** `git add -f local_knowledge/` or push it to GitHub
- Content stays local: session notes, gotchas, agent guidance
- The nested `.git` was removed during 2026-05-29 session; `.gitignore` exclusion is now the sole protection mechanism
