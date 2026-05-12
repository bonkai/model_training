---
id: 2026-05-12-001
date: 2026-05-12
project: model-training
phase: phase-0-bootstrap
tags: [forge, scaffolding, setup, git, auto-push, retrieval, mlx]
status: completed
outcome: success
duration_min: 60
files_touched:
  - PLAN.md
  - 2B_Model_Development_Plan.md
  - tasks/2026-05-12-001-project-bootstrap.md
related_tasks: []
---

# Task: Bootstrap model_training as a forge project with cross-project task retrieval

## Context
Starting a new 2B-parameter language model training project on Apple Silicon (MLX). Wanted the project to be a proper forge project from day one — not "we'll add structure later." Also wanted task-level memory that persists across projects so that future work can search "have we hit X before?" and pull the actual lessons rather than having to recall from chat history.

## What happened
1. Audited existing infrastructure: confirmed `~/Documents/auto_push_git.sh` exists (launchd-driven, reads `~/Documents/.auto_push_dirs`); the forge template at `~/Documents/forge/template/` has a working scaffold but no task-recording schema.
2. Upgraded forge template to v1.1.0:
   - Added `tasks/` with README + `_schema.md` + `INDEX.md`.
   - Added `_templates/task.md`.
   - Updated template `CLAUDE.md` to instruct future-Claude to write task files for meaningful units of work and to run cross-project search before non-trivial work.
3. Patched `~/Documents/forge/create-project.sh`:
   - Allows scaffolding into an existing directory if it's flat (no subdirs) and has no filename conflicts. Preserves existing files.
   - Auto-runs `git init` + initial commit, including a default `.gitignore`.
   - Auto-registers the project with `~/Documents/forge/projects.txt`.
4. Ran the patched script against `~/Documents/model_training/` with the 2B plan markdown already present as seed. In-place scaffold succeeded, `2B_Model_Development_Plan.md` was preserved as the source-of-truth technical roadmap.
5. Rewrote `PLAN.md` to point at the 2B doc and list near-term milestones.
6. Registered the project with `~/Documents/.auto_push_dirs` so the daily auto-push cron picks it up.
7. Built `~/Documents/forge/bin/` retrieval scripts: `forge-search` (lexical, pure Python, no rg dependency), `forge-search-vec` (semantic via sentence-transformers all-MiniLM-L6-v2, lazy-installed), `forge-search-all` (runs both), `forge-projects` (registry CRUD).

## Hazards encountered
- `~/Documents/.auto_push_dirs` did NOT end in a newline, so a plain `echo >> file` concatenated the new entry to the previous line. Always check trailing-newline state before appending to config files, or use `printf '\n%s\n' "$path" >> file` defensively.
- The system `rg` is a shell function that proxies through Claude Code — not a real ripgrep binary. Standalone shell scripts that call `rg` would silently fail outside Claude Code. Used pure-Python search instead.
- `cp -R "$TEMPLATE" "$EXISTING_DIR"` puts the template *as a subdirectory* of the existing dir. To merge contents, use `cp -R "$TEMPLATE"/. "$EXISTING_DIR"/` (note the trailing `/.`).
- The model_training directory was inside `~/Documents/` which is itself inside a parent git repo (humann-shopify checkout sits adjacent and apparently traverses up). Running `git init` inside `model_training/` correctly isolated it as its own repo, but be aware that `git status` from this directory before `git init` would show the parent repo's state.
- No GitHub remote was created — auto-push will fail at the `git push` step until a remote is added. This is a TODO, not done yet.

## Learnings
- The cross-project task-search payoff comes from schema discipline. The `_schema.md` file is load-bearing; if individual projects' task files drift, retrieval scores drop. Worth re-validating periodically.
- Lazy-install pattern for ML deps (`sentence-transformers` is ~80MB) keeps the toolchain light — only pay the cost when semantic search is actually used.
- For "list of paths" config files, prefer the registry to be appended via a wrapper script (`forge-projects add`) rather than raw `echo >>` — handles trailing-newline gotchas and dedup.
- For new projects, in-place scaffolding into a seed directory beats "create new dir + move things" because it preserves the dir's git history (or lack thereof) and any IDE state.

## Result
model_training is now a properly scaffolded forge project. Git repo initialized, registered for auto-push, registered with the forge cross-project retrieval index, and the 2B plan is preserved as the source of truth alongside a near-term `PLAN.md`. The retrieval scripts (`forge-search`, `forge-search-vec`, `forge-search-all`) work against this and any future forge projects. Next concrete step (Phase 1 of the 2B plan): set up the MLX environment and observability stack.
