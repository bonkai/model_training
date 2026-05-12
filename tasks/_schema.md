# Task file schema

Every file in `tasks/` (except `INDEX.md`, `README.md`, and files starting with `_`) MUST match this schema. Retrieval scripts in `~/Documents/forge/bin/` parse this format — divergence will silently break search.

## Filename
`YYYY-MM-DD-NNN-short-kebab-slug.md`

- `YYYY-MM-DD` — date the task was started.
- `NNN` — zero-padded sequence number for that date (001, 002, ...).
- `short-kebab-slug` — 2–5 words, lowercased, hyphenated. Describes the *outcome*, not the activity ("rope-bake-stability-fix", not "fixing-stuff").

## Frontmatter (YAML)

```yaml
---
id: 2026-05-12-001
date: 2026-05-12
project: model-training
phase: phase-1-lab-setup     # free-form phase/area within the project
tags: [mlx, environment, setup, uv]
status: completed             # in_progress | completed | abandoned
outcome: success              # success | partial | fail | unknown
duration_min: 45              # rough wall-clock minutes; null if unknown
files_touched:
  - path/relative/to/project/root.py
related_tasks: []             # IDs of related task files (cross-project allowed)
---
```

### Field rules
- **id** — must equal the filename minus the slug. e.g. `2026-05-12-001`.
- **tags** — lowercase, kebab-case, 1–8 items. These drive ripgrep retrieval. Think "what would future-me grep for?"
- **status** — `in_progress` is allowed if a task spans sessions; flip to `completed` or `abandoned` when done. Tasks left at `in_progress` for >7 days are stale.
- **outcome** — only meaningful when `status` is `completed` or `abandoned`. `partial` = some goals hit, others not. `fail` = approach didn't work.
- **files_touched** — relative paths. Helps locate the task via file-based search later.

## Body sections (Markdown)
All sections are required. Use empty bullet lists if a section truly has nothing — never delete the section heading.

```markdown
# Task: <Short imperative title, sentence case>

## Context
2–4 sentences. What were we trying to do, and *why*. Include the user-visible motivation if any.

## What happened
Concrete actions, in order. Commands, decisions, surprises. Past tense.
This is the narrative section — write enough that a future engineer can reconstruct what was tried.

## Hazards encountered
One bullet per hazard. Each bullet is grep-bait — make the key terms findable.
- 

## Learnings
One bullet per learning. Each bullet must be self-contained — readable out of context.
- 

## Result
One paragraph: did this work? what's left? what was the user-visible outcome?
```
