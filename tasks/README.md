# tasks/ — Permanent record of every meaningful unit of work

Every meaningful unit of work in this project gets one file in this directory.

## Why this exists
- Future-you (and future-Claude) can search across all projects for "did we ever do X?" and pull the hazards, fixes, and dead-ends in seconds.
- Wins and mistakes don't have to be re-derived every time we touch similar code.
- Cross-project retrieval works because every project follows the same schema.

## What to log
A "task" is any unit of work big enough that future-you would want to find it again. Heuristics:
- Took more than ~15 min of real attention.
- Hit a hazard, a surprise, or a non-obvious decision.
- Produced a result (good or bad) that someone might want to reproduce or avoid.

Trivial typo fixes and one-line obvious edits do **not** need entries. Quality > quantity.

## How to log
1. Copy `../_templates/task.md` to `tasks/YYYY-MM-DD-NNN-short-slug.md`.
2. Fill in frontmatter — tags matter most for retrieval.
3. Write the body in plain language. Pretend a future engineer with no context is reading.
4. Add a one-line entry to `INDEX.md`.

The schema is defined in `_schema.md`. Do not invent new frontmatter keys without updating that file — retrieval scripts depend on it.
