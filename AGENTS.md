# model_training — Agent Instructions

## What This Project Is
2B parameter local AI language model trained natively on Apple Silicon via MLX

## Directory Structure

```
model-training/
  AGENTS.md              — You are here (canonical). CLAUDE.md/.cursorrules point here.
  PLAN.md                — Current plan and priorities
  PROGRESS.md            — Weekly progress and metrics

  knowledge/             — Accumulated learnings (READ BEFORE WORKING)
    MISTAKES.md          — Things that failed and why. Never repeat these.
    WINS.md              — Things that worked. Keep doing these.
    HAZARDS.md           — Known pitfalls and traps.
    DECISIONS.md         — Key decisions, reasoning, and outcomes.
    PRINCIPLES.md        — Rules for making decisions.
    TECH_STACK.md        — Tools, platforms, infrastructure.
    GRADUATED.md         — Universal lessons inherited from all past projects.

  products/              — Per-product/feature status and plans
    INDEX.md             — Dashboard: current status of everything

  sessions/              — Conversation summaries and decisions
    INDEX.md             — Chronological index

  tasks/                 — Permanent, searchable record of every meaningful unit of work
    README.md            — Read this first if you're unfamiliar with the task log
    _schema.md           — Frontmatter + body schema (DO NOT diverge from this)
    INDEX.md             — Human-readable index (one line per task)
    YYYY-MM-DD-NNN-*.md  — Individual task files

  retros/                — Post-launch/milestone retrospectives
    INDEX.md             — Retrospective index

  guides/                — How-to docs and checklists

  _templates/            — File templates for consistency
```

## Workflow — Every Conversation

### Starting a conversation:
1. Read this file (AGENTS.md)
2. Read `products/INDEX.md` for current status of everything
3. Read `sessions/INDEX.md` for recent session context
4. Read `knowledge/MISTAKES.md` if doing anything that could repeat a past error
5. Read the specific product file if working on a specific product

### During a conversation:
- When something fails → add to `knowledge/MISTAKES.md`
- When something works → add to `knowledge/WINS.md`
- When making a significant choice → add to `knowledge/DECISIONS.md`
- When a product status changes → update `products/INDEX.md`
- **When you finish a meaningful unit of work** → write a task file in `tasks/` using `_templates/task.md`. This is the permanent searchable record across all projects. See `tasks/README.md` for the "what counts" heuristics.

### Before starting non-trivial work:
- Run `forge-search-all "<topic or tag>"` (or just `rg -l "tag-name" ~/Documents/*/tasks/`) to see if we've done something similar before. The point is not to repeat past hazards.

### Ending a conversation:
1. Create/update `sessions/YYYY-MM-DD.md` with what was done and decided
2. Update `sessions/INDEX.md` with one-line summary
3. Update `PROGRESS.md` with completed items
4. Update any product files that changed
5. Make sure every meaningful unit of work has a `tasks/` entry (don't batch all into one file — one file per unit makes retrieval work)

### After a launch or milestone:
1. Create a retro in `retros/YYYY-MM-DD-name.md` using the retro template
2. Move retro findings into MISTAKES.md and WINS.md
3. If any lesson is universal (applies to ALL projects), add it to `knowledge/GRADUATED.md`
4. Copy that graduated lesson to the global template at `Documents/forge/template/knowledge/GRADUATED.md`

## Rules
Add project-specific rules here
