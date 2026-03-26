# Prompt Assets (`src/prompts`)

This directory stores all prompt assets as YAML files.

## Layout

- `assets/llm/instruction_parsing/`: all LLM instruction-parsing prompts (single "hat" namespace).
- `assets/vlm/`: VLM judging/grounding/reasoning prompts.
- `assets/vlm/assessment/<assessment_type>/<version>.yaml`: prompts used by `pair-judge` / `viescore` flows.
- `scripts/migrate_legacy_prompt_base.py`: one-shot migration script from legacy `prompt_base/*.py` constants.

## Naming

- Folder names represent task usage (`grounding`, `human_editing`, `assessment_type`, etc.).
- Files use semantic versions: `v1.yaml`, `v2.yaml`, ...
- Runtime lookup uses `prompt_id + version`:
  - `prompt_id` examples:
    - `llm.instruction_parsing.character_reference`
    - `vlm.grounding`
    - `vlm.assessment.instruction_following`
  - slash form is also accepted (e.g. `vlm/assessment/instruction_following`).

## YAML Contract

Each prompt asset YAML should follow this structure:

```yaml
system_prompt: "..."
user_prompt: "..."
version: v1
description: "short summary"
tags:
  - llm
  - object
metadata:
  source_file: object_extract.py
  source_key: SYSTEM_PROMPT_FOR_OBJECT_EXTRACTION
```

Required fields:
- `system_prompt` (string)
- `user_prompt` (string)

Recommended fields:
- `version` (e.g. `v1`)
- `description`
- `tags` (string list)
- `task` (e.g. `pair-judge`, `viescore`, `grounding`, `llm-parse`)
- `assessment_type` (when used by judge/viescore)
- `metadata` (free-form map)
