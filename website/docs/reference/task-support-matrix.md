# Task and Config Matrix

This page is not an exhaustive metric catalog. Its purpose is to map tasks to starter configs so you can begin from an existing setup instead of authoring YAML from scratch.

## Object-Centric

| Starter Config | Typical task | Notes |
| --- | --- | --- |
| `object_centric/subject_add.yaml` | `subject_add` | Add a subject or object |
| `object_centric/subject_remove.yaml` | `subject_remove` | Remove a target subject |
| `object_centric/subject_replace.yaml` | `subject_replace` | Replace a subject |
| `object_centric/color_alter.yaml` | `color_alter` | Modify color attributes |
| `object_centric/material_alter.yaml` | `material_alter` | Modify material attributes |
| `object_centric/size_adjustment.yaml` | `size_adjustment` | Resize a target |
| `object_centric/text_editing.yaml` | `text_editing` | Edit text inside an image |
| `object_centric/cref.yaml` | `cref` | Character or subject reference tasks |
| `object_centric/oref.yaml` | `oref` | Object reference tasks |

## Human-Centric

| Starter Config | Typical task | Notes |
| --- | --- | --- |
| `human_centric/ps_human.yaml` | `ps_human` | Portrait or human-appearance editing |
| `human_centric/motion_change.yaml` | `motion_change` | Human pose or motion editing |

## VLM-as-a-Judge

| Starter Config | Use case | Notes |
| --- | --- | --- |
| `vlm_as_a_judge/openai.yaml` | API backend | OpenAI-style chat endpoint |
| `vlm_as_a_judge/gemini.yaml` | Google-style backend | Gemini-compatible backend |
| `vlm_as_a_judge/vllm.yaml` | Self-hosted backend | Local or internal vLLM service |
| `vlm_as_a_judge/eval_if.yaml` | Benchmark eval | Instruction-following judge prompt |
| `vlm_as_a_judge/eval_vc.yaml` | Benchmark eval | Visual-consistency judge prompt |
| `vlm_as_a_judge/eval_vq.yaml` | Benchmark eval | Visual-quality judge prompt |

## Benchmark Keys

`eval` reads benchmark definitions from `configs/datasets/bmk.json`. Repository-visible keys include:

- `openedit`
- `vc_reward`
- `editscore_consistency`
- `editscore_prompt_following`
- `editreward_visual_quality`

## How to use this matrix

1. Decide whether you need structured scores or pairwise winners.
2. Choose the pipeline family.
3. Start from the closest existing config instead of creating a blank YAML file.
