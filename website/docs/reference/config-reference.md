# Config Reference

This page answers one practical question: which configuration files does AutoPipeline read at runtime?

## 1. User configuration

```text
configs/pipelines/user_config.yaml
```

This is the file most users edit. The two important sections are:

### `client_config`

Controls model services or API backends:

- `ip_address`
- `vlm_port`
- `llm_port`
- `api_key`
- `base_url`
- `max_tokens`
- `retries`
- `timeout`

### `model_paths`

Controls local model paths:

- `dino_v3_path`
- `clip_path`
- `sam_path`
- `sam_cfg`
- `arcface_root`
- `hair_segmentation_path`
- `depth_anything_v2_path`

## 2. Default pipe configuration

```text
configs/pipelines/modules_init/pipes_default.yaml
```

This file stores reusable defaults for registered pipes, for example:

- `lpips-pipe`
- `clip-pipe`
- `dino-v3-pipe`
- `sam-pipe`
- `face-identity-pipe`
- `default_vlm_config`
- `default_llm_config`

Recommended practice:

- keep shared initialization arguments here
- keep task-specific differences in `init_config` inside the active pipeline YAML

## 3. Default expert configuration

```text
configs/pipelines/modules_init/experts_default.yaml
```

Typical entries include:

- `face-detector`
- `hair-segmenter`
- `human-segmenter`

## 4. Pipeline configs

```text
configs/pipelines/object_centric/
configs/pipelines/human_centric/
configs/pipelines/vlm_as_a_judge/
```

The most important fields in a typical pipeline YAML are:

| Field | Description |
| --- | --- |
| `name` | Registered pipeline name |
| `support_task` | List of supported tasks |
| `parser_grounder_config` | Instruction parsing and grounding config |
| `expert_configs` | Human-centric expert config, loaded as needed by metric families |
| `metric_configs` | Metric-to-pipe mapping plus runtime parameters |

## 5. Dataset configs

### Candidate pools

```text
configs/datasets/candidate_pools/*.json
```

`annotation` resolves the matching file automatically from `--edit-task`.

### Benchmark config

```text
configs/datasets/bmk.json
```

`eval` resolves the benchmark definition from `--bmk`.

## 6. Prompt assets

```text
src/prompts/assets/
```

Whenever the config contains:

```yaml
prompt_info:
  prompt_id: ...
  version: ...
```

the runtime resolves a concrete prompt template from the prompt asset store.

## Safety note

Do not publish real API keys, internal service endpoints, or private model paths in documentation examples. Prefer placeholders such as:

- `<PROJECT_ROOT>`
- `<YOUR_API_KEY>`
- `<MODEL_PATH>`
