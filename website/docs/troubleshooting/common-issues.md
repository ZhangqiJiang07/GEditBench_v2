# Common Issues

## `Unsupported pipeline: ...`

Cause:

- The `name` field in the pipeline YAML does not match a registered pipeline

Check:

- Whether `name` is one of `human-centric`, `object-centric`, or `vlm-as-a-judge`

## `Pipeline ... does not support task ...`

Cause:

- The CLI `--edit-task` is not listed in the config's `support_task`

Check:

- Whether the task name uses the expected lowercase underscore form
- Whether you started from the wrong config file

## `Candidate pool config not found`

Cause:

- `annotation` automatically looks for a candidate pool JSON that matches `--edit-task`

Check:

- Whether the matching file exists under `configs/datasets/candidate_pools/`
- Whether you need to pass a custom `--candidate-pool-dir`

## `Prompt file not found`

Cause:

- `prompt_info.prompt_id/version` does not resolve to a prompt asset

Check:

- Whether the template exists under `src/prompts/assets/`
- Whether the requested version is correct

## Model paths cannot be loaded

Cause:

- The paths in `user_config.yaml` do not exist or cannot be accessed from the current environment

Check:

- Whether every configured path is valid on the current machine
- Whether a container or remote runner can actually access those locations

## Many output fields are `null`

Common reasons include:

- Grounding did not produce valid bounding boxes
- The face detector did not find a face
- Segmentation failed
- A metric was skipped by the measurement rubric

Recommended checks:

- Validate on a small sample first
- Increase logging verbosity
- Confirm that parser-grounder is stable before debugging downstream metrics

## `winner` is often `Failed`

Common reasons include:

- The judge output does not satisfy the expected schema
- The prompt is not well aligned with the current model
- The backend returns text that cannot be parsed reliably

Recommended checks:

- Validate a single sample end to end first
- Inspect `raw_responses`
- Confirm that `prompt_info` points to the intended prompt template

## Documentation site build fails locally

If you are working on the docs site itself rather than the AutoPipeline runtime:

- The site currently requires Node.js `>= 20`
- If your host environment is still on Node 16, Docusaurus build will fail immediately
