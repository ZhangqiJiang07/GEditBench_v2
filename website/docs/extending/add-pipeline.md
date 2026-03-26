# Add a Pipeline Family

Adding a new pipeline family is the most powerful extension path, but it is also the most invasive one. Use it only when the runtime orchestration itself no longer fits `object-centric`, `human-centric`, or `vlm-as-a-judge`.

<div class="doc-kind-row"><span class="doc-kind doc-kind--overview">Guide</span></div>

<div class="doc-section-card">

## Start from the Closest Existing Family

The current families already give you three useful templates:

- `object-centric`
  parser-grounder plus region-aware metrics
- `human-centric`
  parser-grounder plus experts plus region-aware metrics
- `vlm-as-a-judge`
  metric-only pairwise evaluation without parser-grounder or experts

In most cases, the fastest path is to copy the nearest family and then remove what you do not need.

</div>

<div class="doc-section-card">

## Add and Register the New Family

Pipeline family implementations live under `src/autopipeline/pipelines/`.

Minimal pattern:

```python
from .base_pipeline import BasePipeline
from . import PIPELINE_REGISTRY

@PIPELINE_REGISTRY.register("my-pipeline")
class MyPipeline(BasePipeline):
    required_configs = ["metric_configs"]

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        ...

    def __call__(self, input_dict, **kwargs):
        ...
```

Then import the new module from `src/autopipeline/pipelines/__init__.py`, otherwise `PipelineLoader` will not find it.

</div>

<div class="doc-section-card">

## Reuse `BasePipeline` Where Possible

`BasePipeline` already provides several useful building blocks:

- required-config validation
- image parsing
- parser-grounder loading
- expert loading
- metric config parsing
- pipe caching and smart loading

If your new family can reuse those helpers, do so. If not, make the divergence explicit in your own class rather than fighting hidden assumptions later.

</div>

<div class="doc-section-card">

## Add Config Files, Not New CLI Flags

The CLI already accepts a pipeline YAML path. That means most integration happens through config files, not argument parsing.

Typical required files:

- `configs/pipelines/<family>/...`
- optionally shared defaults in `configs/pipelines/modules_init/pipes_default.yaml`
- optionally expert defaults in `configs/pipelines/modules_init/experts_default.yaml`
- candidate pool files under `configs/datasets/candidate_pools/` if the new family supports new annotation tasks

The YAML `name` field must exactly match the pipeline registry key.

</div>

<div class="doc-section-card">

## Expect Some Non-Generic Runtime Wiring

This is the part most people underestimate.

The current runner stack still contains explicit assumptions about the existing families, including:

- executor choice
- annotation output grouping
- eval eligibility
- worker-side result shaping
- some dataset loading paths

So a truly new pipeline family usually requires updates outside `src/autopipeline/pipelines/`, especially in:

- `src/cli/autopipeline.py`
- `src/autopipeline/runners/workers.py`
- dataset-loading or result-formatting code when the new family changes input or output shape

</div>

<div class="doc-section-card">

## Important Caveats

There are a few framework assumptions to check early:

- `BasePipeline` only maps `edit_area` and `unedit_area` to `mask_mode`
- expert loading is tied to metric-name prefixes used by the current human-centric design
- `parser-grounder` is treated as a special pre-step rather than a regular metric

If your new family violates those assumptions, plan for code changes in the base layer instead of trying to hide the mismatch in YAML.

</div>

<div class="doc-section-card">

## Validation Checklist

1. the class is imported from `pipelines/__init__.py`
2. a pipeline YAML with `name: <your-family>` loads successfully
3. one minimal single-sample run succeeds
4. the runner emits the correct artifact shape
5. multi-worker execution still behaves correctly
6. the family does not accidentally depend on hardcoded behavior from another family

</div>
