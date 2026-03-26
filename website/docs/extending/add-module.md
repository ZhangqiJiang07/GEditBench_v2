# Add a Module or Pipe

A `module` or `pipe` is the executable runtime unit selected by `metric_configs[*].pipe_name`. This is the right layer for a new metric, judge behavior, or other callable block that the pipeline should execute directly.

<div class="doc-kind-row"><span class="doc-kind doc-kind--overview">Guide</span></div>

<div class="doc-section-card">

## Add the Class and Register It

Pipe implementations live under `src/autopipeline/components/modules/`.

Minimal pattern:

```python
from . import BasePipe, PIPE_REGISTRY

@PIPE_REGISTRY.register("my-pipe")
class MyPipe(BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...

    def __call__(self, ...):
        ...
```

After adding the file, import it in `src/autopipeline/components/modules/__init__.py`. Without that import, the registry will never see the new pipe.

</div>

<div class="doc-section-card">

## Decide the Runtime Contract Up Front

The correct `__call__` signature depends on the target pipeline family.

Typical call patterns are:

- `object-centric`
  `ref_image`, `edited_image`, `coords`, `mask_mode`, `metric`, plus `runtime_params`
- `human-centric`
  cropped human images, masks, face boxes, `metric`, and region-dependent runtime parameters
- `vlm-as-a-judge`
  the normalized input dictionary rather than image pairs plus coords

Do not design the pipe in isolation. Start from the pipeline call site you intend to support.

</div>

<div class="doc-section-card">

## Expose Constructor Defaults

If the pipe has reusable init-time parameters, add them to:

- `configs/pipelines/modules_init/pipes_default.yaml`

Then consume them from pipeline YAML:

```yaml
metric_configs:
  my_metric:
    pipe_name: my-pipe
    default_config: ${pipes_default.my-pipe}
    init_config:
    scope: edit_area
    runtime_params:
      threshold: 0.2
```

`ConfigEngine` merges `default_config` and `init_config` before the class is instantiated, so the constructor receives only the final merged `init_config`.

</div>

<div class="doc-section-card">

## Keep `metric` and `pipe_name` Separate in Your Design

One pipe can serve several metrics.

That is already a core framework pattern:

- the YAML key under `metric_configs` becomes the runtime `metric` argument
- `pipe_name` selects the class

This is why one pipe can branch internally on several metric names without forcing you to duplicate the implementation class.

</div>

<div class="doc-section-card">

## Be Careful with Shared Pipe Instances

Inside a pipeline runtime, pipes are cached by:

- `pipe_name`
- merged `init_config`

If two metrics reference the same pipe with the same init config, they will share one instance.

That means a pipe should generally avoid storing mutable per-metric runtime state on `self`. Treat `runtime_params` and the `metric` argument as the per-call surface instead.

</div>

<div class="doc-section-card">

## Human-Centric and Parser-Grounder Caveats

Two areas need extra care:

- human-centric metrics may be skipped unless they are consistent with the measurement rubric and region routing used by `human_centric.py`
- `parser-grounder` is not just another metric pipe; it is loaded explicitly as a pre-step by the current region-aware pipeline families

If your new pipe depends on new region semantics, you may need both a new pipe and a small pipeline change.

</div>

<div class="doc-section-card">

## Validation Checklist

1. import the new file from `modules/__init__.py`
2. confirm the registry key resolves
3. add one minimal entry to `pipes_default.yaml` if the pipe needs init args
4. add one minimal `metric_configs` entry to a real pipeline YAML
5. run one sample through the target pipeline family
6. confirm the emitted score shape and `None` behavior are intentional

</div>
