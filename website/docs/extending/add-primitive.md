# Add a Primitive

In AutoPipeline, `primitive` is a broad term. Some primitives are registry-backed and instantiated from config, while others are plain reusable helpers mixed directly into a pipe.

<div class="doc-kind-row"><span class="doc-kind doc-kind--overview">Guide</span></div>

<div class="doc-section-card">

## First Decide Which Primitive You Mean

| Primitive kind | When to use it | Registration |
| --- | --- | --- |
| Expert primitive | The pipeline should instantiate it from `expert_configs` | `EXPERT_REGISTRY` |
| Backend client | A client-backed pipe needs a new transport backend | `CLIENT_REGISTRY` |
| Prompt adapter | A client backend needs a new payload format | `PROMPT_ADAPTER_REGISTRY` |
| Direct helper or mixin | A pipe imports and uses it directly | no registry required |

If the new logic is only used inside one pipe and does not need config-driven lookup, keep it as a direct helper or mixin.

</div>

<div class="doc-section-card">

## Add an Expert-Style Primitive

Expert primitives live under `src/autopipeline/components/primitives/` and are the right choice when a pipeline should load the primitive from `expert_configs`.

Minimal pattern:

```python
from . import EXPERT_REGISTRY

@EXPERT_REGISTRY.register("my-expert")
class MyExpertMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...
```

After adding the file, import it from `src/autopipeline/components/primitives/__init__.py`. Registration depends on that import side effect.

</div>

<div class="doc-section-card">

## Add Config Defaults

If the primitive is registry-backed, give it a stable config surface.

The usual places are:

- `configs/pipelines/modules_init/experts_default.yaml`
- `configs/pipelines/user_config.yaml` when the primitive needs machine-specific paths, ports, or API keys

That keeps pipeline YAMLs small and prevents model paths from being duplicated across many tasks.

Typical pattern:

```yaml
my-expert:
  model_path: ${user_config.model_paths.my_expert_path}
  threshold: 0.5
```

</div>

<div class="doc-section-card">

## Wire the Primitive into a Pipeline

For expert-style primitives, the runtime entry point is `expert_configs` inside the pipeline YAML:

```yaml
expert_configs:
  my-expert: ${experts_default.my-expert}
```

This matters most for the `human-centric` family, because that family explicitly loads experts before metric execution.

Two important caveats:

- expert loading is currently tied to metric-name prefixes such as `face_*`, `hair_*`, and `body_*`
- if your new expert feeds a new region family, you will probably need to update the expert-routing logic in `BasePipeline` and the metric input preparation inside `human_centric.py`

</div>

<div class="doc-section-card">

## Add a Direct Helper or Mixin Instead

Not every primitive should be registry-backed.

If the new code is only a reusable implementation detail for one or several pipes, a direct import is usually cleaner:

```python
from ..primitives.my_helper import MyHelperMixin
```

This is how several visual and semantic helpers are already used. In that case, you do not need:

- `EXPERT_REGISTRY`
- `expert_configs`
- `primitives/__init__.py` registration imports

</div>

<div class="doc-section-card">

## Related Special Cases

Two primitive families have their own extension flow:

- new backend clients belong under `components/primitives/clients/` and must be imported from `clients/__init__.py`
- new prompt adapters belong in `prompt_adapters.py` and register through `PROMPT_ADAPTER_REGISTRY`

Use those routes when the real change is transport or prompt formatting, not visual analysis.

</div>

<div class="doc-section-card">

## Validation Checklist

Before you treat the primitive as done, verify all of the following:

1. the class imports without errors
2. the registry key is visible if the primitive is registry-backed
3. config defaults resolve after `ConfigEngine` merges `_base_`, `default_config`, and `${...}` references
4. one pipeline YAML can instantiate the primitive successfully
5. one end-to-end sample reaches the expected output shape

</div>
