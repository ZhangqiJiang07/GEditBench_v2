# Modules Overview

Modules are the executable units registered in `PIPE_REGISTRY`. They are the objects referenced by pipeline configs through `pipe_name`, and they are the components that actually compute scores, winners, or parsed region artifacts.

<div class="doc-kind-row"><span class="doc-kind doc-kind--overview">Overview</span></div>

<div class="doc-section-card">

## Runtime contract

At the pipeline layer, a module is selected through `metric_configs` or `parser_grounder_config`:

```yaml
metric_configs:
  lpips:
    pipe_name: lpips-pipe
    init_config:
      model_path: ...
    scope: unedit_area
    runtime_params:
      ...
```

The pipeline runtime then:

1. resolves `pipe_name`
2. instantiates the module with `init_config`
3. maps `scope` to `mask_mode`
4. calls the module with the metric name and `runtime_params`

## Scope and masking

`BasePipeline` translates scopes into mask behavior:

- `edit_area` -> `mask_mode="inner"`
- `unedit_area` -> `mask_mode="outer"`

This matters for modules that consume full images plus `coords`, especially `lpips-pipe`, `ssim-pipe`, `clip-pipe`, `dino-v3-pipe`, and `depth-anything-v2-pipe`.

</div>

<div class="doc-section-card">

## Registered module catalog

| Registry key | Implementation | Role | Typical metric or output |
| --- | --- | --- | --- |
| `parser-grounder` | `parser_grounder.py` | Parse edit instructions and ground regions | parsed object dict + bbox list |
| `pairwise-judge` | `judge.py` | Compare two candidate images with an LLM or VLM | `Image A`, `Image B`, `Tie`, `Failed` |
| `viescore` | `judge.py` | Score one or two edited images with a client-backed prompt | single score or pairwise winner |
| `clip-pipe` | `clip_pipe.py` | CLIP-based semantic similarity | `emd`, `sam_clip_cls_sim` |
| `dino-v3-pipe` | `dino_pipe.py` | DINO-based structure or object similarity | `dinov3_structure_similarity`, `sam_dino_cls_sim` |
| `lpips-pipe` | `lpips_pipe.py` | Perceptual similarity | `lpips` |
| `ssim-pipe` | `ssim_pipe.py` | Structural similarity | `ssim`, `L_channel_ssim` |
| `sam-pipe` | `sam_pipe.py` | Region mask overlap inside a bbox | `iou` |
| `depth-anything-v2-pipe` | `depth_anything_v2_pipe.py` | Depth-map similarity | `depth_ssim` |
| `face-geometry-pipe` | `face_pipe.py` | Facial landmark geometry consistency | `L2_distance` |
| `face-texture-pipe` | `face_pipe.py` | Facial texture or color consistency | `high_frequency_diff`, `color_similarity`, `energy_ratio` |
| `face-identity-pipe` | `face_pipe.py` | Face identity preservation | `face_ID_sim`, `bg_faceID_sim`, `max_match_face_ID_sim` |
| `hair-consistency-pipe` | `hair_pipe.py` | Hair-region consistency | `color_distance`, `texture_energy_diff`, `high_frequency_diff` |
| `body-pose-and-shape-pipe` | `human_body.py` | Body pose and silhouette consistency | `body_shape_iou`, `body_pose_position_error` |
| `body-appearance-pipe` | `human_body.py` | Body appearance similarity | `body_appearance_dino_cosine_sim` |

</div>

<div class="doc-section-card">

## Important design caveats

The module interface is intentionally flexible, not perfectly uniform:

- some modules return `float`
- some return `dict`
- `parser-grounder` returns a tuple
- many modules return `None` or `0.0` on soft failure

That flexibility is by design. The pipelines already tolerate sparse or partially missing scores, so extension code should generally follow the same pattern unless a failure must be fatal.

</div>

<div class="doc-section-card">

## How to extend this layer safely

- Reuse an existing module if your need is only a new task-specific config.
- Add a new module when you need a new executable metric or judge behavior.
- Import the new module in `components/modules/__init__.py`, otherwise it will not be registered.
- Document the module's expected input contract, because not all modules consume the same input shape.

</div>

<div class="doc-section-card">

## Source-aligned module pages

The module docs are now split to stay closer to the source tree:

- [ParserGrounderPipe](/docs/components/modules/parser-grounder-pipe)
- [Judge Modules](/docs/components/modules/judge-modules)
- [CLIPPipe](/docs/components/modules/clip-pipe)
- [DINOv3Pipe](/docs/components/modules/dino-v3-pipe)
- [LPIPSPipe](/docs/components/modules/lpips-pipe)
- [SSIMPipe](/docs/components/modules/ssim-pipe)
- [SAMPipe](/docs/components/modules/sam-pipe)
- [DepthAnythingv2Pipe](/docs/components/modules/depth-anything-v2-pipe)
- [Face Pipes](/docs/components/modules/face-pipes)
- [HairConsistencyPipe](/docs/components/modules/hair-consistency-pipe)
- [Body Pipes](/docs/components/modules/body-pipes)

</div>
