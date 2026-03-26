# Primitives Overview

Primitives are the low-level building blocks that modules inherit from, compose with, or resolve through registries. They are not the main CLI-facing surface, but they define most of the reusable behavior inside AutoPipeline.

<div class="doc-kind-row"><span class="doc-kind doc-kind--overview">Overview</span></div>

<div class="doc-section-card">

## Registry-backed primitive surfaces

The primitive layer exposes three registries:

- `CLIENT_REGISTRY`
- `PROMPT_ADAPTER_REGISTRY`
- `EXPERT_REGISTRY`

These registries are the official primitive-level extension points. They are initialized through import side effects in:

- `src/autopipeline/components/primitives/clients/__init__.py`
- `src/autopipeline/components/primitives/__init__.py`

Other helpers such as `CLIPMixin`, `MaskProcessor`, or the low-level SSIM utilities are reused through direct inheritance or direct import rather than registry lookup.

</div>

<div class="doc-section-card">

## Source-aligned primitive catalog

The primitive docs are now organized closer to the source tree:

| Source file | Main classes or functions | Role |
| --- | --- | --- |
| `clients/base_client.py` | `BaseClient` | abstract client contract and JSON recovery |
| `clients/openai_client.py` | `OpenAIAPIClient` | OpenAI-compatible HTTP backend |
| `clients/google_client.py` | `GoogleAPIClient` | Google GenAI backend |
| `clients/vllm_client.py` | `vLLMOnlineClient` | vLLM OpenAI-compatible backend |
| `prompt_adapters.py` | `BasePromptAdapter`, `OpenAIStylePromptAdapter`, `GoogleGenAIStylePromptAdapter` | prompt payload formatting |
| `face_analyzer.py` | `FaceDetectionMixin`, `FaceMeshMixin`, `FaceIDMixin` | face detection, landmarks, identity |
| `grounding_dino.py` | `GroundingDINOMixin` | zero-shot detection helper |
| `human_segmenter.py` | `HumanSegmentationMixin`, `HairSegmentationMixin` | body and hair masks |
| `human_skeleton.py` | `HumanSkeletonMixin` | pose landmarks |
| `mask_processor.py` | `MaskProcessor` | bbox-to-mask conversion and patch masking |
| `semantic_consistency.py` | `CLIPMixin`, `DINOv3Mixin` | embedding extraction for similarity modules |
| `visual_consistency.py` | `SSIMMixin`, `LPIPSMixin`, `SAMSegmentationMixin`, `DepthAnythingv2Mixin` | visual metric backends |
| `ssim.py` | `ssim`, `ms_ssim`, `SSIM`, `MS_SSIM` | low-level SSIM infrastructure |

</div>

<div class="doc-section-card">

## When you need this section

Read the primitives docs if:

- you want to add a new backend client
- you want to support a new prompt payload format
- you need to understand where human-centric face, body, or hair artifacts come from
- you want to build a new module on top of an existing model backend

</div>

<div class="doc-section-card">

## Recommended reading order

1. [BaseClient](/docs/components/primitives/clients/base-client)
2. [OpenAIAPIClient](/docs/components/primitives/clients/openai-client)
3. [Prompt Adapters](/docs/components/primitives/prompt-adapters)
4. [Face Analyzer Primitives](/docs/components/primitives/face-analyzer)
5. [MaskProcessor](/docs/components/primitives/mask-processor)
6. [Semantic Consistency Mixins](/docs/components/primitives/semantic-consistency)
7. [Visual Consistency Mixins](/docs/components/primitives/visual-consistency)
8. [SSIM Utilities](/docs/components/primitives/ssim-utilities)

</div>

<div class="doc-section-card">

## Practical boundary between modules and primitives

As a rule:

- if the component appears in a pipeline config as `pipe_name`, it is a module
- if the component is reused by several modules or pipelines, it is usually a primitive

That distinction keeps the public runtime surface relatively compact while preserving reuse internally.

</div>

<div class="doc-section-card">

## Documentation granularity

Primitive pages are split by source file unless several classes form one tightly coupled unit. That is why:

- each backend client now has its own page
- `face_analyzer.py` stays together because detection, mesh, and identity logic are part of one face-analysis surface
- `visual_consistency.py` stays together because its mixins are the shared metric backends for several modules

</div>
