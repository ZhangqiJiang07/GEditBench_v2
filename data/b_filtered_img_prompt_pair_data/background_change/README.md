# background_change

This folder stores the filtered source set and generated candidates for the `background_change` task.

Typical files

- `meta_info.jsonl`: filtered source examples.
- `<model>_generation_results.jsonl`: generated edited images for one model.
- `gen_cache/<model>.jsonl`: resumable generation cache.

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Replace the background with a rainy street scene."}
```

In `meta_info.jsonl`, `image_path` points to the source image. In model result files, it points to the edited image produced by that model.
