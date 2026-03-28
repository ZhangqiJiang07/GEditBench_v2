# Instruction-Following Evaluation Results

This folder stores timestamped pairwise evaluation outputs for the instruction-following dimension on the GEditBench v2/OpenEdit benchmark.

Expected JSONL schema

```json
{"key":"sample_pair_ModelA_vs_ModelB","results":{"input_dict":{"instruction":"...","input_image":"...","edited_images":["...","..."]},"winner":"Image A","raw_responses":{"pair-judge":{"type":"pairwise_comparison","value":"Image A"}}}}
```
