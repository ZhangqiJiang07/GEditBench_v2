# VCReward Default Evaluation Outputs

This folder stores timestamped JSONL files for the default `eval_vc` pipeline on the `vc_reward` benchmark.

Expected JSONL schema

```json
{"key":"task_pair_000001","results":{"gt_winner":"Image B","winner":"Image B","raw_responses":{"pair-judge":{"type":"pairwise_comparison","value":"Image B"}}}}
```
