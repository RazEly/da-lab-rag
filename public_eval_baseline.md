# Public Evaluation Baseline

Current public evaluation after downloading the real Git LFS artifacts.

## Official public evaluation

- public_queries: 29
- mean_ndcg@10: 0.5203
- query_phase_time: 37.83s

## Retrieval diagnostics

- pool_depth: 100
- query_phase_time: 33.37s
- NDCG@10: 0.5203
- recall@10: 0.6234
- hit@10: 0.8276
- recall@20: 0.6944
- hit@20: 0.9310
- recall@50: 0.8548
- hit@50: 1.0000
- recall@100: 0.9166
- hit@100: 1.0000
- mrr@100: 0.5141
- r_precision: 0.2506
- first_rel_rank: median 2, p90 12, found_any 1.00

## Interpretation

The first-stage retrieval usually finds at least one relevant page in the top 100, but the NDCG@10 is lower than recall@100. This suggests that improving the final ordering/reranking of the retrieved candidates is the main opportunity for improvement.