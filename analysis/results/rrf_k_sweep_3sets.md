# RRF_K sweep across query sets

mean NDCG@10, no-filter BM25, all else shipped.

| set | n | RRF_K=28 | RRF_K=30 | RRF_K=60 | RRF_K=100 |
|---|---|---|---|---|---|
| public | 29 | **0.5297** | 0.5295 | 0.5279 | 0.5180 |
| llm | 30 | **0.8597** | 0.8597 | 0.8546 | 0.8546 |
| squad | 119 | **0.4618** | 0.4606 | 0.4490 | 0.4424 |
