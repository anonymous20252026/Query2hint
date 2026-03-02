# Query2hint

Query2Hint: Reframing Optimizer Steering as SQL-Aware Hint
Classification with Fast Meta-Adaptation 

 # Abstract
 Recent studies have explored the use of large language models
(LLMs) for query optimizer steering; however, practical deployment
remains limited by dependence on proprietary APIs, inference costs,
limited cross-workload generalization, and reproducibility issues.
Wepresent Query2Hint, a transfer-aware framework for relational
optimizer steering that combines domain-specific representation
learning with rapid adaptation. Query2Hint uses a two-stage training
design: (1) contrastive fine-tuning of compact SQLencodersonSQL
hint pairs constructed from measured hint-performance data, and
(2) Reptile-based meta-initialization to improve low-shot adaptation
under workload shifts. At inference time, Query2Hint performs
deterministic steering through lightweight hint classification, thereby
eliminating the need for text generation and dependence on external
APIs. In an end-to-end evaluation on JOB and CEB under a safety
first binary steering protocol (choosing between the default optimizer
and a fixed alternative steering option), Query2Hint reduced the total
execution latency and P90 tail latency by up to ∼70% relative to the
default optimizer, and further improved the end-to-end latency over
the LLMSteer baseline. Under cross-workload transfer between CEB
and JOB, Query2Hint remains competitive, with meta-initialization
improving adaptation in low-shot settings. These results suggest that
compact, task-tuned encoders can provide practical, reproducible,
and cost-effective query steering without proprietary LLM services.

 #  Architecture
![image](https://github.com/user-attachments/assets/a8889890-e081-48f4-9389-bd4e55a24a92)
