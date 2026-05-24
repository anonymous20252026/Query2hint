# AdaptSteer

AdaptSteer: Workload-Adaptive Optimizer Steering via
Execution-Derived Preference Supervision

 # Abstract
Database query optimizers often select suboptimal execution plans,
and effective steering requires identifying configurations that im-
prove query and workload performance without modifying DBMS
internals. Existing external steering systems can reason over mul-
tiple optimizer configurations, but many rely on online feedback,
exploration, or engine-specific steering mechanisms. More recent
LLM-based SQL steering approaches avoid deeper optimizer integra-
tion, but often depend on narrow supervision and general-purpose
embeddings that are costly, not explicitly aligned with execution
behavior, and difficult to adapt under workload shifts.
We present AdaptSteer, a novel workload-adaptive optimizer-
steering framework that learns from execution-derived preference
supervision. AdaptSteer converts latencies measured across diverse
query–configuration pairs into preference-based contrastive train-
ing signals for a compact SQL encoder, without invasive DBMS mod-
ifications or external APIs. The resulting representation supports
two lightweight decision heads: a calibrated classifier for safety-first
binary steering and a pairwise Condorcet ranker that decomposes
𝐾-way configuration selection into binary execution-preference
comparisons. We further apply Reptile-based meta-learning to the
trained encoder, enabling few-shot adaptation to unseen workloads.
Evaluations on the JOB and CEB benchmarks show that Adapt-
Steer reduces total workload latency by 70.1% relative to the Post-
greSQL default and achieves 4.4% lower total workload latency than
LLMSteer under the same binary protocol, without relying on exter-
nal API calls. Beyond binary steering, AdaptSteer improves multi-
action selection through pairwise preference ranking and benefits
from multi-configuration supervision as the action space grows.
Under CEB→JOB adaptation, AdaptSteer’s meta-learned variant
improves AUROC by 4.2 percentage points over the contrastive-
only baseline with only 20 target-workload adaptation queries.
Together, AdaptSteer unifies execution-derived preference supervi-
sion, binary and multi-action steering, and few-shot adaptation in
a compact, deployable optimizer-steering framework.
 #  Architecture
<img width="647" height="186" alt="image" src="https://github.com/user-attachments/assets/993bde1f-7a43-4f7b-8a3b-6ec7f238cf9b" />

