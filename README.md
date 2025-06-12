# Query2hint
 Query2Hint is a lightweight, quantized framework that replaces general-purpose LLM embeddings with those from compact language models fine-tuned on SQL-hint pairs using contrastive learning. 

 # Abstract
 Query optimization is a significant challenge in database systems,
 as it directly affects performance, scalability, and cost. Traditional
 machine learning methods often require extensive feature engineer
ing. Recent work has shown that LLM-based embeddings of SQL
 query text can effectively capture important semantic information
 relevant to optimization tasks. While it is commonly understood in
 the database community that structural elements primarily deter
mine query execution, the semantic intent of a query can provide
 critical context that improves the evaluation and selection of al
ternative execution plans. Building on this insight, we introduce
 Query2Hint a lightweight, quantized framework that replaces
 general-purpose LLM embeddings with those from compact lan
guage models fine-tuned on SQL-hint pairs using contrastive learn
ing. These pairs are generated from variations in execution plans
 triggered by different optimizer hints, thereby encoding critical
 structural differences while preserving query intent. Our prelimi
nary results indicate that these SQL-aware embeddings outperform
 proprietary large language models (LLMs) in terms of embedding
 quality, plan selection accuracy, and execution latency. Importantly,
 the fine-tuned model generalizes well across diverse SQL query for
mats and remains robust to syntactic variations. In our evaluation,
 Query2Hint reduced total execution latency by 70.37% and P90
 latency by 70.77% compared to PostgreSQL, and achieved 7.69%
 and 3.39% improvements over LLMSteer, respectively. 

 #  Architecture
![image](https://github.com/user-attachments/assets/a8889890-e081-48f4-9389-bd4e55a24a92)
