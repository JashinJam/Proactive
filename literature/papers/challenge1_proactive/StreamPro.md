# StreamPro: From Reactive Perception to Proactive Decision-Making in Streaming Video

Ao Li<sup>∗,1,2</sup> Zihan Xiao<sup>∗,1,2</sup> Zihao Yue<sup>1</sup> Boshen Xu<sup>1</sup> Linli Yao<sup>3</sup> Jiaze Li<sup>2</sup> Pei Fu<sup>2</sup> Jianzhong Ju<sup>2</sup> Jian Luan<sup>2</sup> Qin Jin<sup>†,1</sup> 

<sup>1</sup>AIM3 Lab, Renmin University of China <sup>2</sup>MiLM Plus, Xiaomi Inc. <sup>3</sup>Peking University liaolea0808@gmail.com 

## Abstract

Proactive streaming video understanding requires models to continuously process video streams and decide when to respond, rather than merely what to respond. This naturally introduces a decision-making problem under partial observations, where models must balance early prediction against sufficient evidence. However, existing benchmarks largely follow a “see-then-answer” paradigm, where responses are triggered only after explicit evidence appears, effectively reducing proactive reasoning to delayed perception. As a result, they fail to evaluate a model’s ability to make timely and reliable decisions under incomplete observations. Moreover, training proactive models is inherently challenging due to the extreme imbalance between silence and response signals in streaming trajectories, as well as the need to jointly optimize response correctness and timing. To address these challenges, we introduce StreamPro-Bench, a new benchmark that evaluates streaming models from three complementary perspectives: Perception Understanding, Temporal Reasoning, and Proactive Agency, where the last measures a model’s ability to make early yet reliable decisions under partial observations. We further propose Stream Pro, a two-stage training framework for proactive learning. First, we introduce CB-Stream Loss to mitigate the severe supervision imbalance during supervised fine-tuning (SFT). Then, we apply Group Relative Policy Optimization (GRPO) with a multi-grained reward design that involves both turn-level and trajectory level rewards. Experiments show that StreamPro significantly improves proactive performance. On StreamPro-Bench, it achieves 41.5, substantially outperforming the previous best (10.4), while also maintaining strong performance on real-time streaming benchmarks, achieving 78.9 on StreamingBench-RTVU. 

## 1 Introduction

Streaming video understanding requires models to process visual inputs sequentially and make decisions in an online manner. Unlike offline settings where the entire video is fully observable, streaming scenarios inherently involve partial observations, requiring models to reason about when to respond in addition to what to respond. This introduces a fundamental challenge: models must make timely decisions under uncertainty, balancing early prediction against sufficient evidence. 

Existing streaming video tasks [1–14] can be broadly categorized into two paradigms. Real-time streaming tasks focus on low-latency responses, where models are required to answer queries based on currently visible content. These tasks emphasize response efficiency but largely bypass the question of when a response should be generated. In contrast, proactive tasks aim to evaluate temporal decisionmaking over evolving video streams, requiring models to determine the appropriate response timing. However, despite this motivation, existing benchmarks predominantly follow a “see-then-answer” paradigm, where responses are triggered only after explicit evidence appears in the video. 

![](images/8d88c366c36471a5edf7da8b37775b5f8e7e1893d5adfc5538d4e2ea5174e7ff.jpg)



Figure 1: Overview of streaming video paradigms and our contributions. Top: Different streaming paradigms. Real-Time Streaming emphasizes immediate responses. Proactive paradigms include the conventional “see-then-answer” approach, where responses are triggered upon observing explicit evidence, and our proposed Proactive Agency, which enables models to autonomously plan ahead and anticipate potential needs or risks. Bottom left: The StreamPro framework optimizes proactive models via SFT and GRPO. Bottom right: StreamPro-Bench evaluates capabilities from three dimensions: Perceptual Understanding, Temporal Reasoning, and Proactive Agency.


We argue that such a paradigm fundamentally reduces proactive reasoning to delayed perception. Instead of actively reasoning under uncertainty, models are encouraged to passively wait until sufficient evidence becomes observable, making the problem essentially reactive. As a result, these benchmarks fail to evaluate a model’s ability to make decisions under incomplete observations, such as anticipating future events, inferring latent user needs, or issuing early warnings before risks fully materialize. We refer to this missing capability as Proactive Agency, which captures the ability to perform timely and reliable decision-making under partial observations. 

Beyond evaluation, training proactive models is also inherently challenging. In streaming scenarios, most time steps correspond to silence, while only a small fraction require actual responses, resulting in a highly imbalanced supervision signal. Standard supervised fine-tuning (SFT) with cross-entropy loss is therefore dominated by silence tokens, biasing models toward remaining silent. Moreover, proactive behavior involves dual objectives: producing correct responses and generating them at appropriate times. Designing training objectives that jointly optimize both aspects remains non-trivial. 

To address these challenges, we propose a unified framework for proactive streaming video understanding, consisting of both a new benchmark and a training paradigm. First, we introduce StreamPro-Bench, a comprehensive benchmark that evaluates streaming models from three complementary perspectives: Perception Understanding, Temporal Reasoning, and Proactive Agency. The last dimension explicitly measures a model’s ability to make early yet reliable decisions under incomplete observations, going beyond conventional perception-driven evaluation. To enable effective learning of proactive behaviors, we further propose a two-stage training framework. In the first stage, we perform SFT and introduce CB-Stream Loss to mitigate the severe imbalance between silence and response signals. In the second stage, we adopt Group Relative Policy Optimization (GRPO) with a multi-grained reward design, including a Turn-level reward that captures per-response correctness and timing, and a Trajectory-level reward that evaluates holistic proactive behavior over the entire video via a rubric-based signal. This design enables models to balance accuracy and timeliness in a principled manner. 

<table><tr><td>Benchmark</td><td>QA Count</td><td>Video Length</td><td>Proactive QA Ratio</td><td>Perceptual Understanding</td><td>Temporal Reasoning</td><td>Proactive Agency</td></tr><tr><td>StreamingBench [15]</td><td>4.5K</td><td>4.1min</td><td>5.5%</td><td>√</td><td>×</td><td>×</td></tr><tr><td>OVO-Bench [16]</td><td>2.8K</td><td>7.1min</td><td>10.5%</td><td>√</td><td>×</td><td>×</td></tr><tr><td>ProactiveVideoQA [17]</td><td>1.4K</td><td>2.1min</td><td>100.0%</td><td>√</td><td>×</td><td>×</td></tr><tr><td>Omni-MMI [18]</td><td>2.3K</td><td>5.4min</td><td>22.0%</td><td>√</td><td>×</td><td>×</td></tr><tr><td>StreamPro-Bench</td><td>1.2K</td><td>2.2min</td><td>100.0%</td><td>√</td><td>√</td><td>√</td></tr></table>


Table 1: Comparison of mainstream proactive streaming benchmarks. StreamPro-Bench is specifically designed for proactive tasks and provides a more complete evaluation coverage.


To support training and evaluation, we construct two multi-task datasets, StreamPro-SFT-63K and StreamPro-RL-3K, along with a rubric-based evaluation protocol for trajectory-level assessment. Extensive experiments demonstrate that StreamPro significantly improves proactive capability, achieving substantial gains over existing methods on proactive benchmarks while maintaining strong performance on real-time streaming tasks and offline tasks. 

In summary, our contributions are as follows: 

• We propose StreamPro-Bench, a new benchmark redefining proactive capability with three evaluation dimensions: Perception Understanding, Temporal Reasoning, and a newly introduced Proactive Agency dimension, encompassing 7 representative tasks. 

• We propose StreamPro, a two-stage training framework tackle inherent challenges in proactive tasks. It features CB-Stream Loss to solve severe response-silence imbalances during SFT, and a multi-grained reward system combining Turn-level and Trajectory-level Rewards for GRPO optimization. 

• We construct two multi-task datasets, StreamPro-SFT-63K and StreamPro-RL-3K, for both supervised and reinforcement learning, enabling effective training of proactive models. 

## 2 Related Work

Proactive Streaming Video Understanding. Existing approaches to proactive streaming video understanding can be broadly categorized into two groups. (1) Module-based Proactive Models introduce additional modules to explicitly control response timing and activation, often relying on heuristics or task-specific components [7, 5, 12, 19, 4, 13, 10]. For example, Dispider [4] decomposes streaming interaction into perception, decision, and reaction modules. StreamBridge [7] augments offline VideoLLMs with a lightweight external gating model. StreamAgent [5] constructs a complete agent framework for temporal decision-making. (2) Token-based End-to-End Proactive Model use spatial tokens to integrate proactive capability directly into a single model [1, 20, 11]. A key challenge in this paradigm is the imbalance between silence and response supervision signals, which biases the model toward remaining silent excessively. VideoLLM-Online [1] mitigates the imbalance by controlling the activation threshold of the EOS token during inference. Streamo [11] further alleviates this issue by introducing a focal loss. To better address this issue, we propose CB-Stream Loss, a simple yet effective loss that re-weights streaming control tokens based on their effective frequency and assigns higher weights to response tokens. 

Recently, several works explore RL to optimize proactive behavior in streaming settings. MM-Duet2 [3] introduces PAUC reward optimization for proactive models. Other methods [21, 22] also design rewards that jointly consider response correctness and timing. However, such approaches mainly focus on per-timestep response quality, without explicitly modeling whether the overall response trajectory is coherent and well-structured. In contrast, we propose a multi-grained reward design that better captures both per-timestep response quality and trajectory-level coherence. 

Proactive Streaming Video Benchmark. With the rapid progress of streaming video understanding, many benchmarks have been proposed to evaluate proactive capabilities. StreamingBench [15] evaluates proactive models through alert-based questions, where models are required to respond upon observing visual evidence in the video. OVO-Bench [16] evaluates proactive models through three tasks: repetition event counting, sequential step recognition, and clue-revealing response. ProactiveV ideoQA [17] is the first benchmark specifically designed for proactive tasks, and it introduces PAUC evaluation metric. Omni-MMI [18] aims for comprehensive multimodal interaction evaluation by introducing audio inputs and multi-turn dialogue tasks. However, existing benchmarks predominantly follow a “see-then-answer” paradigm, reducing proactive reasoning to delayed perception. In contrast, we construct StreamPro-Bench to evaluate proactive capabilities from a more comprehensive perspective, jointly considering both delayed perception and proactive reasoning. 

![](images/7c7e977d5041be52042eb47d983eced80505222a0bef39662d82c0c373fa9cca.jpg)



Figure 2: Task case illustration of StreamPro-Bench. It contains 7 tasks categorized into three major dimensions: Perceptual Understanding, Temporal Reasoning and Proactive Agency. The timestamps of the user query and the ground truth are explicitly annotated along the timelines.


## 3 StreamPro-Bench

In this section, we introduce StreamPro-Bench. We first define the task taxonomy across three complementary capabilities (Section 3.1). We then detail the data construction pipeline and benchmark statistics (Section 3.2). Finally, we present the StreamPro-F1 Score, a tailored evaluation metric that jointly assesses temporal alignment and semantic correctness (Section 3.3). 

## 3.1 Task Taxonomy

We argue that a strong proactive model should possess three key capabilities: Perceptual Understanding, Temporal Reasoning, and Proactive Agency. The first two dimensions evaluate a model’s delayed perception ability, i.e., its ability to respond immediately after observing sufficient evidence. In contrast, proactive agency assesses timely and reliable decision-making under partial observations. Across these three dimensions, we define 7 specific tasks, as illustrated in Figure 2. 

Perceptual Understanding. This dimension evaluates the foundational ability to continuously perceive the states, dynamics, and changes of entities within streaming video inputs. We assess this capability through three tasks: (1) Event Understanding (EU): Describe the continuous steps of an evolving event as it unfolds (e.g., the sequential actions of a person setting up a tent). (2) Object Understanding (OU): Track the status of specific objects, requiring the model to instantly confirm when state transition occurs. (3) Anomaly Alert (AA): Trigger immediate warnings upon detecting sudden anomalous events, such as a person falling or a fire breaking out. 

Temporal Reasoning. Building upon foundational perception, this dimension assesses the ability to trace the exact timing and temporal dependencies of occurring events. It includes two tasks: (1) Temporal Perception (TP): reasoning about chronological order by identifying which events occur after a target event. (2) Temporal Grounding (TG): Perform video grounding in a streaming setting. 

Given a query, the model must immediately output the precise start and end timestamps the moment the described event concludes. 

Proactive Agency. This dimension evaluates the ability to proactively plan actions based on ongoing observations. It comprises two tasks: (1) Goal Planning (GP): Given a goal, the model is required to provide the next-step instruction in a timely manner once the previous step is completed. (2) Risk Forecasting (RF): Provide early risk anticipation in first-person scenarios to assist visually impaired pedestrians. The model must forecast potential environmental hazards at around 3 seconds before they materialize and offer actionable navigation advice. 

## 3.2 Benchmark Construction

To generate high-quality data across the 7 tasks, we design a pipeline based on a two-agent verification loop, followed by thorough human refinement for all samples. Given the inherent complexity of Risk Forecasting task, we rely entirely on human annotation and verification to guarantee data quality. Further details are provided in Appendix A.1. In total, StreamPro-Bench comprises 577 videos and 1,285 QA pairs. Figure 3 details the task distribution and the average video duration for each category. The comparison with current mainstream benchmarks is shown in Table 1. We provide more cases in Appendix D. 

![](images/cf3eb2665eed5660ecdff7d77ab35863594da68fcb116b7a5721d410830184c2.jpg)


Figure 3: StreamPro-Bench Statistics: the number of tasks and the average video length.

## 3.3 Evaluation Protocol

Evaluation Metric. To effectively evaluate proactive models, we introduce StreamPro-F1 Score, a trajectory-level metric that jointly assesses temporal alignment and semantic correctness, while naturally penalizing excessive and missed responses. For each generated response, the semantic accuracy $( S _ { \mathrm { a c c } } )$ is evaluated via an LLM judge for most tasks. For Temporal Grounding tasks, Intersection over Union (IoU) is used. The time score $( S _ { \mathrm { t i m e } } )$ evaluates the timing accuracy of the triggered response. The further the model’s response time $( t _ { \mathrm { p r e d } } )$ is from the ground-truth time $( t _ { \mathrm { g t } } )$ the lower the score, linearly dropping to zero once the distance reaches τ . To align with real-world requirements, this temporal tolerance τ varies across different tasks. The joint score (S) multiplies both aspects, ensuring a response is credited only when both timing and content are accurate: 

$$
S _ {\text {time}} = \max \left(0, 1 - \frac {| t _ {\text {pred}} - t _ {\text {gt}} |}{\tau}\right), \quad S _ {\text {acc}} = \text {LLMScore or IoU}, \quad S = S _ {\text {time}} \cdot S _ {\text {acc}}.\tag{1}
$$

At the trajectory level, we compute a score-weighted precision (P ) and recall (R) based on the matched response-ground truth pairs (S<sub>i</sub>), penalizing excessive and missed responses, respectively. These two metrics are then integrated into the F1 score, which serves as the final evaluation metric: 

$$
P = \frac {\sum_ {i} S _ {i}}{N _ {\mathrm{pred}}}, R = \frac {\sum_ {i} S _ {i}}{N _ {\mathrm{gt}}}, F _ {1} = \frac {2 P R}{P + R}.\tag{2}
$$

Evalution Method. For proactive models, we evaluate only those with officially released proactive inference scripts to ensure fair and reproducible evaluation [1–3, 23, 11]. For offline models [24, 25], we design a separate decision-tree-based frame-by-frame evaluation protocol to assess their proactive capabilities. Further details regarding the evaluation protocol, alignment with human preference, and benchmark result analysis are provided in Appendix A.2, Appendix A.3, and Appendix C.2. 

## 4 StreamPro Training Framework

This section introduces the two-stage StreamPro training framework, as illustrated in Figure 4. Section 4.1 details the SFT stage, Section 4.2 describes the RL stage, and Section 4.3 introduces the training datasets used across both stages. 

![](images/6812e432987b4d31678fd3100009c08fe4b1e67c1355fb60aeebcda2cb605611.jpg)



Figure 4: Overview of the StreamPro training framework. In the SFT stage, we propose the CB-Stream Loss to mitigate token imbalance by down-weighting the frequent </Silence> signals and up-weighting the sparse </Response> signals. In the GRPO stage, we optimize multi-grained rewards. The turn-level reward is computed using an additive step score that combines the semantic correctness $S _ { \mathrm { a c c } }$ assessed by an LLM judge with the timeliness $S _ { \mathrm { t i m e } }$ decayed based on temporal distance. Here, $N _ { \mathrm { g t } }$ and $N _ { \mathrm { p r e d } }$ denote the respective total numbers of ground truth and predicted events, which are both 2 in this illustrated example. Furthermore, the trajectory-level reward ensures global coherence by utilizing an LLM to evaluate the complete response sequence against multidimensional checklist criteria.


## 4.1 Supervised Fine-Tuning with CB-Stream Loss

Proactive Streaming Decision Format. At each time step, the model either remains silent or generates a response conditioned on the current video context. Specifically, the model outputs </Silence> when no response is required, and outputs </Response> followed by the corresponding textual answer when a response is triggered. We denote the decision token set as $\bar { \mathcal { S } } = \{ < / \mathrm { S i } 1 \mathrm { e n c e } > , < / \mathrm { R e s p o n s e } > \}$ , and the standard language token set as $\tau$ 

CB-Stream Loss. Proactive streaming video data exhibits a severe imbalance between response and silence signals, which causes standard cross-entropy training to favor conservative policies that over-predict silence. To mitigate this issue, we adopt a simple yet effective class-balanced reweighting strategy [26] based on the effective number of samples. For each decision token class $k \in { S } .$ , we define the effective sample size $E _ { k }$ and corresponding class-balanced weight $\hat { w } _ { k } ^ { \mathrm { C B } }$ as: 

$$
E _ {k} = \frac {1 - \beta^ {n _ {k}}}{1 - \beta}, \quad \hat {w} _ {k} ^ {\mathrm{CB}} = \frac {1 / E _ {k}}{\sum_ {j \in \mathcal {S}} 1 / E _ {j}} \cdot | \mathcal {S} |,\tag{3}
$$

where $n _ { k }$ is the frequency of class k computed over the current batch, and $\beta \in [ 0 , 1 )$ controls the degree of reweighting. We further introduce a constant scaling factor $\lambda _ { \mathrm { t e x t } }$ to balance optimization between decision tokens and language tokens. The final training objective is defined as: 

$$
\mathcal {L} _ {\mathrm{CB}} = \frac {1}{N} \sum_ {i = 1} ^ {N} w _ {i} ^ {\mathrm{CB}} \cdot \left[ - \log p _ {i} \right], \quad w _ {i} ^ {\mathrm{CB}} = \left\{ \begin{array}{l l} \hat {w} _ {y _ {i}} ^ {\mathrm{CB}}, & y _ {i} \in \mathcal {S}, \\ \lambda_ {\mathrm{text}}, & y _ {i} \in \mathcal {T}. \end{array} \right.\tag{4}
$$

## 4.2 Reinforcement Learning with Multi-Grained Rewards

While SFT provides a strong initialization, it suffers from exposure bias and struggles to foster the core proactive capability of balancing response timeliness and accuracy. To address this, we employ GRPO [27] with a multi-grained reward design comprising format, turn-level F1, and trajectory-level rubric components. Given a generated trajectory Y of length K, the overall reward is calculated as a weighted sum: 

$$
R (\mathcal {Y}) = w _ {\mathrm{fmt}} R _ {\mathrm{fmt}} + w _ {\mathrm{turn}} R _ {\mathrm{turn}} + w _ {\mathrm{traj}} R _ {\mathrm{traj}},\tag{5}
$$

where $w _ { \mathrm { f m t } } , w _ { \mathrm { t u r n } }$ , and $w _ { \mathrm { t r a j } }$ are the corresponding weight coefficients. 

Format Reward. To ensure structural integrity, $R _ { \mathrm { f m t } }$ strictly evaluates the decision token outputs. At each timestep, a step-level score of 1 is awarded if the model outputs a standalone </Silence> token or a </Response> token followed by non-empty text. Any other output format receives a score of 0. The final format reward is calculated by averaging these step-level scores over the entire trajectory length K. 

Turn-level F1 Reward. To optimize proactive triggering and factual correctness, we reuse the StreamPro-F1 metric to formulate a turn-level reward. However, directly applying the exact benchmark metric to RL introduces severe reward sparsity. Specifically, as StreamPro-F1 is designed for ideal proactive responses in real-world scenarios, it becomes overly stringent when applied to current models with still limited proactive capability. Its multiplicative form $( S _ { \mathrm { t i m e } } \times S _ { \mathrm { a c c } } )$ , strict temporal tolerance $\tau ,$ and greedy matching strategy—where a prediction only matches the first ground truth within its window—therefore cause the vast majority of exploratory steps to receive a reward of zero. To ensure stable optimization and provide denser, highly discriminative signals, we introduce three key modifications. First, we adopt an additive step score $\bar { S } ^ { \prime } = S _ { \mathrm { t i m e } } + S _ { \mathrm { a c c } }$ to prevent the entire reward from being nullified by a single poor component. Second, we employ a larger, universal temporal tolerance τ across all tasks. The enlarged window ensures that slightly misaligned predictions still receive partial rewards, and its universality reduces overall design complexity. Third, instead of greedy matching, for each ground truth timestamp $t _ { g t }$ , we consider all predictions falling within the window $[ t _ { g t } - \tau , t _ { g t } + \tau ]$ , and match it to the prediction that achieves the highest $S ^ { \prime }$ within this window. By aggregating these optimal matches, $R _ { \mathrm { t u r n } }$ is calculated using the F1 formulation: 

$$
R _ {\mathrm{turn}} = \frac {2 \sum_ {i} S _ {i} ^ {\prime}}{N _ {\mathrm{pred}} + N _ {\mathrm{gt}}}, \quad \mathrm{where} S _ {i} ^ {\prime} = S _ {\mathrm{time}, i} + S _ {\mathrm{acc}, i}.\tag{6}
$$

Trajectory-level Rubric Reward. $R _ { \mathrm { t u r n } }$ evaluates each turn independently and therefore fails to capture global trajectory coherence and semantic consistency, which is particularly important for complex tasks such as Goal Planning and Event Understanding. Therefore, we introduce a holistic rubric-based reward $R _ { \mathrm { t r a j } }$ . During offline data preparation, an LLM designer generates a customized rubric of $N _ { c }$ binary checkpoints for each training sample based on its question and ground truth. During online RL training, an LLM evaluator scores the predicted trajectory against these checkpoints, yielding a binary score $c _ { i } \in \{ 0 , 1 \}$ for each. The rubric verifies: (1) Granularity, ensuring responses are neither too fragmented nor too coarse relative to event duration; (2) Sequencing, confirming chronological consistency; and (3) Coverage, ensuring essential points are included while penalizing hallucinations. The final reward averages these checkpoint scores: 

$$
R _ {\mathrm{traj}} = \frac {1}{N _ {c}} \sum_ {i = 1} ^ {N _ {c}} c _ {i}.\tag{7}
$$

## 4.3 Training Data: StreamPro-SFT-63K and StreamPro-RL-3K

To enable effective training, we construct two datasets using the StreamPro-Bench data pipeline: StreamPro-SFT-63K and StreamPro-RL-3K. We then use these datasets in the StreamPro training framework. In the SFT stage, we jointly train on real-time streaming and proactive data, including TimeChat-Online-139K [14], VideoChat-Flash-3K [28], StreamPro-SFT-63K, and 287K filtered samples from Streamo-Instruct-465K [11]. In the RL stage, we focus exclusively on proactive tasks and train the model using StreamPro-RL-3K. The dataset statistics are provided in Appendix B.2. 

## 5 Experiments

## 5.1 Settings

Benchmarks. We evaluate our proposed model across three distinct categories of tasks: 

• Proactive Tasks. We evaluate on StreamPro-Bench (SPB) and Forward Active Responding (FAR) tasks from OVO–Bench [16]. Performance is measured using StreamPro-F1. We compare our approach with baseline models that provide open-source scripts for proactive inference [1– 3, 23, 11]. 

<table><tr><td rowspan="3">Methods</td><td rowspan="3">Venue</td><td rowspan="3">Params</td><td colspan="11">StreamPro-Bench</td><td colspan="3">OVO-Bench</td></tr><tr><td colspan="3">PU</td><td colspan="3">TR</td><td colspan="3">PA</td><td colspan="2">Overall F1</td><td colspan="3">FAR</td></tr><tr><td>Time↑</td><td>Acc.↑</td><td>F1↑</td><td>Time↑</td><td>Acc.↑</td><td>F1↑</td><td>Time↑</td><td>Acc.↑</td><td>F1↑</td><td>Avg.↑</td><td>W-Avg.↑</td><td>Time↑</td><td>Acc.↑</td><td>F1↑</td></tr><tr><td colspan="17">Offline Models + Proactive Prompt</td></tr><tr><td>Qwen2.5-VL-7B [24]</td><td>-</td><td>7B</td><td>7.5</td><td>7.7</td><td>3.4</td><td>2.4</td><td>0.4</td><td>0.1</td><td>6.5</td><td>2.2</td><td>0.9</td><td>1.5</td><td>1.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen3-VL-8B [25]</td><td>-</td><td>8B</td><td>60.1</td><td>51.4</td><td>6.6</td><td>27.6</td><td>16.6</td><td>0.8</td><td>23.0</td><td>12.2</td><td>2.6</td><td>3.3</td><td>3.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="17">Open-Source Proactive Models</td></tr><tr><td>VideoLLM-Online [1]</td><td>CVPR&#x27;24</td><td>8B</td><td>17.0</td><td>1.6</td><td>0.4</td><td>0.5</td><td>0.1</td><td>0.0</td><td>21.8</td><td>7.0</td><td>5.7</td><td>2.0</td><td>0.6</td><td>4.4</td><td>0.4</td><td>0.1</td></tr><tr><td>MMDuet [2]</td><td>EMNLP&#x27;25</td><td>7B</td><td>47.4</td><td>33.1</td><td>9.3</td><td>71.8</td><td>20.3</td><td>1.5</td><td>34.8</td><td>15.7</td><td>4.0</td><td>4.9</td><td>5.0</td><td>42.5</td><td>19.2</td><td>5.7</td></tr><tr><td>MMDuet2 [3]</td><td>ICLR&#x27;26</td><td>3B</td><td>37.2</td><td>28.5</td><td>9.8</td><td>47.2</td><td>30.1</td><td>3.1</td><td>22.3</td><td>7.4</td><td>2.7</td><td>5.2</td><td>5.9</td><td>33.5</td><td>18.1</td><td>6.5</td></tr><tr><td>MiniCPM-o-4.5 [23]</td><td>-</td><td>9B</td><td>20.6</td><td>16.9</td><td>13.5</td><td>5.2</td><td>0.8</td><td>0.6</td><td>6.6</td><td>4.9</td><td>3.8</td><td>6.0</td><td>6.4</td><td>15.6</td><td>10.6</td><td>8.4</td></tr><tr><td>Streamo [11]</td><td>CVPR&#x27;26</td><td>3B</td><td>20.3</td><td>17.2</td><td>7.8</td><td>27.2</td><td>19.1</td><td>14.0</td><td>6.0</td><td>2.6</td><td>2.7</td><td>8.2</td><td>10.4</td><td>11.5</td><td>7.7</td><td>5.4</td></tr><tr><td colspan="17">StreamPro Framework</td></tr><tr><td>StreamPro-SFT</td><td>-</td><td>3B</td><td>47.7</td><td>40.4</td><td>13.3</td><td>44.7</td><td>26.1</td><td>21.1</td><td>41.0</td><td>17.8</td><td>3.2</td><td>12.5</td><td>16.3</td><td>41.6</td><td>25.4</td><td>9.5</td></tr><tr><td>StreamPro-GRPO</td><td>-</td><td>3B</td><td>48.1</td><td>44.6</td><td>27.3</td><td>59.7</td><td>47.3</td><td>32.9</td><td>13.7</td><td>7.8</td><td>4.2</td><td>21.5</td><td>28.1</td><td>37.2</td><td>51.5</td><td>17.6</td></tr><tr><td>StreamPro-SFT</td><td>-</td><td>4B</td><td>66.6</td><td>61.7</td><td>24.7</td><td>45.1</td><td>32.5</td><td>27.9</td><td>46.1</td><td>20.5</td><td>5.7</td><td>19.4</td><td>24.7</td><td>41.6</td><td>32.2</td><td>17.3</td></tr><tr><td>StreamPro-GRPO</td><td>-</td><td>4B</td><td>66.0</td><td>61.6</td><td>45.0</td><td>67.5</td><td>60.4</td><td>44.4</td><td>41.2</td><td>52.7</td><td>7.6</td><td>32.3</td><td>41.5</td><td>44.2</td><td>33.9</td><td>20.6</td></tr></table>


Table 2: Performance on proactive tasks results acorss StreamPro-Bench and OVO-Bench. PU: Perception Understanding, TR: Temporal Reasoning, PA: Proactive Agency.


<table><tr><td rowspan="2">Methods</td><td rowspan="2">Venue</td><td rowspan="2">Params</td><td rowspan="2">Memory</td><td colspan="3">OVO-Bench</td><td colspan="3">StreamingBench</td><td colspan="2">Overall</td></tr><tr><td>RTVP</td><td>BT</td><td>W-Avg.</td><td>RTVU</td><td>CU</td><td>W-Avg.</td><td>Avg.</td><td>W-Avg.</td></tr><tr><td colspan="12">Streaming Models (Non-Proactive)</td></tr><tr><td>Flash-VStream [8]</td><td>ICCV&#x27;25</td><td>7B</td><td>√</td><td>28.4</td><td>27.4</td><td>28.0</td><td>23.2</td><td>25.6</td><td>23.8</td><td>25.9</td><td>25.1</td></tr><tr><td>TimeChat-Online [14]</td><td>MM&#x27;25</td><td>7B</td><td>×</td><td>58.6</td><td>42.0</td><td>51.5</td><td>75.3</td><td>38.1</td><td>66.7</td><td>59.6</td><td>61.9</td></tr><tr><td>StreamForest [6]</td><td>NeurIPS&#x27;25</td><td>7B</td><td>√</td><td>61.2</td><td>52.0</td><td>57.3</td><td>77.3</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>StreamingVLM [9]</td><td>ICLR&#x27;26</td><td>7B</td><td>√</td><td>62.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="12">Module-based Proactive Models</td></tr><tr><td>Dispider [4]</td><td>CVPR&#x27;25</td><td>7B</td><td>×</td><td>54.6</td><td>36.1</td><td>46.7</td><td>67.6</td><td>34.0</td><td>59.8</td><td>53.8</td><td>55.7</td></tr><tr><td>ViSpeak [10]</td><td>ICCV&#x27;25</td><td>7B</td><td>×</td><td>66.3</td><td>57.5</td><td>62.5</td><td>74.4</td><td>39.9</td><td>66.4</td><td>64.8</td><td>65.2</td></tr><tr><td>StreamBridge [7]</td><td>NeurIPS&#x27;25</td><td>7B</td><td>×</td><td>71.3</td><td>68.1</td><td>69.9</td><td>77.0</td><td>26.5</td><td>65.3</td><td>67.6</td><td>66.7</td></tr><tr><td>StreamAgent [5]</td><td>ICLR&#x27;26</td><td>7B</td><td>√</td><td>61.3</td><td>41.7</td><td>52.9</td><td>74.3</td><td>36.5</td><td>65.6</td><td>59.9</td><td>61.7</td></tr><tr><td>QueryStream [12]</td><td>ICLR&#x27;26</td><td>7B</td><td>√</td><td>61.4</td><td>42.1</td><td>53.1</td><td>74.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Thinking-QwenVL [13]</td><td>ICLR&#x27;26</td><td>7B</td><td>×</td><td>64.7</td><td>44.3</td><td>55.9</td><td>71.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="12">Token-based End-to-End Proactive Models</td></tr><tr><td>VideoLLM-Online [1]</td><td>CVPR&#x27;24</td><td>8B</td><td>×</td><td>20.8</td><td>17.7</td><td>19.5</td><td>36.0</td><td>28.1</td><td>34.2</td><td>27.0</td><td>29.7</td></tr><tr><td>Streamo-3B<eq>^{\dagger}</eq> [11]</td><td>CVPR&#x27;26</td><td>3B</td><td>×</td><td>60.9</td><td>40.5</td><td>52.1</td><td>75.8</td><td>41.1</td><td>67.8</td><td>60.0</td><td>62.4</td></tr><tr><td>Streamo-7B [11]</td><td>CVPR&#x27;26</td><td>7B</td><td>×</td><td>66.0</td><td>46.1</td><td>57.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>StreamPro-SFT</td><td>-</td><td>3B</td><td>×</td><td>62.0</td><td>40.0</td><td>52.5</td><td>77.1</td><td>40.0</td><td>68.5</td><td>60.5</td><td>63.5</td></tr><tr><td>StreamPro-GRPO</td><td>-</td><td>3B</td><td>×</td><td>62.5</td><td>34.9</td><td>50.7</td><td>76.0</td><td>39.3</td><td>67.5</td><td>59.1</td><td>62.3</td></tr><tr><td>StreamPro-SFT</td><td>-</td><td>4B</td><td>×</td><td>67.9</td><td>46.0</td><td>58.5</td><td>79.3</td><td>46.9</td><td>71.8</td><td>65.2</td><td>67.7</td></tr><tr><td>StreamPro-GRPO</td><td>-</td><td>4B</td><td>×</td><td>66.9</td><td>45.2</td><td>57.6</td><td>78.9</td><td>45.7</td><td>71.2</td><td>64.4</td><td>67.0</td></tr></table>


Table 3: Performance on real-time streaming tasks across OVO-Bench and StreamingBench. <sup>†</sup> indicates our own implementation.


• Real-time Streaming Tasks. We evaluate on Backward Tracing (BT) and Real-Time Visual Perception (RTVP) from OVO–Bench, alongside Real-Time Visual Understanding (RTVU) and Contextual Understanding (CU) from StreamingBench [15]. Note that we do not evaluate Proactive Output (PO) tasks within CU. For these tasks, we compare against 12 representative streaming models [8, 14, 6, 9, 4, 10, 7, 5, 12, 13, 1, 11]. 

• Offline Tasks. We evaluate performance on VideoMME [29] and LongVideoBench [30]. 

Implementation Details. We employ Qwen2.5-VL-3B [24] and Qwen3-VL-4B [25] as our backbone models. During SFT, we train both the projector and the LLM for 1 epoch using 64 H100 GPUs for 24 hours, with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a batch size of 512. The reweighting hyperparameter β is set to 0.9999, and the text scaling factor $\lambda _ { \mathrm { t e x t } }$ is set to 2. During RL, we implement the GRPO pipeline using veRL [31] and vLLM [32], training for 1 epoch using 8 H100 GPUs for 24 hours with a learning rate of $1 0 ^ { - 6 }$ and a global batch size of 16. We sample $G = 8$ generations per video context with a temperature of 1.0. For the GRPO multi-grained rewards, we set the temporal tolerance $\tau = 8 .$ along with the reward weights $w _ { \mathrm { f m t } } = 0 . 1 , w _ { \mathrm { t u r n } } = 0 . 4 5$ , and $w _ { \mathrm { t r a j } } = 0 . 4 5$ . For all LLM-based rubric generation and evaluations, we utilize Gemini 2.5 Pro [33]. All videos are sampled at 1 FPS. During inference, we employ a sliding window of 200 dialogue turns to improve inference efficiency. We use Qwen2.5-VL-3B as the backbone model for all ablation experiments. 

## 5.2 Main Result

We present the main results in Table 2 3 and 4. Green denotes the Qwen2.5-VL-3B backbone, and yellow denotes the Qwen3-VL-4B backbone. Bold and underlined values indicate the best and second-best results, respectively. 

Proactive Tasks. As shown in Table 2, StreamPro-GRPO-4B achieves SOTA performance on both SPB and the FAR tasks of OVO-Bench, reaching 41.5 on SPB and 20.6 on OVO-Bench, substantially outperforming the previous best baseline. 

Real-time Streaming Tasks. As shown in Table 3, StreamPro-GRPO-4B achieves 57.6 on OVO– Bench and 71.2 on StreamingBench, and with only 4B parameters it surpasses most existing 7B-scale models, demonstrating strong effectiveness in real-time streaming scenarios. Compared to StreamPro-SFT, StreamPro-GRPO shows a slight performance drop on this task, mainly because the RL stage focuses on optimizing proactive capabilities using only proactive data. 

Offline Tasks. As shown in Table 4, without using any offline data during training, StreamPro maintains the original offline performance of the backbone models. A slight performance drop is observed, consistent with the behavior of TimeChat-Online [14] on Qwen2.5-VL and AURA [34] on Qwen3-VL. More experimental results are provided in Appendix C. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen2.5-VL-3B</td><td colspan="2">Qwen3-VL-4B</td></tr><tr><td>VideoMME</td><td>LVBench</td><td>VideoMME</td><td>LVBench</td></tr><tr><td>Baseline</td><td>58.6</td><td>55.2</td><td>69.6</td><td>63.8</td></tr><tr><td>SFT</td><td>60.7</td><td>54.6</td><td>66.5</td><td>61.9</td></tr><tr><td>GRPO</td><td>60.4</td><td>52.9</td><td>67.3</td><td>60.4</td></tr></table>


Table 4: Performance on offline tasks.


<table><tr><td>Loss</td><td>SPB</td><td>OVO-RTVP</td><td>VideoMME</td></tr><tr><td>CE</td><td>6.6</td><td>62.3</td><td>60.4</td></tr><tr><td>Focal [11]</td><td>14.2</td><td>60.5</td><td>60.7</td></tr><tr><td>CB-Stream</td><td>16.3</td><td>62.0</td><td>60.7</td></tr></table>


Table 5: Ablation on loss functions.


## 5.3 Ablation Study

Effect of CB-Stream Loss. As shown in Table 5, we compare CB-Stream loss with cross-entropy (CE) loss and the focal loss in Streamo [11]. On proactive tasks, CB-Stream loss outperforms both CE and focal loss on SPB, demonstrating its effectiveness in alleviating the imbalance between silence and response signals. Meanwhile, compared to focal loss, CB-Stream loss also improves real-time streaming performance, achieving higher OVO-RTVP scores (62.0 vs. 60.5). 

<table><tr><td>τ</td><td>SPB</td><td>OVO-RTVP</td><td>VideoMME</td></tr><tr><td>3</td><td>25.8</td><td>61.2</td><td>60.7</td></tr><tr><td>8</td><td>28.1</td><td>62.5</td><td>60.4</td></tr></table>


Table 6: Ablation on τ .


<table><tr><td><eq>w_{fmt}</eq></td><td><eq>w_{turn}</eq></td><td><eq>w_{traj}</eq></td><td>SPB</td><td>OVO-RTVP</td><td>VideoMME</td></tr><tr><td>0.1</td><td>0.9</td><td>-</td><td>25.5</td><td>60.6</td><td>60.7</td></tr><tr><td>0.1</td><td>0.6</td><td>0.3</td><td>24.8</td><td>61.9</td><td>60.7</td></tr><tr><td>0.1</td><td>0.45</td><td>0.45</td><td>28.1</td><td>62.5</td><td>60.4</td></tr></table>


Table 7: Ablation on reward weights.


Effect of Temporal Tolerance τ . As shown in Table 6, we investigate the impact of the temporal tolerance τ in the turn-level F1 reward. Raising the temporal tolerance to τ = 8 improves performance on SPB and OVO-RTVP compared to a strict $\tau = 3 .$ , increasing scores from 25.8 to 28.1 and 61.2 to 62.5 respectively. This confirms our design choice in Section 4.2: a larger temporal window provides denser, more stable optimization signals by ensuring that slightly misaligned predictions still receive partial rewards during RL, all without significantly compromising offline performance. 

Effect of Trajectory-level Rubric Reward. As shown in Table 7, we analyze the impact of the trajectory-level reward by varying its weight $w _ { \mathrm { t r a j } }$ against the turn-level weight $w _ { \mathrm { t u r n } }$ . We observe that the balanced configuration $( w _ { \mathrm { t u r n } } = 0 . 4 5 , w _ { \mathrm { t r a j } } = 0 . 4 5 )$ yields the best performance on both SPB and OVO-RTVP, whereas relying more heavily on the turn-level reward leads to sub-optimal results. These results confirm that because the turn-level reward evaluates each turn independently, optimizing it in isolation often fails to capture global trajectory coherence. By contrast, appropriately incorporating the trajectory-level rubric explicitly enforces chronological consistency and information coverage across the entire video. 

## 6 Conclusion

In this paper, we propose StreamPro-Bench, which comprehensively evaluates proactive models from three dimensions: perception understanding, temporal reasoning, and proactive agency. Besides, we introduce StreamPro framework, which mitigates the imbalance between response and silence signals using the CB-Stream loss during the SFT stage, and further adopts a multi-granularity reward design in the RL stage to optimize proactive behavior. Extensive experiments demonstrate that StreamPro achieves substantial improvements in proactive capability, while maintaining strong performance on real-time streaming tasks and competitive results on offline benchmarks. Ablation studies further validate the effectiveness of each proposed component. We believe StreamPro provides a systematic solution for proactive streaming video understanding toward real-world proactive assistant systems. 

## References



[1] J. Chen, Z. Lv, S. Wu, K. Q. Lin, C. Song, D. Gao, J.-W. Liu, Z. Gao, D. Mao, and M. Z. Shou, “Videollm online: Online video large language model for streaming video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 407–18 418. 





[2] Y. Wang, X. Meng, Y. Wang, J. Liang, J. Wei, H. Zhang, and D. Zhao, “Videollm knows when to speak: Enhancing time-sensitive video comprehension with video-text duet interaction format,” arXiv preprint arXiv:2411.17991, vol. 1, no. 3, p. 5, 2024. 





[3] Y. Wang, S. Liu, D. Wang, N. Xu, G. Wan, H. Zhang, and D. Zhao, “Mmduet2: Enhancing proactive interaction of video mllms with multi-turn reinforcement learning,” arXiv preprint arXiv:2512.06810, 2025. 





[4] R. Qian, S. Ding, X. Dong, P. Zhang, Y. Zang, Y. Cao, D. Lin, and J. Wang, “Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 24 045–24 055. 





[5] H. Yang, F. Tang, L. Zhao, X. An, M. Hu, H. Li, X. Zhuang, Y. Lu, X. Zhang, A. Swikir et al., “Streamagent: Towards anticipatory agents for streaming video understanding,” arXiv preprint arXiv:2508.01875, 2025. 





[6] X. Zeng, K. Qiu, Q. Zhang, X. Li, J. Wang, J. Li, Z. Yan, K. Tian, M. Tian, X. Zhao et al., “Streamforest: Efficient online video understanding with persistent event memory,” arXiv preprint arXiv:2509.24871, 2025. 





[7] H. Wang, B. Feng, Z. Lai, M. Xu, S. Li, W. Ge, A. Dehghan, M. Cao, and P. Huang, “Streambridge: Turning your offline video large language model into a proactive streaming assistant,” arXiv preprint arXiv:2505.05467, 2025. 





[8] H. Zhang, Y. Wang, Y. Tang, Y. Liu, J. Feng, and X. Jin, “Flash-vstream: Efficient real-time understanding for long video streams,” in Proceedings of the IEEE/CVF international conference on computer vision, 2025, pp. 21 059–21 069. 





[9] R. Xu, G. Xiao, Y. Chen, L. He, K. Peng, Y. Lu, and S. Han, “Streamingvlm: Real-time understanding for infinite video streams,” arXiv preprint arXiv:2510.09608, 2025. 





[10] S. Fu, Q. Yang, Y.-M. Li, Y.-X. Peng, K.-Y. Lin, X. Wei, J.-F. Hu, X. Xie, and W.-S. Zheng, “Vispeak: Visual instruction feedback in streaming videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 21 778–21 788. 





[11] J. Xia, P. Chen, M. Zhang, X. Sun, and K. Zhou, “Streaming video instruction tuning,” arXiv preprint arXiv:2512.21334, 2025. 





[12] K. Zhang, Z. Yang, B. Wang, S. Qian, and C. Xu, “Querystream: Advancing streaming video understanding with query-aware pruning and proactive response,” in The Fourteenth International Conference on Learning Representations, 2026. 





[13] K. Zhang, Z. Yang, M. Han, H. Hao, Y. Zhuge, C. Li, J. Zhao, Z. Li, and X. Chang, “Progressive online video understanding with evidence-aligned timing and transparent decisions,” arXiv preprint arXiv:2604.18459, 2026. 





[14] L. Yao, Y. Li, Y. Wei, L. Li, S. Ren, Y. Liu, K. Ouyang, L. Wang, S. Li, S. Li et al., “Timechat-online: 80% visual tokens are naturally redundant in streaming videos,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 10 807–10 816. 





[15] J. Lin, Z. Fang, C. Chen, H. Cheng, Z. Wan, F. Luo, Z. Wang, P. Li, Y. Liu, and M. Sun, “Streamingbench: Assessing the gap for mllms to achieve streaming video understanding,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 12 147–12 151. 





[16] J. Niu, Y. Li, Z. Miao, C. Ge, Y. Zhou, Q. He, X. Dong, H. Duan, S. Ding, R. Qian et al., “Ovo-bench: How far is your video-llms from real-world online video understanding?” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 18 902–18 913. 





[17] Y. Wang, X. Meng, Y. Wang, H. Zhang, and D. Zhao, “Proactivevideoqa: A comprehensive benchmark evaluating proactive interactions in video large language models,” arXiv preprint arXiv:2507.09313, 2025. 





[18] Y. Wang, Y. Wang, B. Chen, T. Wu, D. Zhao, and Z. Zheng, “Omnimmi: A comprehensive multi-modal interaction benchmark in streaming video contexts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 18 925–18 935. 





[19] X. Ding, H. Wu, Y. Yang, S. Jiang, Q. Zhang, D. Bai, Z. Chen, and T. Cao, “Streammind: Unlocking full frame rate streaming video dialogue through event-gated cognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 13 448–13 459. 





[20] Y. Zhang, C. Shi, Y. Wang, and S. Yang, “Eyes wide open: Ego proactive video-llm for streaming video,” arXiv preprint arXiv:2510.14560, 2025. 





[21] Z. Liu, L. Guo, H. Li, R. Zhen, X. He, R. Ji, X. Ren, Y. Zhang, H. Lu, and J. Liu, “Thinking in streaming video,” arXiv preprint arXiv:2603.12938, 2026. 





[22] J. Qian, H. Du, G. Nan, S. Huang, J. Yu, H. Wang, J. Chen, M. Cai, M. Yang, J. Li, Z. Li, H. Wang, J. Liu, X. Jiang, and S. Leng, “Learning to respond: A large-scale benchmark and progressive learning framework for trigger-centric online video understanding,” https://openreview.net/pdf?id=gmpnSSiJt7, 2025. 





[23] OpenBMB, “Minicpm-o 4.5 technical report,” https://github.com/OpenBMB/MiniCPM-o/blob/main/docs/ MiniCPM_o_45_technical_report.pdf, 2026, gitHub technical report. 





[24] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2502.13923 





[25] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025. 





[26] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9268–9277. 





[27] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025. 





[28] X. Li, Y. Wang, J. Yu, X. Zeng, Y. Zhu, H. Huang, J. Gao, K. Li, Y. He, C. Wang et al., “Videochat-flash: Hierarchical compression for long-context video modeling,” arXiv preprint arXiv:2501.00574, 2024. 





[29] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025, pp. 24 108–24 118. 





[30] H. Wu, D. Li, B. Chen, and J. Li, “Longvideobench: A benchmark for long-context interleaved videolanguage understanding,” Advances in Neural Information Processing Systems, vol. 37, pp. 28 828–28 857, 2024. 





[31] G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu, “Hybridflow: A flexible and efficient rlhf framework,” arXiv preprint arXiv: 2409.19256, 2024. 





[32] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 





[33] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025. 





[34] X. Lu, Y. Bo, J. Chen, S. Li, X. Guo, H. Guan, F. Liu, D. Xu, P. Sun, H. Sun et al., “Aura: Always-on understanding and real-time assistance via video streams,” arXiv preprint arXiv:2604.04184, 2026. 





[35] Y. Liu, Z. Ma, Z. Qi, Y. Wu, Y. Shan, and C. W. Chen, “Et bench: Towards open-ended event-level videolanguage understanding,” Advances in Neural Information Processing Systems, vol. 37, pp. 32 076–32 110, 2024. 





[36] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li, “Llava-video: Video instruction tuning with synthetic data,” arXiv preprint arXiv:2410.02713, 2024. 





[37] L. Zhu, L. Wang, A. Raj, T. Gedeon, and C. Chen, “Advancing video anomaly detection: A concise review and a new dataset,” in The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. 





[38] W.-L. Chiang, L. Zheng, Y. Sheng, A. N. Angelopoulos, T. Li, D. Li, H. Zhang, B. Zhu, M. Jordan, J. E. Gonzalez et al., “Chatbot arena: An open platform for evaluating llms by human preference,” arXiv preprint arXiv:2403.04132, 2024. 





[39] R. A. Bradley and M. E. Terry, “Rank analysis of incomplete block designs: I. the method of paired comparisons,” Biometrika, vol. 39, no. 3/4, pp. 324–345, 1952. 





[40] B. Efron and R. J. Tibshirani, An introduction to the bootstrap. Chapman and Hall/CRC, 1994. 





[41] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024. 





[42] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023. 





[43] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llavaonevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024. 





[44] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024. 





[45] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025. 



## Appendices

A Details of StreamPro-Bench 14
A.1 Data Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.1.1 Video Collection and Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
A.1.2 Data Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
A.1.3 Data Verification .. 14
A.1.4 Annotation Guideline for Risk Forecasting 14
A.2 Evaluation Protocol 15
A.2.1 Evaluation Method 15
A.2.2 Evaluation Metric 15
A.3 Human Preference Alignment Validation 16
B Details of StreamPro Training Framework 17
B.1 Details of Reinforcement Learning 17
B.2 StreamPro-SFT-63K and StreamPro-RL-3K 18
C Additional Experiments 19
C.1 Additional Ablation Studies 19
C.1.1 Effectiveness of StreamPro-SFT-63K in SFT 19
C.1.2 Impact of Coefficient of Language Tokens λ in SFT 19
C.1.3 Impact of Matching Strategy in RL 19
C.2 All Results on StreamPro-Bench 19
C.3 All Results on Streamingbench RTVU and CU 20
C.4 All Results on OVO-Bench RTVP and BT 20
D Case Study 23
D.1 Benchmark Examples 23
D.2 Model Comparison 23
E Prompt Design 29
F Limitations 32
G Societal Impact 32
H Acknowledgment 32 

## A Details of StreamPro-Bench

In this section, we first detail the data pipeline in Section A.1, which includes video collection and filtering, multi-granularity QA generation, and a multi-stage verification process to ensure highquality annotations. We then present the evaluation protocol in Section A.2, covering the evaluation methods for both proactive and offline models, as well as the details of the StreamPro-F1 metric. All experimental results and further analysis on StreamPro-Bench are provided in Section C.2. 

## A.1 Data Pipeline

We construct StreamPro-Bench through a three-stage pipeline: video collection and filtering, data generation, and multi-stage verification. 

## A.1.1 Video Collection and Filtering

We collect raw videos from four sources: ET-Instruct-164k [35], LLaVA-Video-178k [36], MSAD [37], and manually crawled videos from diverse online platforms. To ensure sufficient temporal dynamics for proactive QA, we employ Qwen3VL-8B-Instruct [25] to retain videos that satisfy two criteria: (1) the number of scene transitions exceeds a predefined threshold, and (2) each scene exhibits a sufficiently long duration. This pre-filtering stage removes static or overly monotonous videos that are unsuitable for proactive QA tasks. For task-specific quality control, we further apply targeted filtering strategies: for Goal Planning, we enforce step uniqueness to ensure that procedural steps are clearly distinguishable and non-redundant; for Risk Forecasting, we apply an egocentric + outdoor filter to retain first-person outdoor videos aligned with real-world assistive navigation scenarios. 

## A.1.2 Data Generation

For each filtered video, we construct three complementary types of captions: (1) event-level captions that describe high-level activities and temporal events; (2) action-level captions that focus on human actions and behaviors; and (3) object-level captions that enumerate salient objects and their attributes. Based on these multi-granularity captions, we synthesize QA pairs. 

## A.1.3 Data Verification

Given that proactive streaming video understanding requires each question to be associated with multiple temporally grounded answers and precise temporal localization—an inherently strict require ment—we design a three-stage verification pipeline to ensure data quality. (1) Basic Verification. Each synthesized QA pair is first subjected to a basic verification and refinement process, where we correct semantic inconsistencies and align answers with their corresponding temporal segments. (2) Two-Agent Iterative Verification. A Discrimination Agent evaluates each QA pair against predefined quality criteria. If a sample is deemed unsatisfactory, it returns structured rejection feedback. A Generation Agent then revises the sample accordingly. This discriminate–generate loop is executed for up to three iterations to progressively improve data quality. (3) Human Review. QA pairs that pass the two-agent verification stage undergo final human review to ensure correctness and consistency. All steps involving LLMs are implemented using Gemini 2.5 Pro [33]. 

## A.1.4 Annotation Guideline for Risk Forecasting

For Risk Forecasting task, due to its prohibitive difficulty for reliable automatic annotation, we adopt a human annotation pipeline followed by human review to ensure high-quality and consistent labels. We design the annotation protocol based on a 3-second collision prediction principle, aiming to simulate proactive assistance for visually impaired users from an egocentric perspective. The 3-second horizon is chosen based on consultations with visually impaired individuals, corresponding to a practical and perceptually meaningful reaction window for real-world navigation. Annotators are instructed to estimate the time-to-collision (TTC) for each potential hazard under the assumption of constant pedestrian velocity and trajectory. A warning moment is defined as the timestamp at which the remaining TTC equals 3 seconds, and annotators are required to mark this moment accordingly. After initial annotation, all labels are further reviewed and refined by two independent human to ensure annotation quality. 

<table><tr><td>Task</td><td>Best Response Timestamp</td><td>Time Tolerance Window</td></tr><tr><td>Event Understanding</td><td>Within event duration</td><td>A [start time, end time + 3s] window</td></tr><tr><td>Object Understanding</td><td>Object state transition moment</td><td>A [-3s, +3s] window centered at the transition moment</td></tr><tr><td>Anomaly Alert</td><td>Anomaly onset time</td><td>A [0, +5s] post-onset window following anomaly onset</td></tr><tr><td>Temporal Perception</td><td>Within event duration</td><td>A [start time, end time + 3s] window</td></tr><tr><td>Temporal Grounding</td><td>Event ending time</td><td>A [-3s, +3s] window centered at the event end time</td></tr><tr><td>Goal Planning</td><td>Previous step completion time</td><td>A [-3s, 0] window around step completion time</td></tr><tr><td>Risk Forecasting</td><td>3s before hazard onset (warning time)</td><td>A [-1s, +3s] window centered at the warning time</td></tr></table>


Table 8: Temporal definitions for best response timestamps and evaluation windows across tasks.


Risk Taxonomy. We define four categories of risks to systematically characterize navigation hazards in streaming egocentric video. 

(1) Structural Terrain, referring to ground-level geometric irregularities that may compromise pedestrian stability (e.g., stairs, curbs, slopes). 

(2) Path Blockage, denoting physical obstacles that directly obstruct the walking trajectory (e.g., parked bicycles, barriers). 

(3) Overhanging Objects, representing elevated hazards that are difficult to detect using a white cane due to their height (e.g., low branches, signboards). 

(4) Functional Elements, capturing semantically critical environmental cues necessary for navigation decisions (e.g., traffic lights, crosswalks). 

## A.2 Evaluation Protocol

## A.2.1 Evaluation Method

For proactive models, we select approaches that provide open-source proactive inference scripts to ensure reproducibility and consistency [1–3, 11]. Following prior evaluation protocols, videos are sampled at 1 FPS and fed into the model in a streaming manner. At each time step, the model is required to produce a response based on the currently observed content. 

For offline models, we evaluate Qwen2.5-VL-7B [24], and Qwen3-VL-8B [25] using a decision-treestyle evaluation protocol to simulate proactive response behavior in a streaming setting. We display the evaluation prompt in Table E. Specifically, the model processes the video in a frame-by-frame manner. Before the query is issued, it passively observes the visual stream without producing any output. After the query is issued, the model follows a structured decision process at each subsequent time step. First, it assesses whether the accumulated visual context (from the beginning of the video up to the current frame) is sufficient to answer the query without speculation. If the information is insufficient, the model outputs “Wait” and continues observing. If the information is sufficient, the model then compares the currently inferred answer with its previously generated output. If the answer remains unchanged, it still outputs “Wait” to avoid redundant responses. Otherwise, it generates the updated answer. This protocol enables a systematic evaluation of whether offline models can support proactive decision-making under a streaming setting. 

## A.2.2 Evaluation Metric

Temporal Evaluation Metrics. To accommodate the heterogeneous temporal characteristics of different tasks, we define task-specific optimal response timings and corresponding temporal tolerance windows, as summarized in Table 8. Based on the form of the optimal response timing, we categorize tasks into two groups. For interval-based tasks, including Event Understanding and Temporal Perception, the optimal response is defined as an interval spanning the event duration. A response achieves full score if it falls within this interval. For timestamp-based tasks, i.e., all tasks except the interval-based ones, the optimal response corresponds to a specific moment (e.g., state transition, anomaly onset, or step completion). A response achieves full score if it occurs exactly at the designated timestamp. 

Beyond the optimal response timing, we introduce task-dependent temporal tolerance windows to allow deviations while penalizing temporally misaligned responses. Specifically, the penalty follows a linear decay as the response time moves away from the optimal point or interval, controlled by a tolerance parameter τ . Concretely, for Event Understanding and Temporal Perception, we allow a tolerance window extending 3 seconds after the event ends, with $\tau = 4$ . For Object Understanding, a symmetric tolerance of $\pm 3$ seconds around the state transition moment is adopted, with $\tau = 4$ For Anomaly Alert, we define a post-onset tolerance window of 5 seconds, with $\tau = 6 ,$ reflecting delayed but acceptable responses. For Temporal Grounding, a symmetric $\pm 3$ second window around the event end time is used, with $\tau = 4 .$ . For Goal Planning, where proactive behavior is essential, the optimal response is defined as the moment immediately after the completion of the previous step. To encourage anticipatory planning, we allow responses within a $[ - 3 , 0 ]$ second window prior to step completion, with $\tau = 4$ . Finally, for Risk Forecasting, the best response timestamp is defined as 3 seconds before hazard onset, emphasizing early warning capability. We adopt an asymmetric tolerance window of $[ - 1 , + 3 ]$ seconds around this point. To further reflect the asymmetry between premature and delayed responses, we use different decay factors: $\tau = 2$ for early responses (before the optimal timestamp) to more strongly penalize overly early predictions, and $\tau = 4$ for delayed responses, allowing slightly greater tolerance after the optimal warning time. 

We define the time score as a linear decay function based on the deviation from the optimal response timing: 

$$
S (t) = \max \left(0, 1 - \frac {\Delta t}{\tau}\right),\tag{8}
$$

where $\Delta t$ denotes the temporal deviation from the optimal timing. For timestamp-based tasks, ∆t is the absolute difference between the response time and the optimal timestamp. For interval-based tasks, $\Delta t = 0$ if the response falls within the target interval; otherwise, it is defined as the minimal temporal distance to the interval boundary. 

Answer–Prediction Matching Design. Since both the ground-truth answer and the model prediction are represented as trajectories, establishing a one-to-one correspondence between them is a critical problem. We adopt a prediction-first matching strategy, where each prediction is sequentially checked against all answer time windows. If a prediction falls within the temporal window of an answer, it is considered a successful match and is not allowed to match any other answer. Under this matching scheme, a single answer may be associated with multiple candidate predictions. We then select the prediction with the highest overall score as the final score for that answer. This matching strategy is simple and efficient. 

## A.3 Human Preference Alignment Validation

To validate the proposed StreamPro-F1 metric and demonstrate its alignment with human judgment, we conduct a pairwise human evaluation. 

![](images/3bc12f98016643f969080b128e742b2143b7c6599dbbf1661faa9bb33dafe11f.jpg)


Figure 5: Bootstrap rank stability heatmap.

![](images/8ab72571c3aabdb0e9d8c771f7058b71fd6a642b83285b15154b6c16fba4bd47.jpg)


Figure 6: Estimated Bradley-Terry scores and 95% confidence intervals via bootstrap resampling.

Evaluation Methodology. Following the Chatbot Arena paradigm [38], human annotators evaluate anonymized response trajectories from two sampled models and declare a winner or a tie. To infer global rankings from these relative preferences, we employ the Bradley-Terry (BT) model [39]. Given the inherent difficulty of streaming tasks, ties are frequent. We handle them via the Weighted BT approach [38], encoding each tie as two symmetric 0.5-weighted pseudo-observations, allowing seamless likelihood optimization without discarding data. 

Ranking Reliability Assessment. To quantify the statistical reliability of the derived rankings, we employ Bootstrap resampling [40]. Specifically, we generate 1,000 bootstrap samples with replacement from the collected pairwise comparisons and independently fit the Weighted BT model on each. This yields a Rank Stability Matrix (frequency of a model occupying each rank) and Score Confidence Intervals (median BT parameter with 95% CI). 

Results and Analysis. As shown in the rank stability heatmap (Figure 5) and score confidence intervals (Figure 6), the evaluated models exhibit highly confident stratification. Our proposed StreamPro-GRPO-4B unambiguously secures the top rank across all bootstrap replicates. Its confidence bounds are completely detached from the rest of the cohort, demonstrating absolute stability and significant superiority in proactive streaming capabilities. While minor variance exists among closely matched middle-tier models (e.g., Streamo and MiniCPM-o-4.5), the hierarchical boundaries remain strictly defined, with VideoLLM-Online consistently anchoring the final rank. 

Alignment with Benchmark Metric. Crucially, the final human preference ranking perfectly matches the ranking produced by our proposed StreamPro-F1 metric (yielding a Spearman’s rank correlation $\rho = 1 . 0 )$ . This absolute consensus demonstrates that our automated evaluation protocol faithfully captures human perception of proactive streaming quality, validating StreamPro-Bench as a robust and reliable evaluation standard. 

## B Details of StreamPro Training Framework

## B.1 Details of Reinforcement Learning

Matching Algorithm for Turn-level F1 Reward. Here, we detail the matching algorithm used to compute the turn-level F1 reward $R _ { \mathrm { t u r n } }$ . To ensure dense and discriminative optimization signals, we match each ground truth to the optimal prediction within a universal temporal tolerance window τ by maximizing the additive step score $S ^ { \prime }$ . The detailed procedure is outlined in Algorithm 1. 

It is worth noting that for tasks under the Proactive Agency dimension, we modify the matching window to strictly encourage predictive behavior. Since the model is expected to make predictions before an event fully unfolds rather than answering afterwards, the temporal window is restricted to $[ t _ { \mathrm { g t } } - \tau , t _ { \mathrm { g t } } ]$ ]. Any predictions generated after $t _ { \mathrm { g t } }$ are not considered for matching in these tasks. 

Algorithm 1 Turn-level F1 Reward Matching Algorithm
Require: Ground truth set G, Prediction set P, Temporal tolerance τ
Ensure: Turn-level F1 Reward $R_{turn}$ 1: Initialize matched prediction set $M \leftarrow \emptyset$ 2: Initialize total score $\Sigma_{score} \leftarrow 0$ 3: for each ground truth $g \in G$ do
4: Initialize best score $S'_{best} \leftarrow 0$ 5: Initialize best prediction $p_{best} \leftarrow NULL$ 6: for each prediction $p \in P \setminus M$ do
7: if $p.t \in [g.t - \tau, g.t + \tau]$ then
8: {Use $[g.t - \tau, g.t]$ for Proactive Agency}
9: Calculate $S_{time}$ and $S_{acc}$ between g and p
10: Calculate step score $S' = S_{time} + S_{acc}$ 11: if $S' > S'_{best}$ then
12: $S'_{best} \leftarrow S'$ 13: $p_{best} \leftarrow p$ 14: end if
15: end if
16: end for
17: if $p_{best} \neq NULL$ then
18: $M \leftarrow M \cup \{p_{best}\}$ 19: $\Sigma_{score} \leftarrow \Sigma_{score} + S'_{best}$ 20: end if
21: end for
22: $R_{turn} \leftarrow \frac{2 \cdot \Sigma_{score}}{|P| + |G|}$ 23: return $R_{turn}$ 

## B.2 StreamPro-SFT-63K and StreamPro-RL-3K

We use the StreamPro-Bench data pipeline (excluding the human review step in the verification stage) to construct a 66K-scale dataset. We then further filter the data to obtain StreamPro-SFT-63K and StreamPro-RL-3K. Due to the difficulty of constructing the Risk Forecasting task, this dataset does not include Risk Forecasting samples. 

In the SFT stage, our primary objective is to endow the model with fundamental proactive capabilities. The training tasks encompass event captioning (EC), action captioning (AC), event understanding (EU), object understanding (OU), temporal perception (TP), and temporal grounding (TG). In the RL stage, we further introduce more challenging proactive tasks. The training tasks encompass event understanding (EU), object understanding (OU), Anomaly Alert (AA), temporal preception (TP), temporal grounding (TG) and Goal Planning (GP). The task distributions of the two datasets are shown in Figure 7. 

![](images/6fe063b898d1c8f546aace40013a414d19c436e6697a781d6c9a21b5b462b182.jpg)


![](images/3e55dcd00a1bbe20e700b85aea06296de35231599dfe6e232b3b757180a1d0cd.jpg)


Figure 7: Dataset Statistics. Left: StreamPro-SFT-63K; Right: StreamPro-RL-3K.

## C Additional Experiments

## C.1 Additional Ablation Studies

## C.1.1 Effectiveness of StreamPro-SFT-63K in SFT

To verify the quality of the data generated by our pipeline, we conduct an ablation study as shown in Table 9. When training only with the 429K open-source data, the model achieves 8.9 on SPB and 60.7 on OVO-RTVP. After incorporating the additional StreamPro-SFT-63K data, performance improves substantially, reaching 16.3 on SPB and 62.0 on OVO-RTVP, while also improving VideoMME from 58.5 to 60.7. These results demonstrate the effectiveness and high quality of our constructed SFT data in enhancing both proactive and real time streaming tasks. 

<table><tr><td>Data</td><td>SPB</td><td>OVO-RTVP</td><td>OVO-BT</td><td>VideoMME</td></tr><tr><td>Open source 429k</td><td>8.9</td><td>60.7</td><td>38.4</td><td>58.5</td></tr><tr><td>429k+StreamPro-SFT-63k</td><td>16.3</td><td>62.0</td><td>40.0</td><td>60.7</td></tr></table>


Table 9: Ablation study on the effectiveness of StreamPro-SFT-63K data during SFT training.


## C.1.2 Impact of Coefficient of Language Tokens λ in SFT

To investigate the effect of the language token weighting coefficient λ, we conduct an ablation study as shown in Table 10. Increasing the weight of language tokens improves performance on offline tasks, but it weakens the model’s ability to accurately model response timing decisions, leading to degraded performance on SPB. Considering the trade-off among the three evaluation tasks, we set the weighting coefficient to 2 as the default choice. 

<table><tr><td>Weight</td><td>SPB</td><td>OVO-RTVP</td><td>OVO-BT</td><td>VideoMME</td></tr><tr><td>1</td><td>16.3</td><td>61.8</td><td>40.0</td><td>60.4</td></tr><tr><td>2</td><td>16.3</td><td>62.0</td><td>40.0</td><td>60.7</td></tr><tr><td>3</td><td>15.3</td><td>60.8</td><td>38.9</td><td>60.9</td></tr></table>


Table 10: Ablation study on the language token weighting coefficient λ.


## C.1.3 Impact of Matching Strategy in RL

We also ablate the matching strategy used to assign predictions to ground truth during RL. We compare our proposed GT-first (Optimal) matching with the Prediction-first (Greedy) matching directly adopted from the benchmark evaluation. As demonstrated in Table 11, our proposed strategy improves the overall performance. Although the advantage on proactive and real-time tasks such as SPB (28.0 to 28.1) and OVO-RTVP (62.4 to 62.5) is relatively marginal, it yields a much more significant improvement on OVO-BT, increasing the score from 32.7 to 34.9. 

Since the RL stage only trains on proactive tasks, we attribute this improvement on OVO-BT to the stability of the reward signals. Greedy matching often penalizes high-quality predictions if poorer exploratory outputs precede them, introducing noise that degrades the model’s foundational temporal memory. Conversely, optimal matching consistently rewards the best prediction within the window. This stable optimization helps preserve the model’s core temporal understanding during RL, naturally benefiting memory-sensitive tasks like backward tracing. 

## C.2 All Results on StreamPro-Bench

As shown in Table 12, Table 13 and Table 14, we present comprehensive results on StreamPro-Bench across all three evaluation dimensions and seven tasks, reporting Time Score, Accuracy, Precision, Recall, and StreamPro-F1, where higher values indicate better performance. 

<table><tr><td>Matching Strategy</td><td>SPB</td><td>OVO-RTVP</td><td>OVO-BT</td><td>VideoMME</td></tr><tr><td>Prediction-first</td><td>28.0</td><td>62.4</td><td>32.7</td><td>60.7</td></tr><tr><td>GT-first</td><td>28.1</td><td>62.5</td><td>34.9</td><td>60.4</td></tr></table>


Table 11: Ablation study on the matching strategy during GRPO.


Existing proactive models perform poorly on StreamPro-Bench, indicating that their capabilities are still far from those required for an ideal real-world proactive system. Among the three evaluated dimensions, they achieve relatively acceptable performance in perceptual understanding, while showing significant deficiencies in temporal reasoning and proactive agency. 

Specifically, compared with other proactive models, MMDuet [2] and MMDuet2 [3] achieve relatively high Time Score, Accuracy, and Recall, since they tend to generate responses more frequently, which increases the probability of hitting the correct temporal window. However, due to this tendency toward over-generation, Precision is penalized and thus drops to a more moderate level. Besides, Streamo [11] demonstrates strong temporal reasoning capabilities, due to the presence of video temporal grounding tasks in its training data. Note that VideoLLM-Online [1] exhibits an inflated performance on Goal Planning, which is mainly caused by its tendency to produce overly generic and highly abstract responses. Such responses are often assigned relatively stable and moderate scores by the LLM-based evaluator across different cases, leading to an apparently high average score. 

## C.3 All Results on Streamingbench RTVU and CU

Table 15 presents the detailed results for each task in StreamingBench. 

## C.4 All Results on OVO-Bench RTVP and BT

Table 16 presents the detailed results for each task in OVO-Bench. 

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Params</td><td colspan="5">Event Understanding</td><td colspan="5">Object Understanding</td><td colspan="5">Anomaly Alert</td><td rowspan="2">W-Avg. F1</td></tr><tr><td>Time</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Time</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Time</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td colspan="18">Offline Models + Proactive Prompt</td></tr><tr><td>Qwen2.5-VL-7B [24]</td><td>7B</td><td>8.9</td><td>8.0</td><td>0.3</td><td>4.6</td><td>0.5</td><td>15.6</td><td>19.2</td><td>19.8</td><td>11.7</td><td>11.3</td><td>1.7</td><td>21.7</td><td>0.7</td><td>1.4</td><td>0.8</td><td>3.4</td></tr><tr><td>Qwen3-VL-8B [25]</td><td>8B</td><td>43.1</td><td>25.1</td><td>1.7</td><td>24.2</td><td>2.5</td><td>68.9</td><td>61.8</td><td>1.7</td><td>57.8</td><td>3.0</td><td>66.6</td><td>63.3</td><td>7.2</td><td>48.7</td><td>11.7</td><td>6.6</td></tr><tr><td colspan="18">Open-Source Proactive Models</td></tr><tr><td>VideoLLM-Online [1]</td><td>8B</td><td>0.6</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.1</td><td>11.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>31.7</td><td>3.7</td><td>0.7</td><td>2.6</td><td>0.8</td><td>0.4</td></tr><tr><td>MMDuet [2]</td><td>7B</td><td>82.1</td><td>39.4</td><td>8.4</td><td>36.0</td><td>13.0</td><td>44.0</td><td>40.7</td><td>2.7</td><td>23.2</td><td>4.8</td><td>25.5</td><td>24.1</td><td>9.1</td><td>10.4</td><td>9.4</td><td>9.3</td></tr><tr><td>MMDuet2 [3]</td><td>3B</td><td>81.6</td><td>59.3</td><td>11.8</td><td>56.5</td><td>17.7</td><td>31.5</td><td>9.5</td><td>3.1</td><td>4.9</td><td>3.3</td><td>9.9</td><td>18.7</td><td>7.9</td><td>8.7</td><td>8.1</td><td>9.8</td></tr><tr><td>MiniCPM-o-4.5 [23]</td><td>9B</td><td>10.9</td><td>8.0</td><td>15.5</td><td>7.7</td><td>9.8</td><td>14.4</td><td>20.9</td><td>25.9</td><td>9.6</td><td>13.9</td><td>31.0</td><td>21.7</td><td>15.3</td><td>16.9</td><td>15.8</td><td>13.5</td></tr><tr><td>Streamo [11]</td><td>3B</td><td>13.4</td><td>8.2</td><td>20.5</td><td>7.4</td><td>8.9</td><td>57.9</td><td>49.5</td><td>12.5</td><td>40.8</td><td>15.8</td><td>2.6</td><td>3.9</td><td>2.3</td><td>2.3</td><td>2.3</td><td>7.8</td></tr><tr><td colspan="18">StreamPro Framework</td></tr><tr><td>StreamPro-SFT</td><td>3B</td><td>86.1</td><td>58.3</td><td>11.9</td><td>56.1</td><td>18.4</td><td>48.6</td><td>54.2</td><td>8.1</td><td>41.6</td><td>12.2</td><td>20.7</td><td>19.8</td><td>9.9</td><td>12.4</td><td>10.5</td><td>13.3</td></tr><tr><td>StreamPro-GRPO</td><td>3B</td><td>83.6</td><td>63.0</td><td>36.4</td><td>59.9</td><td>43.5</td><td>63.9</td><td>66.8</td><td>29.5</td><td>54.4</td><td>36.7</td><td>14.0</td><td>18.6</td><td>10.4</td><td>10.5</td><td>10.4</td><td>27.3</td></tr><tr><td>StreamPro-SFT</td><td>4B</td><td>90.2</td><td>67.3</td><td>18.0</td><td>65.2</td><td>26.3</td><td>53.2</td><td>58.4</td><td>11.6</td><td>45.9</td><td>16.7</td><td>58.3</td><td>59.8</td><td>24.8</td><td>39.6</td><td>28.5</td><td>24.7</td></tr><tr><td>StreamPro-GRPO</td><td>4B</td><td>82.0</td><td>67.4</td><td>49.2</td><td>64.6</td><td>54.6</td><td>51.9</td><td>49.5</td><td>35.1</td><td>40.1</td><td>36.4</td><td>63.3</td><td>64.9</td><td>41.5</td><td>48.8</td><td>43.5</td><td>45.0</td></tr></table>


Table 12: Detailed Results on Perceptual Understanding.


<table><tr><td rowspan="2">Methods</td><td rowspan="2">Params</td><td colspan="5">Temporal Perception</td><td colspan="5">Temporal Grounding</td><td rowspan="2">W-Avg F1</td></tr><tr><td>Time</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Time</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td colspan="13">Offline Models + Proactive Prompt</td></tr><tr><td>Qwen2.5-VL-7B [24]</td><td>7B</td><td>1.1</td><td>1.2</td><td>0.7</td><td>1.0</td><td>0.8</td><td>8.8</td><td>0.5</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.1</td></tr><tr><td>Qwen3-VL-8B [25]</td><td>8B</td><td>50.8</td><td>34.8</td><td>0.9</td><td>34.3</td><td>1.6</td><td>6.8</td><td>0.3</td><td>0.1</td><td>0.2</td><td>0.1</td><td>0.8</td></tr><tr><td colspan="13">Open-Source Proactive Models</td></tr><tr><td>VideoLLM-Online [1]</td><td>8B</td><td>0.7</td><td>0.1</td><td>0.0</td><td>0.1</td><td>0.0</td><td>0.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MMDuet [2]</td><td>7B</td><td>91.8</td><td>40.6</td><td>1.6</td><td>38.6</td><td>3.0</td><td>53.9</td><td>2.1</td><td>0.1</td><td>1.1</td><td>0.1</td><td>1.5</td></tr><tr><td>MMDuet2 [3]</td><td>3B</td><td>85.8</td><td>63.6</td><td>4.0</td><td>61.3</td><td>6.7</td><td>12.7</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>3.1</td></tr><tr><td>MiniCPM-o-4.5 [23]</td><td>9B</td><td>1.3</td><td>0.8</td><td>0.1</td><td>0.6</td><td>0.1</td><td>3.5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.6</td></tr><tr><td>Streamo [11]</td><td>3B</td><td>39.2</td><td>26.6</td><td>18.4</td><td>25.4</td><td>20.4</td><td>16.5</td><td>12.4</td><td>8.1</td><td>8.6</td><td>8.2</td><td>14.0</td></tr><tr><td colspan="13">StreamPro Framework</td></tr><tr><td>StreamPro-SFT</td><td>3B</td><td>65.8</td><td>38.7</td><td>37.2</td><td>38.6</td><td>37.6</td><td>25.8</td><td>14.9</td><td>5.4</td><td>11.3</td><td>6.4</td><td>21.1</td></tr><tr><td>StreamPro-GRPO</td><td>3B</td><td>86.2</td><td>64.3</td><td>40.2</td><td>62.8</td><td>46.2</td><td>36.1</td><td>32.1</td><td>19.8</td><td>24.5</td><td>21.0</td><td>32.9</td></tr><tr><td>StreamPro-SFT</td><td>4B</td><td>69.4</td><td>51.4</td><td>50.6</td><td>51.3</td><td>50.7</td><td>23.3</td><td>15.6</td><td>6.3</td><td>11.8</td><td>7.4</td><td>27.9</td></tr><tr><td>StreamPro-GRPO</td><td>4B</td><td>83.3</td><td>71.6</td><td>54.8</td><td>69.0</td><td>58.8</td><td>53.4</td><td>50.5</td><td>29.0</td><td>38.1</td><td>31.4</td><td>44.4</td></tr></table>


Table 13: Detailed Results on Temporal Reasoning.


<table><tr><td rowspan="2">Methods</td><td rowspan="2">Params</td><td colspan="5">Goal Planning</td><td colspan="5">Risk Forecasting</td><td rowspan="2">W-Avg F1</td></tr><tr><td>Time</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Time</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td colspan="13">Offline Models + Proactive Prompt</td></tr><tr><td>Qwen2.5-VL-7B [24]</td><td>7B</td><td>9.8</td><td>9.3</td><td>7.5</td><td>5.5</td><td>5.8</td><td>5.1</td><td>2.9</td><td>9.0</td><td>2.0</td><td>3.0</td><td>0.9</td></tr><tr><td>Qwen3-VL-8B [25]</td><td>8B</td><td>58.7</td><td>30.7</td><td>4.8</td><td>29.1</td><td>4.3</td><td>7.3</td><td>4.0</td><td>2.9</td><td>3.3</td><td>1.8</td><td>2.6</td></tr><tr><td colspan="13">Open-Source Proactive Models</td></tr><tr><td>VideoLLM-Online [1]</td><td>8B</td><td>20.6</td><td>10.5</td><td>47.4</td><td>10.4</td><td>16.9</td><td>22.4</td><td>5.5</td><td>0.9</td><td>4.8</td><td>0.7</td><td>5.7</td></tr><tr><td>MMDuet [2]</td><td>7B</td><td>30.2</td><td>16.1</td><td>2.1</td><td>9.3</td><td>3.4</td><td>36.9</td><td>15.5</td><td>3.1</td><td>9.3</td><td>4.2</td><td>4.0</td></tr><tr><td>MMDuet2 [3]</td><td>3B</td><td>40.2</td><td>15.4</td><td>6.4</td><td>12.4</td><td>6.5</td><td>14.4</td><td>3.9</td><td>0.6</td><td>2.9</td><td>1.0</td><td>2.7</td></tr><tr><td>MiniCPM-o-4.5 [23]</td><td>9B</td><td>9.8</td><td>9.3</td><td>7.5</td><td>5.5</td><td>5.8</td><td>5.1</td><td>2.9</td><td>9.0</td><td>2.0</td><td>3.0</td><td>3.8</td></tr><tr><td>Streamo [11]</td><td>3B</td><td>17.2</td><td>7.7</td><td>15.5</td><td>7.6</td><td>8.8</td><td>1.1</td><td>0.3</td><td>0.0</td><td>0.3</td><td>0.0</td><td>2.7</td></tr><tr><td colspan="13">StreamPro Framework</td></tr><tr><td>StreamPro-SFT</td><td>3B</td><td>44.6</td><td>26.8</td><td>4.6</td><td>20.7</td><td>7.1</td><td>39.4</td><td>13.9</td><td>0.9</td><td>9.7</td><td>1.5</td><td>3.2</td></tr><tr><td>StreamPro-GRPO</td><td>3B</td><td>22.1</td><td>16.4</td><td>5.9</td><td>11.5</td><td>7.7</td><td>10.0</td><td>4.1</td><td>5.8</td><td>2.6</td><td>2.7</td><td>4.2</td></tr><tr><td>StreamPro-SFT</td><td>4B</td><td>37.2</td><td>25.6</td><td>6.8</td><td>20.9</td><td>8.8</td><td>50.1</td><td>18.2</td><td>3.2</td><td>13.0</td><td>4.3</td><td>5.7</td></tr><tr><td>StreamPro-GRPO</td><td>4B</td><td>28.0</td><td>23.0</td><td>8.6</td><td>15.1</td><td>10.6</td><td>13.6</td><td>8.2</td><td>8.9</td><td>5.6</td><td>6.3</td><td>7.6</td></tr></table>


Table 14: Detailed Results on Proactive Agency.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Param</td><td colspan="11">Real-Time Visual Understanding</td><td colspan="4">Contextual Understanding</td><td>Overall</td></tr><tr><td>OP</td><td>CR</td><td>CS</td><td>ATP</td><td>EU</td><td>TR</td><td>PR</td><td>SU</td><td>ACP</td><td>CT</td><td>Avg.</td><td>ACU</td><td>MCU</td><td>SQA</td><td>Avg.</td><td>Avg.</td></tr><tr><td colspan="18">Human</td></tr><tr><td>Human</td><td>-</td><td>89.5</td><td>92.0</td><td>93.6</td><td>91.5</td><td>95.7</td><td>92.5</td><td>88.0</td><td>88.8</td><td>89.7</td><td>91.3</td><td>91.5</td><td>88.8</td><td>90.4</td><td>95.0</td><td>91.4</td><td>91.5</td></tr><tr><td colspan="18">Offline Models</td></tr><tr><td>Gemini 1.5 pro [41]</td><td>-</td><td>79.0</td><td>80.5</td><td>83.5</td><td>79.7</td><td>80.0</td><td>84.7</td><td>77.8</td><td>64.2</td><td>72.0</td><td>48.7</td><td>75.7</td><td>51.4</td><td>40.7</td><td>54.8</td><td>49.0</td><td>69.5</td></tr><tr><td>GPT-4o [42]</td><td>-</td><td>77.1</td><td>80.5</td><td>83.9</td><td>76.5</td><td>70.2</td><td>83.8</td><td>66.7</td><td>62.2</td><td>69.1</td><td>49.2</td><td>73.3</td><td>41.2</td><td>38.4</td><td>32.8</td><td>37.5</td><td>65.0</td></tr><tr><td>LLaVA-OneVision [43]</td><td>7B</td><td>80.4</td><td>74.2</td><td>76.0</td><td>80.7</td><td>72.7</td><td>71.7</td><td>67.6</td><td>65.4</td><td>65.7</td><td>45.1</td><td>71.1</td><td>35.6</td><td>36.0</td><td>27.3</td><td>33.0</td><td>62.3</td></tr><tr><td>Qwen2-VL-7B [44]</td><td>7B</td><td>75.2</td><td>82.8</td><td>73.2</td><td>77.5</td><td>68.3</td><td>71.0</td><td>72.2</td><td>61.2</td><td>61.5</td><td>46.1</td><td>69.0</td><td>31.2</td><td>26.0</td><td>39.6</td><td>32.3</td><td>60.5</td></tr><tr><td>InternVL2-8B [45]</td><td>8B</td><td>68.1</td><td>60.9</td><td>69.4</td><td>77.1</td><td>67.7</td><td>62.9</td><td>59.3</td><td>53.3</td><td>55.0</td><td>56.5</td><td>63.7</td><td>32.0</td><td>31.2</td><td>32.3</td><td>31.8</td><td>56.3</td></tr><tr><td colspan="18">Streaming Models</td></tr><tr><td>VideoLLM-Online [1]</td><td>8B</td><td>39.1</td><td>40.1</td><td>34.5</td><td>31.1</td><td>45.5</td><td>32.4</td><td>31.5</td><td>34.2</td><td>42.5</td><td>27.9</td><td>36.0</td><td>24.2</td><td>29.2</td><td>30.8</td><td>28.1</td><td>34.2</td></tr><tr><td>Flash-VStream [8]</td><td>7B</td><td>25.9</td><td>43.6</td><td>24.9</td><td>23.9</td><td>27.3</td><td>13.1</td><td>18.5</td><td>25.2</td><td>23.9</td><td>48.7</td><td>23.2</td><td>24.8</td><td>25.2</td><td>26.8</td><td>25.6</td><td>23.8</td></tr><tr><td>Dispider [4]</td><td>7B</td><td>74.9</td><td>75.5</td><td>74.1</td><td>73.1</td><td>74.4</td><td>59.9</td><td>76.1</td><td>62.9</td><td>62.2</td><td>45.8</td><td>67.6</td><td>39.6</td><td>27.7</td><td>34.8</td><td>34.0</td><td>59.8</td></tr><tr><td>TimeChat-Online [14]</td><td>7B</td><td>80.8</td><td>79.7</td><td>80.8</td><td>83.3</td><td>74.8</td><td>78.8</td><td>78.7</td><td>64.2</td><td>68.8</td><td>58.0</td><td>75.3</td><td>41.2</td><td>30.4</td><td>42.8</td><td>38.1</td><td>66.7</td></tr><tr><td>ViSpeak [10]</td><td>7B</td><td>79.8</td><td>88.3</td><td>83.3</td><td>81.1</td><td>76.4</td><td>75.1</td><td>70.4</td><td>65.9</td><td>77.3</td><td>34.2</td><td>74.4</td><td>38.8</td><td>36.8</td><td>44.0</td><td>39.9</td><td>66.4</td></tr><tr><td>StreamBridge [7]</td><td>7B</td><td>84.7</td><td>82.7</td><td>88.9</td><td>89.8</td><td>77.4</td><td>85.4</td><td>84.3</td><td>69.9</td><td>71.7</td><td>35.8</td><td>77.0</td><td>14.0</td><td>17.2</td><td>48.0</td><td>26.4</td><td>65.3</td></tr><tr><td>StreamForest [6]</td><td>7B</td><td>83.1</td><td>82.8</td><td>82.7</td><td>84.3</td><td>77.5</td><td>78.2</td><td>76.9</td><td>69.1</td><td>75.6</td><td>54.4</td><td>77.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>StreamAgent [5]</td><td>7B</td><td>79.6</td><td>78.3</td><td>79.3</td><td>75.9</td><td>74.7</td><td>76.9</td><td>82.9</td><td>66.3</td><td>73.7</td><td>55.4</td><td>74.3</td><td>39.7</td><td>30.3</td><td>39.6</td><td>36.5</td><td>65.6</td></tr><tr><td>QueryStream [12]</td><td>7B</td><td>82.1</td><td>83.6</td><td>78.2</td><td>82.7</td><td>75.5</td><td>80.1</td><td>79.6</td><td>63.0</td><td>67.9</td><td>42.6</td><td>74.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Streamo [11]</td><td>3B</td><td>83.7</td><td>80.5</td><td>83.9</td><td>84.6</td><td>80.5</td><td>81.0</td><td>71.3</td><td>67.5</td><td>71.6</td><td>37.2</td><td>75.8</td><td>43.6</td><td>38.4</td><td>41.2</td><td>41.1</td><td>67.8</td></tr><tr><td colspan="18">StreamPro Framework</td></tr><tr><td>StreamPro-SFT</td><td>3B</td><td>82.7</td><td>79.7</td><td>80.1</td><td>84.6</td><td>81.1</td><td>82.2</td><td>76.9</td><td>69.9</td><td>76.7</td><td>44.7</td><td>77.1</td><td>42.4</td><td>34.8</td><td>42.8</td><td>40.0</td><td>68.5</td></tr><tr><td>StreamPro-GRPO</td><td>3B</td><td>81.6</td><td>81.2</td><td>81.4</td><td>84.0</td><td>81.8</td><td>82.9</td><td>73.2</td><td>69.5</td><td>76.1</td><td>36.7</td><td>76.3</td><td>42.4</td><td>37.2</td><td>40.8</td><td>40.1</td><td>68.0</td></tr><tr><td>StreamPro-SFT</td><td>4B</td><td>86.7</td><td>78.9</td><td>88.0</td><td>84.9</td><td>83.0</td><td>86.0</td><td>82.4</td><td>74.0</td><td>78.7</td><td>33.0</td><td>79.3</td><td>48.8</td><td>43.6</td><td>48.4</td><td>46.9</td><td>71.8</td></tr><tr><td>StreamPro-GRPO</td><td>4B</td><td>85.4</td><td>77.3</td><td>87.7</td><td>84.0</td><td>84.3</td><td>84.4</td><td>77.8</td><td>73.2</td><td>78.7</td><td>50.0</td><td>79.8</td><td>53.6</td><td>43.2</td><td>43.2</td><td>46.7</td><td>72.1</td></tr></table>


Table 15: Evaluation results on StreamingBench.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Param</td><td colspan="7">Real-Time Visual Perception</td><td colspan="4">Backward Tracing</td><td>Overall</td></tr><tr><td>OCR</td><td>ACR</td><td>ATR</td><td>STU</td><td>FPD</td><td>OJR</td><td>Avg.</td><td>EPM</td><td>ASI</td><td>HLD</td><td>Avg.</td><td>Avg.</td></tr><tr><td colspan="14">Human</td></tr><tr><td>Human</td><td>-</td><td>94.0</td><td>92.6</td><td>94.8</td><td>92.7</td><td>91.1</td><td>94.0</td><td>93.2</td><td>92.6</td><td>93.0</td><td>91.4</td><td>92.3</td><td>92.8</td></tr><tr><td colspan="14">Offline Models</td></tr><tr><td>Gemini 1.5 Pro [41]</td><td>-</td><td>85.9</td><td>67.0</td><td>79.3</td><td>58.4</td><td>63.4</td><td>62.0</td><td>69.3</td><td>58.6</td><td>76.4</td><td>52.6</td><td>62.5</td><td>66.4</td></tr><tr><td>GPT-4o [42]</td><td>-</td><td>69.8</td><td>64.2</td><td>71.6</td><td>51.1</td><td>70.3</td><td>59.8</td><td>64.5</td><td>57.9</td><td>75.7</td><td>48.7</td><td>60.8</td><td>62.9</td></tr><tr><td>LLaVA-Video [36]</td><td>7B</td><td>69.1</td><td>58.7</td><td>68.8</td><td>49.4</td><td>74.3</td><td>59.8</td><td>63.5</td><td>56.2</td><td>57.4</td><td>7.5</td><td>40.4</td><td>53.6</td></tr><tr><td>LLaVA-OneVision [43]</td><td>7B</td><td>66.4</td><td>57.8</td><td>73.3</td><td>53.4</td><td>71.3</td><td>62.0</td><td>64.0</td><td>54.2</td><td>55.4</td><td>21.5</td><td>43.7</td><td>55.3</td></tr><tr><td>Qwen2-VL-7B [44]</td><td>7B</td><td>60.4</td><td>50.5</td><td>56.0</td><td>47.2</td><td>66.3</td><td>55.4</td><td>56.0</td><td>47.8</td><td>35.5</td><td>56.1</td><td>46.5</td><td>51.9</td></tr><tr><td>InternVL2-8B [45]</td><td>8B</td><td>67.1</td><td>60.6</td><td>63.8</td><td>46.1</td><td>68.3</td><td>56.5</td><td>60.4</td><td>48.2</td><td>57.4</td><td>24.7</td><td>43.4</td><td>53.1</td></tr><tr><td colspan="14">Streaming Models</td></tr><tr><td>VideoLLM-online [1]</td><td>8B</td><td>8.1</td><td>23.9</td><td>12.1</td><td>14.0</td><td>45.5</td><td>21.2</td><td>20.8</td><td>22.2</td><td>18.8</td><td>12.2</td><td>17.7</td><td>19.5</td></tr><tr><td>Flash-VStream [8]</td><td>7B</td><td>24.2</td><td>29.4</td><td>28.5</td><td>33.7</td><td>25.7</td><td>28.8</td><td>28.4</td><td>39.1</td><td>37.2</td><td>5.9</td><td>27.4</td><td>28.0</td></tr><tr><td>Dispider [4]</td><td>7B</td><td>57.7</td><td>49.5</td><td>62.1</td><td>44.9</td><td>61.4</td><td>51.6</td><td>54.6</td><td>48.5</td><td>55.4</td><td>4.3</td><td>36.1</td><td>46.7</td></tr><tr><td>TimeChat-Online [14]</td><td>7B</td><td>69.8</td><td>48.6</td><td>64.7</td><td>44.9</td><td>68.3</td><td>55.4</td><td>58.6</td><td>53.9</td><td>62.8</td><td>9.1</td><td>42.0</td><td>51.5</td></tr><tr><td>ViSpeak [10]</td><td>7B</td><td>75.2</td><td>58.7</td><td>71.6</td><td>51.1</td><td>74.3</td><td>66.9</td><td>66.3</td><td>59.9</td><td>48.7</td><td>64.0</td><td>57.5</td><td>62.5</td></tr><tr><td>StreamBridge [7]</td><td>7B</td><td>84.6</td><td>71.6</td><td>74.1</td><td>49.4</td><td>75.3</td><td>72.8</td><td>71.3</td><td>67.7</td><td>57.4</td><td>79.0</td><td>68.1</td><td>69.9</td></tr><tr><td>StreamForest [6]</td><td>7B</td><td>68.5</td><td>53.2</td><td>71.6</td><td>47.8</td><td>65.4</td><td>60.9</td><td>61.2</td><td>58.9</td><td>64.9</td><td>32.3</td><td>52.0</td><td>57.3</td></tr><tr><td>StreamAgent [5]</td><td>7B</td><td>71.2</td><td>53.2</td><td>63.6</td><td>53.9</td><td>67.3</td><td>58.7</td><td>61.3</td><td>54.8</td><td>58.1</td><td>25.8</td><td>41.7</td><td>52.9</td></tr><tr><td>QueryStream [12]</td><td>7B</td><td>74.5</td><td>47.7</td><td>70.7</td><td>46.6</td><td>71.3</td><td>57.6</td><td>61.4</td><td>54.2</td><td>63.5</td><td>8.6</td><td>42.1</td><td>53.1</td></tr><tr><td>Thinking-QwenVL [13]</td><td>7B</td><td>74.1</td><td>57.2</td><td>68.1</td><td>55.3</td><td>75.0</td><td>58.3</td><td>64.7</td><td>48.0</td><td>56.3</td><td>28.8</td><td>44.3</td><td>55.9</td></tr><tr><td>Streamo-3B [11]</td><td>3B</td><td>73.8</td><td>51.4</td><td>74.1</td><td>47.2</td><td>57.4</td><td>63.0</td><td>60.9</td><td>51.0</td><td>57.4</td><td>10.2</td><td>40.5</td><td>52.1</td></tr><tr><td>Streamo-7B [11]</td><td>7B</td><td>79.2</td><td>57.8</td><td>75.0</td><td>49.4</td><td>64.4</td><td>70.1</td><td>66.0</td><td>54.6</td><td>52.0</td><td>31.7</td><td>46.1</td><td>57.4</td></tr><tr><td colspan="14">StreamPro Framework</td></tr><tr><td>StreamPro-SFT</td><td>3B</td><td>73.2</td><td>51.4</td><td>75.0</td><td>50.6</td><td>58.4</td><td>64.1</td><td>62.0</td><td>51.0</td><td>59.5</td><td>7.0</td><td>40.0</td><td>52.5</td></tr><tr><td>StreamPro-GRPO</td><td>3B</td><td>74.5</td><td>57.8</td><td>74.1</td><td>50.6</td><td>64.4</td><td>65.2</td><td>63.9</td><td>46.3</td><td>44.6</td><td>7.0</td><td>34.3</td><td>51.2</td></tr><tr><td>StreamPro-SFT</td><td>4B</td><td>79.2</td><td>68.8</td><td>69.0</td><td>53.9</td><td>66.3</td><td>71.7</td><td>67.9</td><td>60.8</td><td>69.6</td><td>3.8</td><td>46.0</td><td>58.5</td></tr><tr><td>StreamPro-GRPO</td><td>4B</td><td>82.6</td><td>67.9</td><td>73.3</td><td>55.1</td><td>68.3</td><td>71.2</td><td>69.3</td><td>61.1</td><td>67.6</td><td>4.8</td><td>46.0</td><td>59.3</td></tr></table>


Table 16: Evaluation results on OVO-Bench.


## D Case Study

## D.1 Benchmark Examples

We present additional examples of the benchmark tasks, as shown in Figure 8, Figure 9, and Figure 10. 

## D.2 Model Comparison

We compare the outputs of different models to comprehensively evaluate the proactive capabilities of StreamPro, as shown in Figure 11, 12, 13, and 14. StreamPro demonstrates strong performance on both object understanding and temporal grounding tasks, consistently producing accurate and welltimed responses. In contrast, for the risk forecasting examples, all existing models still struggle to achieve satisfactory results, reflecting the limited proactive agency capability in this more challenging setting. 

![](images/d73d4a3e561e00b481661cb0686b048df9bf24ed7b7d0438d49d33cd02e14805.jpg)



Figure 8: Benchmark samples of Temporal Reasoning tasks


![](images/77d6177cc2f132ac1a615d6129912a664cb7f43dc0e89a75b76e20e4c5152175.jpg)



Figure 9: Benchmark samples of Perceptual Understanding tasks


![](images/bca4af493899eb0918e2bd579ec657f7e2e7551cddbac92bb8e1f5993b100ad8.jpg)



Figure 10: Benchmark samples of Proactive Agency tasks.


![](images/11d96a1a31e88aa9930d90c2d132a01a9d58ad996a9e9cd002c1b67f5f7f56f8.jpg)



Figure 11: Comparison of different models on object understanding tasks.


![](images/58510a70a079fbd31202402262cc8ea619df916fef6e66f714f91254d1180070.jpg)



Figure 12: Comparison of different models on temporal grounding tasks.


![](images/6f66d5c0432d422267148911b619eb4ca714cfd82c1246ce6d49e243325ea1d2.jpg)



Figure 13: Comparison of different models on risk forecasting tasks.


![](images/c4fab936b4d2865be8cd3d85e0c4374c5bc668aa21bae9557c5c5a4dd1968464.jpg)



Figure 14: Comparison of different models on risk forecasting tasks.


## E Prompt Design

We present the system prompt used in StreamPro framework, the prompt used for evaluating offline models, the prompt for rubric generation, and the prompt used for open-ended evaluation. 

## System Prompt

You are a helpful assistant specializing in streaming video analysis. You will receive input frame by frame, where each frame is labeled with absolute time intervals in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow the rules below strictly: Response Protocol: 

• Use </Silence> when: 

– A relevant event has not yet fully completed, or 

– The current input is irrelevant to the given question. 

• Use </Response> only when: 

– An event has fully concluded, or 

– The available information is sufficient to fully answer the question. 

At this point, provide a complete and final answer. 

## Additional Constraints:

• Do not provide partial answers. 

• Do not speculate beyond the given information. 

• Every valid answer must begin with </Response>. 

## Offline Proactive Decision Prompt

You are a Streaming Video Analysis Expert. You will receive a video sequence frame-by-frame. A user Query will be issued once at an arbitrary time during the video stream. Before the Query is issued, observe the video carefully. After the Query is issued, follow the decision logic below. 

Objective: Determine whether the visual information observed so far (from the beginning of the video up to the current moment) is sufficient to answer the user’s Query. 

Decision Logic: For each input frame sequence after the Query has been issued: 

1. Is the visual information observed so far sufficient to answer the Query without guessing? 

• If NO → reply Wait. 

• If YES → proceed to Step 2. 

2. Does the inferred answer differ from the last answer you provided? 

• If NO → reply Wait. 

• If YES (First Trigger) → reply with the actual answer. 

• If YES (Answer Update due to new evidence) → reply with the updated answer. 

## Output Constraint:

• Do not output anything other than Wait or the actual content of the answer. 

## Rubric Generation Prompt

You are designing evaluation rubrics for a streaming video QA system. The rubrics will be used to evaluate model trajectories AFTER generation — the model’s trajectory is not available yet. You see the ground truth and the video. 

## Input:

• Question: {question} 

• Ground Truth Answer(s): {ground_truth_text} 

Task: Design 6–10 binary (pass/fail) evaluation rubrics that assess a model’s streaming trajectory quality. Each rubric must be SPECIFIC to this particular video and question — not generic criteria. 

## Important Requirements:

• Each rubric must be an ASSERTIVE statement: 

– Use words like must, shall not, is prohibited. 

– Do NOT write vague descriptions (e.g., “the model should do X well”). 

– Write concrete assertions (e.g., “The response must include Y — missing this is not allowed”). 

## Rubric Dimensions (each rubric covers one):

## • Granularity:

– Assess whether step decomposition matches the natural structure of the video. 

– The model shall not over-split atomic actions. 

– The model shall not merge distinct actions into vague descriptions. 

## • Coherence:

– Steps must follow correct temporal and physical logic. 

– The model shall not skip necessary intermediate steps. 

## • Coverage:

– All essential actions or phases must be included. 

– Missing key stages is not allowed. 

## Post-check (self-review):

• Each rubric must tightly match the specific video and QA pair. 

• No rubric should be generic or broadly applicable. 

• All three dimensions must be covered. 

• Total number must be between 6 and 10. 

• Each rubric must be written as a concrete assertion. 

## Output Format: Return ONLY the final JSON array:

```json
[
    {
    "rubric_id": 0,
    "dimension": "granularity|coherence|coverage",
    "rubric": "Assertive statement about what to check"
    }
] 
```

## Open-Ended Evaluation Prompt

You are an expert evaluator judging whether a model’s answer provides a reasonable and factually correct response that directly addresses the question, based on the reference answer. 

## Evaluation Guideline:

• Focus on whether the model provides information that directly answers the question. 

• The answer does not need to reproduce all details from the reference—it only needs to provide factually correct and relevant information. 

• An answer that captures the essential information required by the question should be considered strong, even if it omits descriptive details. 

• Accept simplified, rephrased, or high-level responses as long as they are consistent with the reference and do not contradict known facts. 

• Do not deduct points for omitting secondary or illustrative details when the key information needed to answer the question is present, or for using concise or abstract phrasing. 

• Only penalize if the answer is factually wrong, fails to provide relevant information, or is so vague that it does not actually answer the question. 

## Scoring (integer 0–5):

• 5: Fully accurate and complete answer. 

• 4: Correct and sufficiently informative answer; may omit non-essential details but contains the key information. 

• 3: Partially relevant but missing some important information. 

• 2: Tangential or speculative without solid factual grounding. 

• 1: Factually incorrect. 

• 0: No attempt to answer or completely off-topic. 

## Output Format:

• Return a valid JSON object with exactly two keys: 

– "explanation": one sentence focusing on whether the answer provides correct and relevant information for the question 

– "score": an integer from 0 to 5 

• Output only the JSON. No other text, markdown, or commentary. 

## Inputs:

• Question: {question} 

• Predicted Answer: {model_output} 

• Correct Answer: {reference_answer} 

## F Limitations

Although StreamPro demonstrates strong performance across proactive tasks and real-time streaming tasks, it still has several limitations. First, this work primarily focuses on proactive capability and does not incorporate dedicated memory mechanisms to improve inference efficiency; instead, we adopt a simple sliding-window strategy to mitigate latency and GPU memory overhead, which may still limit its practicality in real-world deployments. Second, the current framework is limited to video and text modalities and does not extend to an omni-modal setting. As a result, it cannot effectively leverage complementary information from other modalities—particularly audio cues such as speech, environmental sounds, and temporal acoustic patterns—which are often critical for comprehensive scene understanding and timely decision-making in real-world scenarios. In future work, we plan to enhance inference efficiency and extend the model to incorporate audio modality. 

## G Societal Impact

This work can benefit real-time video understanding systems such as assistive agents and autonomous monitoring by enabling more timely and context-aware responses. However, like other video understanding technologies, it may also raise privacy concerns if applied to continuous surveillance or user behavior analysis without appropriate safeguards. 

## H Acknowledgment

This paper utilizes a suite of models, datasets and benchmarks, including TimeChat-Online-139K [14] (available at: https://huggingface.co/datasets/yaolily/TimeChat-Online-139K, with license details provided in its repository), ET-Instruct-164K [35] (available at: https://huggingface.co/datasets/PolyU-ChenLab/ET-Instruct-164K, licensed under CC-BY-NC-SA 4.0), MSAD [37] (available at: https://msad-dataset.github.io/, licensed under CC-BY-NC-SA 4.0), LongVideoBench [30] (available at: https://github.com/longvideobench/LongVideoBench, licensed under CC-BY-NC-SA 4.0), VideoMME [29] (available at: https://github.com/MME-Benchmarks/Video-MME, with license details provided in its repository), and Streamo [11]. 

We confirm that the use of all aforementioned models and datasets is strictly limited to academic research purposes and does not involve any commercial use. 