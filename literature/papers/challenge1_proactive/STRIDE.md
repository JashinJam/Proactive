# STRIDE: When to Speak Meets Sequence Denoising for Streaming Video Understanding

Junho Kim<sup>1∗</sup>, Hosu Lee<sup>2∗</sup>, James M. Rehg<sup>1</sup>, Minsu Kim<sup>3†‡</sup>, Yong Man Ro<sup>2†</sup> 

<sup>1</sup> UIUC, <sup>2</sup> KAIST, <sup>3</sup> Google DeepMind 

<sup>∗</sup>Equal contribution, <sup>†</sup>Corresponding author, <sup>‡</sup>Work done as an advisory role only. 

Recent progress in video large language models (Video-LLMs) has enabled strong ofline reasoning over long and complex videos. However, real-world deployments increasingly require streaming perception and proactive interaction, where video frames arrive online and the system must decide not only what to respond, but also when to respond. In this work, we revisit proactive activation in streaming video as a structured sequence modeling problem, motivated by the observation that temporal transitions in streaming video naturally form span-structured activation patterns. To capture this span-level structure, we model activation signals jointly over a sliding temporal window and update them iteratively as new frames arrive. We propose STRIDE (Structured Temporal Refinement with Iterative DEnoising), which employs a lightweight masked difusion module at the activation interface to jointly predict and progressively refine activation signals across the window. Extensive experiments on diverse streaming benchmarks and downstream models demonstrate that STRIDE shows more reliable and temporally coherent proactive responses, significantly improving whento-speak decision quality in online streaming scenarios. 

Contact: arkimjh@illinois.edu, leehosu01@kaist.ac.kr Project Page: https://interlive-team.github.io/STRIDE Huggingface: https://huggingface.co/interlive Code: https://github.com/interlive-team/STRIDE 

## 1 Introduction

Along with recent advances in large language models (LLMs) Brown et al. (2020); Touvron et al. (2023); OpenAI (2022); Reid et al. (2024); Yang et al. (2025a), large vision-language models (LVLMs) Li et al. (2023); Liu et al. (2023b); Dai et al. (2023); Liu et al. (2023a); Chen et al. (2023) have also achieved impressive performance across a wide range of image understanding and reasoning tasks. Building upon these advances, various video specialized models (i.e., Video-LLMs) Lin et al. (2023); Zhang et al. (2023); Kim et al. (2024); Zhang et al. (2024a); Li et al. (2025c) further extend them to the temporal sequences, demonstrating remarkable capabilities in reasoning over video contents. However, existing Video-LLMs mostly operate in an ofline manner, processing pre-recorded videos with access to the entire temporal context before generating responses. This fundamentally limits their capabilities to real-world streaming deployments such as egocentric assistants Huang et al. (2024c), autonomous driving Xie et al. (2025), or embodied AI agents Wei et al. (2025), where the model must continuously perceive an ongoing video stream and decide when and what to respond in real time. 

Recognizing this gap, recent works have delved into streaming video understanding (SVU), where models continu ously ingest incoming frames and maintain a temporal understanding on-the-fly Wang et al. (2024d); Zhang et al. (2025b); Yang et al. (2025c); Ning et al. (2025); Yao et al. (2025); Zhang et al.. Despite these advances, the approach is still reactive, lacking a capability to determine when a response should be triggered. Expanding beyond the streaming scope, several works have explored proactive response generation by leveraging special tokens Chen et al. (2024a, 2025a); Xu et al. (2025) to implicitly learn response timing or an agent-driven interaction approach Xiong et al. (2025); Yang et al. (2025b). More recently, several standalone activation modules Qian et al. (2024, 2025); Wang et al. (2025a) have been proposed, especially those that decouple the streaming pipeline into two stages: a lightweight front-end that predicts activation signals at each frame to identify triggering moments, followed by a downstream Video-LLM that, when activated, consumes the accumulated frame cache to generate responses. 

Within this decomposed framework, a straightforward way to train the activation module is to treat it as a binary classification problem as in Qian et al. (2024, 2025); Wang et al. (2025a), where at each time step a model predicts whether to trigger a response under binary supervision. However, such approach reduces activation to point-wise 0/1 decisions, answering “should I respond now?” at each time step, without explicitly modeling how activation states transition across a temporal span. This often results in flickering activations and poorly resolved transition boundaries, causing unstable triggering behavior and fragmented activation spans. In practice, a reliable activation module must not only predict isolated labels, but also model how activation states change over time, capturing consistent 0→1 onsets, sustained 1→1 persistence, and well-resolved 1→0 ofsets, so as to form coherent contiguous activation spans. In this sense, streaming and proactive triggering is more analogous to a span-structured decision rather than a point-wise one. To account for this span-level structure, an activation module should jointly model the activation sequence within a temporal neighborhood, so that the downstream Video-LLM can be activated under well-scoped visual context (neither prematurely with insuficient evidence nor too late after the moment has passed). 

Motivated by recent advances in masked difusion models Nie et al. (2025); You et al. (2025); Li et al. (2025a) (MDMs), which enable joint prediction over partially masked discrete sequences, we revisit streaming and proactive activation as structured sequence modeling over an activation window. Unlike point-wise decision-making, masked difusion op erates on an entire sequence and iteratively refines corrupted states within context, naturally aligning with the spanstructure of streaming trigger. Building on this, we propose STRIDE (Structured T emporal Refinement with Iterative DEnoising), a proactive streaming framework that models the when-to-speak decision as structured sequence predic tion, explicitly capturing span-level structure and activation state transitions. Specifically, during training, we employ boundary-aware span masking strategies that corrupt contiguous regions of the activation sequence, encouraging the model to reason about onset and ofset from broader temporal context rather than relying on isolated binary signals. At inference time, as new frames arrive, STRIDE progressively updates the activation window by carrying forward confident states and remasking uncertain positions, enabling temporally coherent span under partial observability while remaining plug-and-play and compatible with of-the-shelf Video-LLMs. 

Through extensive experiments and comprehensive analyses on streaming benchmarks and downstream models, we corroborate that STRIDE produces more reliable and temporally coherent proactive responses in online settings, significantly improving the when-to-speak decisions. 

Our contributions can be summarized as follows: 

• We revisit proactive streaming activation in Video-LLMs and reformulate the when-to-speak problem as structured sequence modeling over a temporal activation window, establishing span-level activation as the predic tion unit. 

• We propose STRIDE (Structured Temporal Refinement with Iterative DE noising), a lightweight masked difusionbased activation model that jointly predicts activation sequences and captures span-level structure. 

• We validate STRIDE through extensive experiments on diverse streaming benchmarks and downstream backbones, demonstrating more stable proactive triggering and improved temporal consistency in online settings. 

## 2 Related Work

## 2.1 Large Vision-Language Models

Early works on LVLMs Liu et al. (2023b); Dai et al. (2023); Li et al. (2024a) have demonstrated that visual instruction tuning, which pairs a vision encoder with a LLM backbone and trains on instruction-following data, can output strong general-purpose capabilities for back-and-forth multi-modal conversation. Subsequent eforts Chen et al. (2023); Wang et al. (2024c); Zhu et al. (2025b); Wang et al. (2025b) have focused on scaling model and data, improving visual tokenization, and aligning vision and language representations at scale. Especially, Qwen families Bai et al. (2023); Wang et al. (2024b); Bai et al. (2025b,a) improve visual processing eficiency and capability with dynamic resolution and stronger multi-modal pretraining, enabling more robust perception and reasoning over complex visual inputs. In addition, Video-LLMs Zhang et al. (2023); Li et al. (2024c); Song et al. (2024); Zhang et al. (2024a) extend its scope to temporal understanding by treating video as a sequence of images, introducing video-specific connector Lin et al. (2023); Kim et al. (2024); Zhang et al. (2025a) and training pipelines Li et al. (2024b); Share (2024); Zhang et al. (2024b) that better capture spatiotemporal dynamics, thereby leading to stronger performance on video QA and captioning tasks. Despite these advances, most LVLMs remain confined to an ofline setting, where the entire video clip is available prior to inference, limiting their applicability in real-time streaming scenarios. 

## 2.2 Streaming Video Understanding

A growing body of works Qian et al. (2024); Zhang et al. (2025c); Li et al. (2025b) has explored expanding video understanding into the streaming regime, where frames arrive online and frameworks must maintain state over time. One line of research adapts models to streaming interaction by redesigning training objectives and data formats for contin uous inputs Chen et al. (2024a), incorporating memory-augmented architectures for multi-turn streaming Zhang et al. (2025b); Xiong et al. (2025), and leveraging real-time commentary pipelines that integrate video speech transcripts with instruction tuning Chen et al. (2025a); Xu et al. (2025). Another branch emphasizes eficiency for unbounded video streams through memory aggregation for long streams Zhang et al. (2025b), streaming-aligned KV-cache strategies Xu et al. (2025); Ning et al. (2025); Yang et al. (2025c), and redundant visual token dropping based on inter-frame similarity Yao et al. (2025). While these approaches have enabled Video-LLMs to process continuous streams, they remain fundamentally reactive, generating responses only upon instantaneous user queries. 

Addressing this gap, another direction tackles the proactive response, which targets deciding when to respond as the video unfolds. Several approaches exploit EOS token within autoregressive generation to implicitly determine response timing Chen et al. (2024a); Xu et al. (2025), conflating the triggering with language generation. Agentic methods explicitly model task-relevant temporal intervals for goal-driven triggering Yang et al. (2025b), or query aware visual pruning with proactive response mechanisms Zhang et al.. Most relevant to our work, recent modular approaches Qian et al. (2024, 2025); Wang et al. (2025a, 2024d) explicitly decouple the pipeline into a lightweight front end that predicts per-frame binary activation signals and a downstream Video-LLM that generates responses upon triggering. While such a modular design preserves the downstream Video-LLM’s capabilities, reducing activation to point-wise binary supervision undermines the temporal coherence of contiguous activation spans. In this work, we retain the modular design while recasting activation as a structured sequence prediction problem, leveraging masked difusion to jointly model activation sequences over a temporal window and capture span-level temporal coherence. 

## 2.3 Discrete Diffusion Language Models

Recent progress in discrete difusion language models (dLLMs) Nie et al. (2025); Sahoo et al. (2024); Lou et al. (2023) revisits difusion as an alternative to autoregressive decoding for text generation using masked difusion models mechanism. Instead of generating tokens strictly left-to-right, dLLMs iteratively denoise masked token sequences, enabling bidirectional conditioning and parallel token updates, which naturally supports controllable generation. Subsequent eforts have further scaled dLLMs by converting pretrained autoregressive models into difusion-based counterparts Gong et al. (2024); Ye et al. (2025), and improved their alignment and inference eficiency through parallel decoding strategies Chen et al. (2025b). Especially, LLaDA series scale masked difusion to large LLMs Nie et al. (2025) and further explores post-training alignment Zhu et al. (2025a) as well as system-level scaling by converting pretrained AR models into difusion models Bie et al. (2025), thereby inheriting knowledge while retaining the nonautoregressive generation benefits. This research scope has also been extended to the multi-modal setting, where vision encoders are coupled with difusion language backbones for visual instruction following Li et al. (2025a); You et al. (2025); Yu et al. (2025); Cheng et al. (2025), demonstrating that dLLMs can benefit from parallel decoding and bidirectional reasoning in vision-language tasks. Diferent from these works that primarily replace the autoregressive decoder for textual response generation, our work leverages masked difusion for proactive streaming activation. We treat the when-to-speak signal as a structured discrete activation sequence over a temporal window, jointly predicting the activation states for the incoming video streams. 

## 3 Proposed Method

## 3.0.1 Preliminaries: Masked Diffusion Models.

Recently, difusion language models (dLLMs) Nie et al. (2025); Zhu et al. (2025a); Li et al. (2025a); You et al. (2025) have shown remarkable progress as an alternative paradigm to autoregressive language modeling, replacing left-toright token generation with a masked difusion process that iteratively denoises discrete token sequences. Given a sequence of L tokens $\mathbf { x } _ { 0 } = ( x _ { 0 } ^ { 1 } , \ldots , x _ { 0 } ^ { L } )$ , the forward process progressively corrupts $\mathbf { x } _ { \mathrm { 0 } }$ by independently replacing each token with a mask token [M] with probability $t \in [ 0 , 1 ]$ , generating a partially masked sequence $\mathbf { x } _ { t } .$ At $t = 0$ the sequence is fully observed, while at $t = 1$ it is entirely masked. 

![](images/62f81ac6f97f6c3fd0ec199aede0f74fd9e32120713a30ad853f259548227894.jpg)



Figure 1 Overview of STRIDE, which operates in a streaming setting where frames arrive online. A lightweight activation model based on masked difusion maintains an activation region over a sliding temporal window and iteratively denoises masked activation states to predict a coherent trigger segment. A trigger is issued only if an active span is sustained for a predefined span ratio. When activation is triggered, the accumulated frame context is forwarded to a downstream Video-LLM to generate the response.


The core of MDMs is a mask predictor $p _ { \theta } ( \cdot \mid \mathbf { x } _ { t } )$ with bidirectional attention that takes $\mathbf { x } _ { t }$ as input and predicts all masked tokens simultaneously. The reverse process Austin et al. (2021); Shi et al. (2024); Sahoo et al. (2024) recovers $\mathbf { x } _ { \mathrm { 0 } }$ from $\mathbf { x } _ { t }$ by iteratively applying this mask predictor, which is trained by minimizing a cross-entropy loss computed only over the masked positions: 

$$
\mathcal {L} (\theta) = - \mathbb {E} _ {t, \mathbf {x} _ {0}, \mathbf {x} _ {t}} \left[ \frac {1}{t} \sum_ {i = 1} ^ {L} \mathbb {1} [ x _ {t} ^ {i} = \mathsf {M} ] \log p _ {\theta} (x _ {0} ^ {i} \mid \mathbf {x} _ {t}) \right],\tag{1}
$$

where $t \sim U [ 0 , 1 ]$ and $\mathbf { x } _ { t }$ is sampled from the forward process. This serves as an upper bound on the negative log-likelihood of the model distribution Nie et al. (2025); Bie et al. (2025). 

At inference, generation proceeds by initializing a fully masked sequence $\mathbf { x } _ { 1 }$ and simulating the reverse process through K discrete steps decreasing from $t = 1  0 . \mathrm { A t }$ each step, the mask predictor predicts all masked positions, and a subset of predictions is accepted while the remaining positions are remasked for subsequent refinement. This iterative predict-and-refine procedure enables MDMs to generate coherent sequences through progressive unmasking with bidirectional context. 

## 3.1 STRIDE: Proactive Streaming Framework

## 3.1.1 Problem Formulation.

The proposed STRIDE (shown in figure 1) considers the streaming video understanding setting where a model continuously processes video streams $\mathcal { V } { = } \{ v _ { 1 } , v _ { 2 } , \dotsc , v _ { T } , \dotsc \}$ with $v _ { T }$ denoting the incoming visual frame arriving at time step T , interleaved with user queries and model-generated responses over time. Unlike ofline Video-LLMs that have access to the holistic video sequences before generating a response, a streaming model must work under partial observability, where only the frames observed so far $\mathcal { V } _ { \le T } { = } \{ v _ { 1 } , \dots , v _ { T } \}$ and context priors $\mathcal { C } _ { T }$ (e.g., user query q and prior interaction history) are available. At every time step T , the model faces two sequential decisions: (i) whether to respond, and (ii) if so, what to respond. STRIDE adopts a two-stage streaming framework to decouple these decisions. 

## 3.1.2 Two-Stage Architecture.

As illustrated in figure 1, STRIDE is designed with the two-stage streaming framework. A lightweight Activation Model π continuously monitors the incoming stream and determines whether a proactive response should be triggered. Once a response is triggered at time step $T ,$ the accumulated visual context since the most recent query time $T _ { q } ,$ denoted $\gamma _ { [ T _ { q } : T ] }$ , together with the interaction context $\mathcal { C } _ { T }$ , is forwarded to a downstream Video-LLM, which generates the response $\mathcal { R } _ { T } ~ = ~ f ( \mathcal { C } _ { T } , \mathcal { V } _ { [ T _ { q } : T ] } )$ . The generated response $\mathcal { R } _ { T }$ is appended to the interaction context, updating it to $\mathcal { C } _ { T ^ { \prime } } = \mathcal { C } _ { T } \cup \mathcal { R } _ { T }$ , enabling awareness of prior responses and maintaining dialogue coherence across multiple activation events. After each triggered response, the visual accumulation is cleared and restarted from the current time step, ensuring that subsequent activation decisions operate on fresh streaming context. This modular design cleanly separates when-to-speak modeling from downstream response generation. 

![](images/4c733b908d10d8754cd7bc201f3b7399630d69b1f0e88e5430c2fdfc871bdfce.jpg)



Figure 2 Activation modeling and inference stage of STRIDE. Training applies sequence duplication and three masking strategies (boundary-anchored masking, span unmasking, full masking). During inference, the activation window slides with incoming frames, retaining confident past decisions while selectively re-masking and progressively denoising uncertain positions.


## 3.1.3 Span-Level Activation Modeling.

To formalize the activation decision, we represent activation as a window-level sequence of size W anchored at time step T , and model it as a sequence-level prediction over this temporal window. Specifically, we define an activation region $\mathsf { a } _ { T } = [ a _ { T - W } , \hdots , \grave { a _ { T } } ] \in \{ \theta , 1 \} ^ { \bar { W } }$ , indicating inactive or active states within the window. This windowed formulation enables the activation model to learn contiguous activation spans and their transition dynamics (0→1 onset, 1→1 persistence, 1→0 ofset), aligning the prediction unit with span-level structures rather than isolated point-wise decisions. 

As the video stream unfolds, incoming frames are sampled at 1 FPS and encoded into visual tokens by a vision encoder, which are accumulated in a running visual cache. At each time step T , the activation region ${ \pmb a } _ { T }$ is appended after the visual cache as the prediction target. Each activation token takes values from the discrete vocabulary {0, 1, [M]}, where [M] denotes masked positions to be denoised. The activation model conditions on the visual cache and jointly infers masked activation states within the temporal window. When the activation state is determined to be active un der the span-based criterion, the accumulated visual context is forwarded to the downstream Video-LLM for response generation. 

## 3.2 Training: Activation as Sequence Denoising

## 3.2.1 Structured Masking Strategies for Activation Denoising.

To train the activation model under the structured formulation, we propose a mixture of three corruption strategies instead of the standard MDM Nie et al. (2025), which samples mask positions independently. Such masking is inappropriate for our activation learning as the target sequence consists of contiguous active regions; isolated unmasked tokens between active positions make the denoising task trivially solvable through local interpolation, bypassing the need for genuine temporal understanding. The proposed masking mixture shown in figure 2 (left) is composed of: 

• Boundary-Anchored Span Masking masks a contiguous block overlapping with at least one activation boundary, forcing the model to determine where the active region begins and ends from broader temporal context. 

• Span Unmasking starts from a fully masked sequence and reveals a contiguous block while keeping boundaryadjacent positions masked, mimicking the inference-time pattern where high-confidence tokens are unmasked consecutively in homogeneous regions. 

• Full Masking initially masks the entire activation sequence (cold-start) to stabilize the reverse step by training the model to estimate the global activation layout from visual context alone. 

During training, each sample is randomly corrupted using one of three structured masking strategies, each selected with equal probability. These structured strategies encourage the model to reason over contiguous activation spans and their boundary transitions, rather than relying on isolated token predictions. As a result, the activation module learns span-level consistency that better aligns with the sequential and partial observability of streaming proactive triggering. 

## 3.2.2 Recovering Bidirectional Conditioning with Sequence Duplication.

Ma-sked difusion predicts masked positions using full-sequence context, whereas an AR-pretrained activation model is trained with causal attention that only exposes left context. We therefore introduce an input reparameterization that enables bidirectional conditioning without altering the underlying causal attention layers. Specifically, we employ sequence duplication, appending a copy of the activation region to form [a, $\pmb { \mathsf { a } } ^ { \prime } ] _ { : }$ , where the copy carry identical activation tokens but serve distinct roles. The duplicated sequence $\widehat { \mathbf { a } } ^ { \prime }$ produces difusion predictions, while a serves as a conditioning prefix under causal attention. Concretely, since a is entirely placed before ${ \sf a ^ { \prime } , }$ every token in $\widehat { \mathbf { a } } ^ { \prime }$ can access all positions of a as left-context, providing full-window visibility for denoising without modifying the causal attention mask. 

## 3.2.3 Training Objective.

Following the denoising process in equation (1), we train the activation module by minimizing the masked cross entropy loss over $\mathsf { a } ^ { \prime } ,$ conditioned on the user query q and the visual cache $\gamma _ { \le T }$ 

$$
\mathcal {L} (\theta) = - \mathbb {E} _ {t, \mathbf {a} _ {0} ^ {\prime}, \mathbf {a} _ {t} ^ {\prime}} \left[ \frac {1}{t} \sum_ {j = 1} ^ {W} \mathbb {1} [ a _ {t} ^ {\prime j} = \mathsf {M} ] \log p _ {\theta} (a _ {0} ^ {\prime j} \mid q, \mathcal {V} _ {\leq t}, \mathbf {a} _ {t} ^ {\prime}) \right],\tag{2}
$$

where $\mathsf { a } _ { 0 } ^ { \prime }$ is the ground-truth activation sequence, $\widehat { \mathbf { a } } _ { t } ^ { \prime }$ is obtained by applying aforementioned our masking strategies at noise level $t \sim U [ 0 , 1 ]$ , and the user query q along with the visual cache $\gamma _ { \leq t }$ serves as a fixed conditioning prefix, analogous to the prompt in the supervised fine-tuning of dLLMs Nie et al. (2025); Li et al. (2025a). 

## 3.3 Inference: Streaming as Progressive Unmasking

At inference time, STRIDE maintains a sliding activation window and performs progressive refinement as illustrated in figure 2 (right); confident past decisions are preserved, while uncertain and newly introduced positions are jointly refined with masked difusion. Concretely, new time step $T + 1$ proceeds in two stages: 

(i) Selective Re-masking: The activation sequence of size W is shifted forward so that the region falling outside the window is evicted and a new frame is appended, causing the activation sequence to advance in time. The fully resolved activation $a _ { T } ^ { j + 1 }$ previously assigned to position $j + 1$ at time $T$ now maps to position j at time $T + 1$ . To determine whether each carried-forward decision remains reasonable given the new visual evidence $v _ { T + 1 }$ , we apply a confidence-based retention: if $p _ { \theta } ( a _ { T + 1 } ^ { j } = a _ { T } ^ { j + 1 } \mid q , \mathcal { V } _ { \leq T + 1 } , \mathsf { a } _ { T + 1 } ) \stackrel {  } { > } \tau _ { \theta }$ , position j inherits its previous decision; otherwise, it is re-masked to [M] so that uncertain positions re-enter the denoising process alongside the newly appended slots. 

(ii) K-Step Progressive Denoising: The masked positions obtained from the previous stage, comprising both newly appended slots and low-confidence re-masked slots, are resolved over K denoising steps by prioritizing high-confidence positions first. At each step, the model computes the activation probability $p ^ { j } = p _ { \theta } ( a ^ { j } = 1 \mid q , \mathcal { V } _ { \leq T + 1 } , \mathsf { a } _ { T + 1 } )$ for every masked position and derives a confidence score $c ^ { j } = \operatorname* { m a x } ( p ^ { j } , 1 { - } p ^ { j } )$ , which measures how strongly the prediction leans toward either triggering or not. The top-k positions ranked by $c ^ { j }$ are unmasked, where $k = \lceil \bar { N } _ { \mathrm { i n i t } } / K \rceil$ and $N _ { \mathrm { i n i t } }$ is the number of masked positions established in stage (i), while the rest remain masked for subsequent refinement. 

By revealing high-confidence decisions first, this schedule establishes reliable temporal anchors that progressively stabilize the remaining ambiguous boundary regions. 

After K steps, the activation window is fully resolved. A trigger at time T + 1 is issued only if an active span is sustained for at least γ consecutive positions, where γ denotes the required span ratio for triggering. 

## 4 Experiments

## 4.1 Experimental Setting

## 4.1.1 Implementation & Training Details.

The activation model is initialized from a compact vision-language model using Qwen3-VL-2B Bai et al. (2025a) to minimize streaming overhead. The downstream Video-LLMs are kept frozen, ensuring full modularity between the two stages. Incoming video frames are sampled at 1 FPS and encoded into the visual cache as the stream progresses. For the denoising process, we adopt the low-confidence remasking strategy Nie et al. (2025) with K=8 sampling steps during inference. τ is set to 0.75, and γ is set to 1 following the benchmark evaluation protocol. The entire activation model is trained on a single node of 8 NVIDIA H100 GPUs, while evaluation is conducted on a single H100 GPU. Comprehensive hyperparameter settings and additional configurations are provided in the Appendix. 

For the training data, we curate a diverse collection of temporally annotated video datasets spanning multiple video understanding tasks, including dense video captioning Caba Heilbron et al. (2015); Liu et al. (2024b); Huang et al. (2024b), temporal activity detection Sigurdsson et al. (2016, 2018), grounded video QA Wang et al. (2024a), sequential step recognition Zhou et al. (2018), and moment localization Anne Hendricks et al. (2017). We convert each temporal annotation into a binary activation sequence aligned with the frame sampling rate, where frames within annotated spans are labeled as active (1) and the remaining frames as inactive (0). 

## 4.1.2 Benchmarks & Baselines.

We evaluate STRIDE on three complementary benchmarks. OVO-Bench Niu et al. (2025) assesses online video understanding across backward tracing, real-time visual perception, and forward active responding, where the model must delay its response until suficient future evidence is available. StreamingBench Lin et al. (2024b) evaluates streaming comprehension through 18 tasks spanning real-time visual understanding, omni-source understanding and contextual understanding including proactive output and sequential question answering. In addition, we evaluate on subsets of ET-Bench Liu et al. (2024b), including Temporal Video Grounding (TVG), Episodic Memory (EPM), Temporal Action Localization (TAL), Dense Video Captioning (DVC), and Step Localization and Captioning (SLC). This setup evaluates activation timing precision by measuring how accurately the model identifies event boundaries. For baselines, we compare against various ofline Video-LLMs, online streaming proactive models Chen et al. (2024a); Qian et al. (2025); Wang et al. (2025a), and proprietary models Reid et al. (2024); OpenAI (2024). In addition, we include Baseline-AR, which serves as the primary counterpart to STRIDE. Baseline-AR follows the same architecture and training setup as our method but replaces the masked difusion activation module with an autoregressive binary prediction head trained with BCE loss, following the activation formulation described in prior work Wang et al. (2025a). This setup isolates the activation modeling strategy, enabling a direct comparison between masked denoising and autoregressive binary prediction. 

## 4.2 Qualitative Results on Streaming Video Understanding

tables 1 and 2 present results on OVO-Bench and StreamingBench, respectively. The proposed STRIDE outperforms the autoregressive baseline (i.e., Baseline-AR) Wang et al. (2025a) by introducing the proposed masked denoising process. Furthermore, across all three downstream models Team et al. (2025); Wang et al. (2025b); Bai et al. (2025a) on OVO-Bench, STRIDE achieves significant gains in Forward Active Responding, which directly evaluates proactive when-to-speak control. This setting benefits from our span-structured prediction, which models response timing over temporal activation region rather than through independent per-frame decisions. STRIDE also consistently improves Real-Time Visual Perception, indicating that stable triggering allows the downstream Video-LLM to ingest well-scoped visual context at the appropriate moment. On StreamingBench, this advantage extends broadly across all three evaluation dimensions: Real-time Visual Understanding, Omni-Source Understanding, and Contextual Under standing, with the most notable improvements in Proactive Output (PO) subtask that requires the model to determine response timing without explicit timing cue. Together, these results suggest that the proposed framework reliably enhances both the precision of when to respond and the relevance of responses across diverse streaming conditions. 


Table 1 Evaluation results on OVO-Bench Niu et al. (2025). Baseline-AR uses autoregressive binary prediction. Ofline models follow the original single-turn protocol with segmented clips, whereas streaming methods process frames sequentially.


<table><tr><td rowspan="2">Method</td><td rowspan="2"># of Frames</td><td colspan="7">Real-Time Visual Perception</td><td colspan="4">Backward Tracing</td><td colspan="4">Forward Act. Responding</td><td>Overall</td></tr><tr><td>OCR</td><td>ACR</td><td>ATR</td><td>STU</td><td>FPD</td><td>OJR</td><td>Avg.</td><td>EPM</td><td>ASI</td><td>HLD</td><td>Avg.</td><td>REC</td><td>SSR</td><td>CRR</td><td>Avg.</td><td>Avg.</td></tr><tr><td colspan="18">Human</td></tr><tr><td>Human</td><td>-</td><td>93.96</td><td>92.57</td><td>94.83</td><td>92.70</td><td>91.09</td><td>94.02</td><td>93.20</td><td>92.59</td><td>93.02</td><td>91.37</td><td>92.33</td><td>95.48</td><td>89.67</td><td>93.56</td><td>92.90</td><td>92.81</td></tr><tr><td colspan="18">Proprietary Models (Offline), Single-Turn Evaluation</td></tr><tr><td>Gemini 1.5 Pro Reid et al. (2024)</td><td>1 FPS</td><td>85.91</td><td>66.97</td><td>79.31</td><td>58.43</td><td>63.37</td><td>61.96</td><td>69.32</td><td>58.59</td><td>76.35</td><td>52.64</td><td>62.54</td><td>35.53</td><td>74.24</td><td>61.67</td><td>57.15</td><td>63.00</td></tr><tr><td>GPT-4o OpenAI (2024)</td><td>64</td><td>69.80</td><td>64.22</td><td>71.55</td><td>51.12</td><td>70.30</td><td>59.78</td><td>64.46</td><td>57.91</td><td>75.68</td><td>48.66</td><td>60.75</td><td>27.58</td><td>73.21</td><td>59.40</td><td>53.40</td><td>59.54</td></tr><tr><td colspan="18">Open-Source Models (Offline), Single-Turn Evaluation</td></tr><tr><td>LLaVA-Video-7B Zhang et al. (2024b)</td><td>64</td><td>69.13</td><td>58.72</td><td>68.83</td><td>49.44</td><td>74.26</td><td>59.78</td><td>63.52</td><td>56.23</td><td>57.43</td><td>7.53</td><td>40.40</td><td>34.10</td><td>69.95</td><td>60.42</td><td>54.82</td><td>52.91</td></tr><tr><td>LLaVA-OV-7B Li et al. (2024a)</td><td>64</td><td>66.44</td><td>57.80</td><td>73.28</td><td>53.37</td><td>71.29</td><td>61.96</td><td>64.02</td><td>54.21</td><td>55.41</td><td>21.51</td><td>43.71</td><td>25.64</td><td>67.09</td><td>58.75</td><td>50.50</td><td>52.74</td></tr><tr><td>LLaVA-N-Video-7B Li et al. (2024b)</td><td>64</td><td>69.80</td><td>59.60</td><td>66.40</td><td>50.60</td><td>72.30</td><td>61.40</td><td>63.30</td><td>51.20</td><td>64.20</td><td>9.70</td><td>41.70</td><td>34.10</td><td>67.60</td><td>60.80</td><td>54.20</td><td>53.10</td></tr><tr><td>Qwen2-VL-7B Wang et al. (2024b)</td><td>64</td><td>60.40</td><td>50.46</td><td>56.03</td><td>47.19</td><td>66.34</td><td>55.43</td><td>55.98</td><td>47.81</td><td>35.48</td><td>56.08</td><td>46.46</td><td>31.66</td><td>65.82</td><td>48.75</td><td>48.74</td><td>50.39</td></tr><tr><td>InternVL-V2-8B Chen et al. (2024b)</td><td>64</td><td>67.11</td><td>60.55</td><td>63.79</td><td>46.07</td><td>68.32</td><td>56.52</td><td>60.39</td><td>48.15</td><td>57.43</td><td>24.73</td><td>43.44</td><td>26.50</td><td>59.14</td><td>54.14</td><td>46.60</td><td>50.15</td></tr><tr><td>LongVU-7B Shen et al. (2024)</td><td>1 FPS</td><td>55.70</td><td>49.50</td><td>59.50</td><td>48.30</td><td>68.30</td><td>63.00</td><td>57.40</td><td>43.10</td><td>66.20</td><td>9.10</td><td>39.50</td><td>16.60</td><td>69.00</td><td>60.00</td><td>48.50</td><td>48.50</td></tr><tr><td colspan="18">Open-Source Models (Streaming)</td></tr><tr><td>Flash-VStream-7B Zhang et al. (2025b)</td><td>1 FPS</td><td>24.16</td><td>29.36</td><td>28.45</td><td>33.71</td><td>25.74</td><td>28.80</td><td>28.37</td><td>39.06</td><td>37.16</td><td>5.91</td><td>27.38</td><td>8.02</td><td>67.25</td><td>60.00</td><td>45.09</td><td>33.61</td></tr><tr><td>VideoLLM-Online-8B Chen et al. (2024a)</td><td>2 FPS</td><td>8.05</td><td>23.85</td><td>12.07</td><td>14.04</td><td>45.54</td><td>21.20</td><td>20.79</td><td>22.22</td><td>18.80</td><td>12.18</td><td>17.73</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VideoLLM-EyeWO Zhang et al. (2025c)</td><td>1 FPS</td><td>24.16</td><td>27.52</td><td>31.89</td><td>32.58</td><td>44.55</td><td>35.87</td><td>32.76</td><td>39.06</td><td>38.51</td><td>6.45</td><td>28.00</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Dispider Qian et al. (2025)</td><td>1 FPS</td><td>57.72</td><td>49.54</td><td>62.07</td><td>44.94</td><td>61.39</td><td>51.63</td><td>54.55</td><td>48.48</td><td>55.41</td><td>4.30</td><td>36.06</td><td>18.05</td><td>37.36</td><td>48.75</td><td>34.72</td><td>41.78</td></tr><tr><td>TimeChat-Online-7B Yao et al. (2025)</td><td>1 FPS</td><td>69.80</td><td>48.60</td><td>64.70</td><td>44.90</td><td>68.30</td><td>55.40</td><td>58.60</td><td>53.90</td><td>62.80</td><td>9.10</td><td>42.00</td><td>32.50</td><td>36.50</td><td>40.00</td><td>36.40</td><td>45.60</td></tr><tr><td>StreamAgent-7B Yang et al. (2025b)</td><td>1 FPS</td><td>71.20</td><td>53.20</td><td>63.60</td><td>53.90</td><td>67.30</td><td>58.70</td><td>61.30</td><td>54.80</td><td>58.10</td><td>25.80</td><td>41.70</td><td>35.90</td><td>48.40</td><td>52.00</td><td>45.40</td><td>49.40</td></tr><tr><td>QueryStream-7B Zhang et al.</td><td>1 FPS</td><td>74.50</td><td>47.70</td><td>70.70</td><td>46.60</td><td>71.30</td><td>57.60</td><td>61.40</td><td>54.20</td><td>63.50</td><td>8.60</td><td>42.10</td><td>33.20</td><td>43.10</td><td>40.80</td><td>39.03</td><td>47.51</td></tr><tr><td colspan="18">Offline Backbones → Online Inference with STRIDE</td></tr><tr><td>Qwen3-VL-8B Bai et al. (2025a)</td><td>1 FPS</td><td>69.80</td><td>59.60</td><td>73.30</td><td>57.30</td><td>71.30</td><td>58.70</td><td>65.00</td><td>55.60</td><td>63.50</td><td>12.90</td><td>44.00</td><td>37.70</td><td>60.80</td><td>40.40</td><td>46.30</td><td>51.77</td></tr><tr><td>+ Baseline-AR Wang et al. (2025a)</td><td>1 FPS</td><td>73.80</td><td>65.10</td><td>73.30</td><td>62.40</td><td>70.30</td><td>71.20</td><td>69.35</td><td>54.90</td><td>66.90</td><td>17.20</td><td>46.33</td><td>29.70</td><td>56.00</td><td>42.50</td><td>42.73</td><td>52.81</td></tr><tr><td>Gemma3-4B Team et al. (2025)</td><td>1 FPS</td><td>65.80</td><td>48.60</td><td>56.00</td><td>36.00</td><td>66.30</td><td>50.00</td><td>53.78</td><td>44.40</td><td>41.90</td><td>3.20</td><td>29.83</td><td>14.40</td><td>61.40</td><td>52.50</td><td>42.77</td><td>42.13</td></tr><tr><td>+ STRIDE</td><td>1 FPS</td><td>73.20</td><td>60.60</td><td>64.70</td><td>39.30</td><td>71.30</td><td>56.50</td><td>60.93</td><td>47.80</td><td>52.00</td><td>4.80</td><td>34.87</td><td>42.60</td><td>64.60</td><td>60.00</td><td>55.73</td><td>50.51</td></tr><tr><td>InternVL3-8B Wang et al. (2025b)</td><td>1 FPS</td><td>65.80</td><td>52.30</td><td>68.10</td><td>51.10</td><td>71.30</td><td>62.00</td><td>61.77</td><td>58.90</td><td>66.90</td><td>9.70</td><td>45.17</td><td>36.60</td><td>64.10</td><td>43.30</td><td>48.00</td><td>51.64</td></tr><tr><td>+ STRIDE</td><td>1 FPS</td><td>75.80</td><td>54.10</td><td>80.20</td><td>56.70</td><td>74.30</td><td>65.20</td><td>67.72</td><td>58.90</td><td>65.50</td><td>11.30</td><td>45.23</td><td>40.10</td><td>67.70</td><td>66.20</td><td>58.00</td><td>56.98</td></tr><tr><td>Qwen3-VL-8B Bai et al. (2025a)</td><td>1 FPS</td><td>69.80</td><td>59.60</td><td>73.30</td><td>57.30</td><td>71.30</td><td>58.70</td><td>65.00</td><td>55.60</td><td>63.50</td><td>12.90</td><td>44.00</td><td>37.70</td><td>68.80</td><td>40.40</td><td>46.30</td><td>51.77</td></tr><tr><td>+ STRIDE</td><td>1 FPS</td><td>76.50</td><td>64.20</td><td>79.30</td><td>61.20</td><td>73.30</td><td>63.60</td><td>69.68</td><td>57.20</td><td>72.30</td><td>14.00</td><td>47.83</td><td>46.40</td><td>63.10</td><td>69.60</td><td>59.70</td><td>59.07</td></tr></table>

## 4.3 Activation Evaluation via Temporal Grounding

Accurate temporal grounding is central to proactive activation. While the previous benchmarks Niu et al. (2025); Lin et al. (2024b) evaluate the end-to-end behavior of streaming pipelines, they do not directly measure the quality of the activation model itself. To isolate this component, we evaluate the activation model independently on ET-Bench Liu et al. (2024b), which focuses on fine-grained event-level temporal understanding. As shown in table 3, the gain from replacing binary classification with masked difusion is substantial: STRIDE outperforms Baseline-AR by 27.1 on TVG and 8.3 on average, demonstrating that structured sequence denoising provides considerably sharper boundary resolution than per-frame supervision. Notably, STRIDE achieves these results with only 2B parameters, outperforming both streaming baselines and temporal-localization specialized MLLMs of standard size (7–13B parameters) on the overall average. 

## 4.4 Ablation Studies on STRIDE Components

## 4.4.1 Effects of Masking Strategies.

table 4(a) evaluates how diferent masking strategies afect the learning of span-structured activation sequences. The standard MDM protocol independent masking performs the worst across all metrics, indicating that activation prediction cannot be treated as independent point-wise denoising since it fails to capture the temporal structure of activation transitions. To better reflect the span-level nature of activation, we adopt three complementary masking patterns described in section 3.2: boundary-anchored span masking (Span), full masking (Full), and span unmasking (Span). Span masks contiguous regions near activation boundaries, Full masks the entire sequence to simulate the cold-start condition, and Span exposes boundary refinement patterns encountered during denoising. Combining these strategies significantly improves performance, suggesting that diverse span corruption patterns help the model learn coherent activation spans for better boundary prediction. 


Table 2 Evaluation results on StreamingBench Lin et al. (2024b). Baseline-AR uses autoregressive binary prediction. Ofline models follow the original single-turn protocol with segmented clips, whereas streaming methods process frames sequentially


<table><tr><td rowspan="2">Method</td><td rowspan="2"># of Frames</td><td colspan="11">Real-Time Visual Understanding</td><td colspan="5">Omni-Source Understanding</td><td colspan="5">Contextual Understanding</td><td>Overall</td></tr><tr><td>OP</td><td>CR</td><td>CS</td><td>ATP</td><td>EU</td><td>TR</td><td>PR</td><td>SU</td><td>ACP</td><td>CT</td><td>Avg.</td><td>ER</td><td>SCU</td><td>SD</td><td>MA</td><td>Avg.</td><td>ACU</td><td>MCU</td><td>SQA</td><td>PO</td><td>Avg.</td><td>Avg.</td></tr><tr><td>Human</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Human</td><td>-</td><td>89.47</td><td>92.00</td><td>93.60</td><td>91.47</td><td>95.65</td><td>92.52</td><td>88.00</td><td>88.75</td><td>89.74</td><td>91.30</td><td>91.46</td><td>88.00</td><td>88.24</td><td>93.60</td><td>90.27</td><td>90.26</td><td>88.80</td><td>90.40</td><td>95.00</td><td>100</td><td>93.55</td><td>91.66</td></tr><tr><td colspan="24">Proprietary Models (Offline)</td></tr><tr><td>Gemini 1.5 pro Reid et al. (2024)</td><td>1 FPS</td><td>79.02</td><td>80.47</td><td>83.54</td><td>79.67</td><td>80.00</td><td>84.74</td><td>77.78</td><td>64.23</td><td>71.95</td><td>48.70</td><td>75.69</td><td>46.80</td><td>39.60</td><td>74.90</td><td>80.00</td><td>60.22</td><td>51.41</td><td>40.73</td><td>54.80</td><td>45.10</td><td>48.73</td><td>67.07</td></tr><tr><td>GPT-4o OpenAI (2024)</td><td>64</td><td>77.11</td><td>80.47</td><td>83.91</td><td>76.47</td><td>70.19</td><td>83.80</td><td>66.67</td><td>62.19</td><td>69.12</td><td>49.22</td><td>73.28</td><td>41.20</td><td>37.20</td><td>43.60</td><td>56.00</td><td>44.50</td><td>41.20</td><td>38.40</td><td>32.80</td><td>56.86</td><td>38.70</td><td>60.15</td></tr><tr><td colspan="24">Open-Source Models (Offline)</td></tr><tr><td>LLaVA-OV-7B Li et al. (2024a)</td><td>32</td><td>80.38</td><td>74.22</td><td>76.03</td><td>80.72</td><td>72.67</td><td>71.65</td><td>67.59</td><td>65.45</td><td>65.72</td><td>45.08</td><td>71.12</td><td>40.80</td><td>37.20</td><td>33.60</td><td>44.80</td><td>38.40</td><td>35.60</td><td>36.00</td><td>27.27</td><td>29.55</td><td>32.74</td><td>56.36</td></tr><tr><td>Qwen2-VL-7B Wang et al. (2024b)</td><td>1 FPS</td><td>75.20</td><td>82.81</td><td>73.19</td><td>77.45</td><td>68.32</td><td>71.03</td><td>72.22</td><td>61.19</td><td>61.47</td><td>46.11</td><td>69.04</td><td>41.20</td><td>22.00</td><td>32.80</td><td>43.60</td><td>34.90</td><td>31.20</td><td>26.00</td><td>39.60</td><td>22.73</td><td>31.66</td><td>54.14</td></tr><tr><td>MiniCPM-V 2.6 8B Yao et al. (2024)</td><td>32</td><td>71.93</td><td>71.09</td><td>77.92</td><td>75.82</td><td>64.60</td><td>65.73</td><td>70.37</td><td>56.10</td><td>62.32</td><td>53.37</td><td>67.44</td><td>40.80</td><td>24.00</td><td>34.00</td><td>41.20</td><td>35.00</td><td>34.00</td><td>31.60</td><td>41.92</td><td>22.22</td><td>34.97</td><td>53.85</td></tr><tr><td>InternVL-V2-8B Chen et al. (2024b)</td><td>16</td><td>68.12</td><td>60.94</td><td>69.40</td><td>77.12</td><td>67.70</td><td>62.93</td><td>59.26</td><td>53.25</td><td>54.96</td><td>56.48</td><td>63.72</td><td>37.60</td><td>26.40</td><td>37.20</td><td>42.00</td><td>35.80</td><td>32.00</td><td>31.20</td><td>32.32</td><td>40.91</td><td>32.42</td><td>51.40</td></tr><tr><td>Kangaroo-7B Liu et al. (2024a)</td><td>64</td><td>71.12</td><td>84.38</td><td>70.66</td><td>73.20</td><td>67.08</td><td>61.68</td><td>56.48</td><td>55.69</td><td>62.04</td><td>38.86</td><td>64.60</td><td>37.60</td><td>31.20</td><td>28.80</td><td>39.20</td><td>34.20</td><td>32.80</td><td>26.40</td><td>33.84</td><td>16.00</td><td>30.06</td><td>51.10</td></tr><tr><td>LongVA-7B Zhang et al. (2024a)</td><td>128</td><td>70.03</td><td>63.28</td><td>61.20</td><td>70.92</td><td>62.73</td><td>59.50</td><td>61.11</td><td>53.66</td><td>54.67</td><td>34.72</td><td>59.96</td><td>39.60</td><td>32.40</td><td>28.00</td><td>41.60</td><td>35.40</td><td>32.80</td><td>29.60</td><td>30.30</td><td>15.91</td><td>29.95</td><td>48.66</td></tr><tr><td>VILA-1.5-8B Lin et al. (2024a)</td><td>14</td><td>53.68</td><td>49.22</td><td>70.98</td><td>56.86</td><td>53.42</td><td>53.89</td><td>54.63</td><td>48.78</td><td>50.14</td><td>17.62</td><td>52.32</td><td>41.60</td><td>26.40</td><td>28.40</td><td>36.00</td><td>33.10</td><td>26.80</td><td>34.00</td><td>23.23</td><td>17.65</td><td>27.35</td><td>43.20</td></tr><tr><td>Video-LLaMA2-7B Cheng et al. (2024)</td><td>32</td><td>55.86</td><td>55.47</td><td>57.41</td><td>58.17</td><td>52.80</td><td>43.61</td><td>39.81</td><td>42.68</td><td>45.61</td><td>35.23</td><td>49.52</td><td>30.40</td><td>32.40</td><td>30.40</td><td>36.00</td><td>32.40</td><td>24.80</td><td>26.80</td><td>18.67</td><td>0.00</td><td>21.93</td><td>40.40</td></tr><tr><td colspan="24">Open-Source Models (Streaming)</td></tr><tr><td>Flash-VStream-7B Zhang et al. (2025b)</td><td>1 FPS</td><td>25.89</td><td>43.57</td><td>24.91</td><td>23.87</td><td>27.33</td><td>13.08</td><td>18.52</td><td>25.20</td><td>23.87</td><td>48.70</td><td>23.23</td><td>25.91</td><td>24.90</td><td>25.60</td><td>28.40</td><td>26.00</td><td>24.80</td><td>25.20</td><td>26.80</td><td>1.96</td><td>24.12</td><td>24.04</td></tr><tr><td>VideoLLM-Online-8B Chen et al. (2024a)</td><td>2 FPS</td><td>39.07</td><td>40.06</td><td>34.49</td><td>31.05</td><td>45.54</td><td>32.40</td><td>31.48</td><td>34.16</td><td>42.49</td><td>27.89</td><td>35.99</td><td>31.20</td><td>26.51</td><td>24.10</td><td>32.00</td><td>28.45</td><td>24.19</td><td>29.20</td><td>30.80</td><td>3.92</td><td>26.55</td><td>32.48</td></tr><tr><td>Dispuser Qian et al. (2025)</td><td>1 FPS</td><td>74.92</td><td>75.53</td><td>74.10</td><td>73.08</td><td>74.44</td><td>59.92</td><td>76.14</td><td>62.91</td><td>62.16</td><td>45.80</td><td>67.63</td><td>35.46</td><td>25.26</td><td>38.57</td><td>43.34</td><td>35.66</td><td>39.62</td><td>27.65</td><td>34.80</td><td>25.34</td><td>33.61</td><td>53.12</td></tr><tr><td>StreamAgent Yang et al. (2025b)</td><td>1 FPS</td><td>79.63</td><td>78.31</td><td>79.28</td><td>75.87</td><td>74.74</td><td>76.92</td><td>82.94</td><td>66.31</td><td>73.69</td><td>55.40</td><td>74.31</td><td>35.86</td><td>26.26</td><td>38.87</td><td>44.04</td><td>36.26</td><td>39.72</td><td>30.25</td><td>39.60</td><td>28.90</td><td>34.62</td><td>57.02</td></tr><tr><td>TimeChat-Online-7B Yao et al. (2025)</td><td>1 FPS</td><td>80.80</td><td>79.70</td><td>80.80</td><td>83.30</td><td>74.80</td><td>78.80</td><td>78.70</td><td>64.20</td><td>68.80</td><td>58.00</td><td>75.28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>QueryStream-7B Zhang et al.</td><td>1 FPS</td><td>82.11</td><td>83.59</td><td>78.23</td><td>82.69</td><td>75.47</td><td>80.06</td><td>79.63</td><td>63.01</td><td>67.90</td><td>42.55</td><td>74.04</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="24">Offline Backbones → Online Inference with STRIDE</td></tr><tr><td>Qwen3-VL-8B Bai et al. (2025a)</td><td>1 FPS</td><td>62.70</td><td>68.00</td><td>69.70</td><td>53.30</td><td>67.50</td><td>65.10</td><td>67.60</td><td>48.00</td><td>68.00</td><td>40.90</td><td>60.88</td><td>36.80</td><td>18.00</td><td>32.80</td><td>34.00</td><td>30.40</td><td>25.60</td><td>23.60</td><td>31.20</td><td>32.40</td><td>28.20</td><td>46.84</td></tr><tr><td>+ Baseline-AR Wang et al. (2025a)</td><td>1 FPS</td><td>79.00</td><td>72.70</td><td>85.50</td><td>70.80</td><td>73.30</td><td>76.60</td><td>81.50</td><td>63.40</td><td>80.70</td><td>44.60</td><td>73.79</td><td>45.20</td><td>29.20</td><td>35.20</td><td>35.20</td><td>36.20</td><td>35.60</td><td>37.20</td><td>48.40</td><td>24.30</td><td>36.38</td><td>57.12</td></tr><tr><td>Gemma3-4B Team et al. (2025)</td><td>1 FPS</td><td>63.80</td><td>69.50</td><td>68.80</td><td>54.40</td><td>60.20</td><td>65.10</td><td>59.30</td><td>40.70</td><td>62.70</td><td>21.80</td><td>57.49</td><td>28.80</td><td>31.20</td><td>30.40</td><td>46.40</td><td>34.20</td><td>34.40</td><td>31.20</td><td>38.80</td><td>12.40</td><td>29.20</td><td>46.03</td></tr><tr><td>+ STRIDE</td><td>1 FPS</td><td>66.80</td><td>71.90</td><td>66.60</td><td>57.20</td><td>66.50</td><td>70.70</td><td>60.20</td><td>43.10</td><td>65.00</td><td>23.80</td><td>60.00</td><td>35.60</td><td>31.60</td><td>36.00</td><td>44.00</td><td>36.80</td><td>33.60</td><td>35.20</td><td>44.80</td><td>41.60</td><td>38.80</td><td>50.14</td></tr><tr><td>InternVL3-8B Wang et al. (2025b)</td><td>1 FPS</td><td>74.90</td><td>82.00</td><td>75.70</td><td>61.20</td><td>72.00</td><td>67.60</td><td>74.10</td><td>66.70</td><td>78.10</td><td>34.20</td><td>68.71</td><td>40.40</td><td>27.60</td><td>38.80</td><td>45.60</td><td>38.10</td><td>38.00</td><td>26.00</td><td>36.80</td><td>31.20</td><td>33.00</td><td>53.97</td></tr><tr><td>+ STRIDE</td><td>1 FPS</td><td>74.90</td><td>78.90</td><td>76.70</td><td>68.60</td><td>77.00</td><td>77.30</td><td>77.80</td><td>71.50</td><td>83.00</td><td>33.20</td><td>72.45</td><td>39.60</td><td>22.40</td><td>44.00</td><td>50.80</td><td>39.20</td><td>34.00</td><td>35.20</td><td>43.20</td><td>42.80</td><td>38.80</td><td>57.58</td></tr><tr><td>Qwen3-VL-8B Bai et al. (2025a)</td><td>1 FPS</td><td>62.70</td><td>68.00</td><td>69.70</td><td>53.30</td><td>67.50</td><td>65.10</td><td>67.60</td><td>48.00</td><td>68.00</td><td>40.90</td><td>60.88</td><td>36.80</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>


Table 3 Online activation accuracy on ET-Bench Liu et al. (2024b). The Baseline-AR uses autoregressive prediction, while other setups are the same as STRIDE.


<table><tr><td></td><td>Frames</td><td>Params</td><td><eq>TVG_{F1}</eq></td><td><eq>EPM_{F1}</eq></td><td><eq>TAL_{F1}</eq></td><td><eq>DVC_{F1}</eq></td><td><eq>SLC_{F1}</eq></td><td>Avg</td></tr><tr><td colspan="9">Temporal-Localization Specialized MLLMs</td></tr><tr><td>VTimeLLM Huang et al. (2024a)</td><td>100</td><td>7B</td><td>7.6</td><td>1.9</td><td>18.2</td><td>12.4</td><td>8.7</td><td>9.8</td></tr><tr><td>VTG-LLM Guo et al. (2025)</td><td>96</td><td>7B</td><td>15.9</td><td>3.7</td><td>14.4</td><td>40.2</td><td>20.8</td><td>19.0</td></tr><tr><td>TimeChat Ren et al. (2024)</td><td>96</td><td>7B</td><td>26.2</td><td>3.9</td><td>10.1</td><td>16.6</td><td>5.6</td><td>12.5</td></tr><tr><td>LITA Huang et al. (2024b)</td><td>100</td><td>13B</td><td>22.2</td><td>4.6</td><td>18.0</td><td>39.7</td><td>21.0</td><td>21.1</td></tr><tr><td>ETChat Liu et al. (2024b)</td><td>1 FPS</td><td>5B</td><td>38.6</td><td>10.2</td><td>30.8</td><td>38.4</td><td>24.4</td><td>28.5</td></tr><tr><td colspan="9">Streaming Baselines</td></tr><tr><td>VideoLLM-Online Chen et al. (2024a)</td><td>2 FPS</td><td>8B</td><td>13.2</td><td>3.8</td><td>9.1</td><td>24.0</td><td>9.9</td><td>12.0</td></tr><tr><td>Dispider Qian et al. (2025)</td><td>1 FPS</td><td>9B</td><td>36.1</td><td>15.5</td><td>27.3</td><td>33.8</td><td>18.8</td><td>26.3</td></tr><tr><td>StreamBridge Wang et al. (2025a)</td><td>1 FPS</td><td>8B</td><td>34.3</td><td>-</td><td>24.3</td><td>38.3</td><td>22.6</td><td>-</td></tr><tr><td>Baseline-AR Wang et al. (2025a)</td><td>1 FPS</td><td>2B</td><td>35.7</td><td>2.5</td><td>21.2</td><td>39.6</td><td>22.6</td><td>24.3</td></tr><tr><td>STRIDE</td><td>1 FPS</td><td>2B</td><td>62.8</td><td>10.7</td><td>24.6</td><td>36.5</td><td>28.5</td><td>32.6</td></tr></table>


Table 4 Ablation studies on ET-Bench evaluating (a) masking strategy design, (b) sequence duplication, and (c) selective re-masking.


<table><tr><td></td><td><eq>TVG_{F1}</eq></td><td><eq>EPM_{F1}</eq></td><td><eq>TAL_{F1}</eq></td><td><eq>DVC_{F1}</eq></td><td><eq>SLC_{F1}</eq></td><td>Avg</td></tr><tr><td colspan="7">(a) Masking Strategy</td></tr><tr><td>Indep. only</td><td>8.5</td><td>3.3</td><td>6.1</td><td>8.8</td><td>9.2</td><td>7.2</td></tr><tr><td>Span only</td><td>30.6</td><td>6.1</td><td>22.9</td><td>25.4</td><td>20.6</td><td>21.1</td></tr><tr><td>Span + Full</td><td>36.8</td><td>7.0</td><td>26.0</td><td>24.0</td><td>21.3</td><td>23.0</td></tr><tr><td>Span + Full + <eq>\overline{Span}</eq></td><td>62.8</td><td>10.7</td><td>24.6</td><td>36.5</td><td>28.5</td><td>32.6</td></tr><tr><td colspan="7">(b) Sequence Duplication</td></tr><tr><td>w/o Seq. Duplication</td><td>49.6</td><td>6.0</td><td>23.6</td><td>19.9</td><td>15.2</td><td>22.9</td></tr><tr><td>w/ Seq. Duplication</td><td>62.8</td><td>10.7</td><td>24.6</td><td>36.5</td><td>28.5</td><td>32.6</td></tr><tr><td colspan="7">(c) Selective Re-masking</td></tr><tr><td>w/o Re-masking (last-only)</td><td>39.5</td><td>2.5</td><td>19.1</td><td>30.7</td><td>21.2</td><td>22.6</td></tr><tr><td>w/ Re-masking (selective)</td><td>62.8</td><td>10.7</td><td>24.6</td><td>36.5</td><td>28.5</td><td>32.6</td></tr></table>

## 4.4.2 Effect of Sequence Duplication.

Masked difusion relies on contextual reasoning over the activation window, whereas the pretrained backbone used in STRIDE follows causal attention and therefore only exposes left context during prediction. This mismatch limits the model’s ability to jointly infer activation states within the window. To mitigate this, we apply sequence duplication, which provides full-window context to the prediction tokens while preserving the causal backbone. As shown in table 4(b), removing sequence duplication leads to a consistent performance drop across all tasks, reducing the average score from 32.6 to 22.9. The degradation is particularly notable in temporally sensitive tasks such as TVG and DVC, indicating that accurate boundary reasoning benefits from access to the full activation window. These results demonstrate that sequence duplication efectively recovers bidirectional context for difusion-based refine ment, enabling full-window conditioning through a simple input reparameterization without modifying the causal architecture. 

## 4.4.3 Effect of Selective Re-masking.

In the streaming setting, activation predictions are carried forward as the window advances. If these states are preserved without revision, early mistakes can propagate and gradually corrupt the activation sequence. To examine this efect, we compare our selective re-masking strategy with a simplified variant that masks only the newly appended position (last-only), leaving previous decisions fixed. As shown in table 4(c), restricting re-masking to the last position leads to a substantial performance drop, reducing the average score from 32.6 to 22.6. As only predicting last token falls into autoregressive prediction, the resulting performance is also similar to Baseline-AR in table 3. In contrast, selectively re-masking low-confidence positions allows the model to revise uncertain decisions as new frames arrive, enabling refinement of the activation sequence by using the updated context information. 

![](images/86a8493db32960c199d592cc7beda076fb5b8dea4c9e1197ffe26ce0a8d6e284.jpg)


(a) Pre-Event

![](images/87c5eedbc765ead367b5e360f0fa0a0cab8a6186bb3d97a0a5cc1e00fec7c089.jpg)


(b) During Event

![](images/482d4d21d9c6c19a626da32137b10bb5983b3818e99673ea9c26474311b13e28.jpg)


(c) Post-Event

Figure 3 Activation transition frequency results around event boundaries on ET-Bench TVG. Baseline-AR model shows frequent oscillations near boundaries, whereas STRIDE produces more robust activation spans.

## 4.5 Behavioral Analysis of STRIDE Properties

## 4.5.1 Flickering Analysis around Event Boundaries.

While ET-Bench quantifies activation accuracy, it is also limited to capture the temporal stability of activation deci sions. In particular, per-frame activation models may sufer from flickering behavior due to their inherently isolated predictions, where predictions rapidly oscillate between active and inactive states (0↔1), resulting in unstable triggering and poorly resolved transition boundaries. To analyze this phenomenon, we measure the frequency of activation transitions relative to event boundaries. Specifically, we align predictions around ground-truth events and accumulate transition counts within three regions (figure 3): (a) pre-event (-60s to onset), (b) during-event (normalized by event progress %), and (c) post-event (ofset to +60s), using the TVG task of ET-Bench where each instance corresponds to a single event. 

As in the figure, across all regions, Baseline-AR exhibits substantially higher transition frequency, indicating unstable activation sequences with frequent on/of oscillations. This instability becomes particularly striking near event boundaries, where transition frequency sharply increases, suggesting dificulty in resolving the precise onset and ofset of events. In contrast, STRIDE produces significantly smoother activation patterns with far fewer transitions. The reduced flickering indicates that modeling activation as structured sequence denoising encourages temporally coherent predictions, allowing the model to maintain consistent activation spans and more reliably capture event boundaries. 

## 4.5.2 Latency–Accuracy Trade-off for Denoising Steps.

We analyze the efect of the denoising step K on both activation model accuracy and inference latency as illustrated in figure 4. This trade-of is particularly important in the streaming setting, where the activation model operates online and directly determines the model’s response latency. Increasing K allows the model to perform more refinement steps, improving activation accuracy but also increasing inference time. In practice, we observe that performance saturates quickly: around K = 8 steps already achieves near-maximum mean F1 across ET-Bench subtasks. This behavior likely stems from the small output space of the activation sequence, where each position only takes binary states (0 or 1), making the denoising process relatively easier to converge than large vocabulary space. At K = 8, the inference latency is approximately ∼100 ms, which is practical enough to support real-time operation for streaming frame rates of downstream models. 

## 4.5.3 Streaming Efficiency and Memory Footprint.

Extending the latency-accuracy trade-of analysis in figure 4, we decompose the computational overhead introduced by the activation model under a streaming setup. The measurement is conducted on a single H100 GPU with a 128- frame context budget. As shown in table 5, when a subsequent response is required, the 113 ms (new frame and K = 8 denoising steps) added by STRIDE incurs only a 7% additional latency compared to the 1511 ms required by the base model Qwen3VL-8B Bai et al. (2025a) without the triggering module. When a trigger is unnecessary, STRIDE saves approximately 91% of the total processing time (113 ms vs. 1276 ms). Furthermore, compared to the per-frame decision baseline (Baseline-AR, 26 ms), the extra latency from the difusion process (113 ms) is limited to 87 ms. In terms of memory, STRIDE maintains a lightweight footprint of 5.2 GB. Executing denoising process requires an additional 10 MB, and each new frame introduces 30 MB of incremental memory usage. These highlight the advantage of the twostage design: a lightweight activation model gates the expensive downstream model. Even with the masked difusion module employed in STRIDE, trigger modeling introduces only minimal latency and memory overhead, maintaining eficient streaming inference. 

![](images/1da142edd13030389de10fb2fe3d66b87eb3d878ab30daee68f46922d423beb3.jpg)


Figure 4 Trade-of between ET-Bench performance (mean F1) and inference latency for denoising step K.

Table 5 Latency and VRAM usage of the downstream Video-LLM and STRIDE activation modules (with AR variation) during streaming inference.

<table><tr><td>Procedure</td><td>Latency (ms)</td><td>VRAM</td></tr><tr><td colspan="3">Downstream Video-LLM</td></tr><tr><td>Response Gen. (TTFT)</td><td>1276</td><td>17.8 GB</td></tr><tr><td>Response Gen. (TTLT)</td><td>1511</td><td>+ 13 MB</td></tr><tr><td colspan="3">STRIDE</td></tr><tr><td>Activation Sate (Base)</td><td></td><td>5.2 GB</td></tr><tr><td>+ 1 Denoising Step</td><td>12</td><td>+ 10 MB</td></tr><tr><td>+ Append Frame</td><td>20</td><td>+ 30 MB</td></tr><tr><td colspan="3">Baseline-AR</td></tr><tr><td>Activation Sate (Base)</td><td></td><td>5.2 GB</td></tr><tr><td>+ Append Frame</td><td>26</td><td>+ 30 MB</td></tr></table>

## 5 Conclusion

We present STRIDE, a framework for proactive streaming video understanding that models activation as a structured temporal sequence rather than independent per-frame decisions. By leveraging a lightweight masked difusion module to jointly refine activation signals over a sliding window, STRIDE captures span-level temporal structure and produces more stable and coherent triggering behavior in streaming settings. Extensive experiments and analyses show that jointly modeling activation over a temporal window significantly improves event boundary localization and reduces unstable triggering, while introducing only minimal overhead to the overall streaming pipeline. 

## References



Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2017. 





Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising difusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021. 





Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 1 (2):3, 2023. 





Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a. 





Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b. 





Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up difusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025. 





Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 





Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 





Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18407–18418, 2024a. 





Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, and Mike Zheng Shou. Livecc: Learning video llm with streaming speech transcription at scale. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29083–29095, 2025a. 





Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 





Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024b. 





Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. dparallel: Learnable parallel decoding for dllms. arXiv preprint arXiv:2509.26488, 2025b. 





Shuang Cheng, Yuhua Jiang, Zineng Zhou, Dawei Liu, Wang Tao, Linfeng Zhang, Biqing Qi, and Bowen Zhou. Sdar-vl: Stable and eficient block-wise difusion for vision-language understanding. arXiv preprint arXiv:2512.14068, 2025. 





Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 





Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems, 2023. 





Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling difusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024. 





Yongxin Guo, Jingyu Liu, Mingda Li, Dingxin Cheng, Xiaoying Tang, Dianbo Sui, Qingbin Liu, Xi Chen, and Kevin Zhao. Vtgllm: Integrating timestamp knowledge into video llms for enhanced video temporal grounding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 3302–3310, 2025. 





Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14271–14280, 2024a. 





De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. In European Conference on Computer Vision, pages 202–218. Springer, 2024b. 





Yifei Huang, Jilan Xu, Baoqi Pei, Yuping He, Guo Chen, Lijin Yang, Xinyuan Chen, Yaohui Wang, Zheng Nie, Jinyao Liu, et al. Vinci: A real-time embodied smart assistant based on egocentric vision-language model. arXiv preprint arXiv:2412.21080, 2024c. 





Junho Kim, Hyunjun Kim, Hosu Lee, and Yong Man Ro. Salova: Segment-augmented long video assistant for targeted retrieval and routing in long-form video analysis. arXiv preprint arXiv:2411.16173, 2024. 





Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a. 





Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024b. 





Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning. PMLR, 2023. 





Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024c. 





Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai Wei Chang, and Aditya Grover. Lavida: A large difusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025a. 





Wei Li, Bing Hu, Rui Shao, Leyang Shen, and Liqiang Nie. Lion-fs: Fast & slow video-language thinker as online video assistant. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3240–3251, 2025b. 





Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2025c. 





Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 





Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024a. 





Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024b. 





Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, 2023b. 





Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024a. 





Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Ying Shan, and Chang W Chen. Et bench: Towards open-ended event-level video-language understanding. Advances in Neural Information Processing Systems, 37:32076–32110, 2024b. 





Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete difusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023. 





Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language difusion models. arXiv preprint arXiv:2502.09992, 2025. 





Zhenyu Ning, Guangda Liu, Qihao Jin, Wenchao Ding, Minyi Guo, and Jieru Zhao. Livevlm: Eficient online video understanding via streaming-oriented kv cache and retrieval. arXiv preprint arXiv:2505.15269, 2025. 





Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, et al. Ovo-bench: How far is your video-llms from real-world online video understanding? In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18902–18913, 2025. 





OpenAI. ChatGPT. https://openai.com/blog/chatgpt/, 2022. 





OpenAI. Hello gpt-4o, 2024. URL https://openai.com/index/hello-gpt-4o/. 





Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understand ing with large language models. Advances in Neural Information Processing Systems, 37:119336–119360, 2024. 





Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24045–24055, 2025. 





Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, An geliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 





Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14313–14323, 2024. 





Subham Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. Advances in Neural Information Processing Systems, 37: 130136–130184, 2024. 





Share. Sharegemini: Scaling up video caption data for multimodal large language models, June 2024. URL https://github.com Share14/ShareGemini. 





Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 





Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked difusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024. 





Gunnar A Sigurdsson, Gül Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In European conference on computer vision, pages 510–526. Springer, 2016. 





Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Charades-ego: A large-scale dataset of paired third and first person videos. arXiv preprint arXiv:1804.09626, 2018. 





Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 





Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 





Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024a. 





Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, and Ping Huang. Streambridge: Turning your ofline video large language model into a proactive streaming assistant. arXiv preprint arXiv:2505.05467, 2025a. 





Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b. 





Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024c. 





Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265, 2025b. 





Yueqian Wang, Xiaojun Meng, Yuxuan Wang, Jianxin Liang, Jiansheng Wei, Huishuai Zhang, and Dongyan Zhao. Videollm knows when to speak: Enhancing time-sensitive video comprehension with video-text duet interaction format. arXiv preprint arXiv:2411.17991, 2024d. 





Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, et al. Streamvln: Streaming vision-and-language navigation via slowfast context modeling. arXiv preprint arXiv:2507.05240, 2025. 





Bin Xie, Yingfei Liu, Tiancai Wang, Jiale Cao, and Xiangyu Zhang. Glad: A streaming scene generator for autonomous driving. arXiv preprint arXiv:2503.00045, 2025. 





Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, and Huchuan Lu. Streaming video understanding and multi-round interaction with memory-enhanced knowledge. arXiv preprint arXiv:2501.13468, 2025. 





Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608, 2025. 





An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a. 





Haolin Yang, Feilong Tang, Lingxiao Zhao, Xiang An, Ming Hu, Huifa Li, Xinlin Zhuang, Yifan Lu, Xiaofeng Zhang, Abdalla Swikir, et al. Streamagent: Towards anticipatory agents for streaming video understanding. arXiv preprint arXiv:2508.01875, 2025b. 





Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, and Mengye Ren. Streammem: Query-agnostic kv cache memory for streaming video understanding. arXiv preprint arXiv:2508.15717, 2025c. 





Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, et al. Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10807–10816, 2025. 





Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 





Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Difusion large language models. arXiv preprint arXiv:2508.15487, 2025. 





Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language difusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025. 





Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete difusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025. 





Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025a. 





Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 





Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, and Xiaojie Jin. Flash-vstream: Eficient real-time understanding for long video streams. arXiv preprint arXiv:2506.23825, 2025b. 





Kairui Zhang, Zhenyu Yang, Bing Wang, Shengsheng Qian, and Changsheng Xu. Querystream: Advancing streaming video understanding with query-aware pruning and proactive response. In The Fourteenth International Conference on Learning Representations. 





Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024a. 





Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024b. 





Yulin Zhang, Cheng Shi, Yang Wang, and Sibei Yang. Eyes wide open: Ego proactive video-llm for streaming video. arXiv preprint arXiv:2510.14560, 2025c. 





Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018. 





Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language difusion models. arXiv preprint arXiv:2505.19223, 2025a. 





Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. In ternvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025b. 



## Appendix Contents

A Detailed Training Setup for STRIDE 16  
A.1 Training Dataset Configuration 16  
A.2 Training Hyperparameters 17  
B Detailed Architecture and Inference of STRIDE 18  
B.1 Masked Diffusion Formulation for Activation Span Modeling 18  
B.2 Training Objective for Activation Diffusion 18  
B.3 Inference Procedure for Activation Span Prediction 19  
C Detailed Benchmark and Evaluation Setup 19  
C.1 Additional Benchmark Explanation 19  
C.2 Comparison with Autoregressive Baseline 20  
D Additional Experimental Analysis 21  
D.1 Sensitivity Analysis for $\tau$ 21  
D.2 Scalability Analysis for Activation Backbone 21  
D.3 Qualitative Examples of STRIDE 22  
E Limitation and Discussion 22  
E.1 Failure Cases and Discussion 22 

## A Detailed Training Setup for STRIDE

## A.1 Training Dataset Configuration

We build the training data for activation span modeling by collecting and carefully curating seven publicly available video understanding datasets Caba Heilbron et al. (2015); Liu et al. (2024b); Sigurdsson et al. (2016, 2018); Wang et al. (2024a); Anne Hendricks et al. (2017); Liu et al. (2024b) that provide temporal annotations of events or actions. These datasets cover tasks such as dense video captioning, temporal activity localization, grounded video question answering, and procedural understanding. For each dataset, we use the provided temporal boundaries of events to construct activation spans that indicate when a relevant event occurs in the video. 

To make the data suitable for our objective, we reorganize the annotations into a unified format where each training sample consists of a video, a query describing the event of interest, and the corresponding temporal span defined by the event start and end times. Based on this span, we construct an activation signal over the video timeline: frames (or tokens) that fall within the annotated event span are labeled as 1 (active), while all other positions are labeled as 0 (inactive). This binary activation sequence serves as the supervision signal for training the model to detect when the queried event becomes relevant in the video stream. 

table 6 summarizes the statistics of the curated training datasets. Training samples are constructed diferently depending on whether a query corresponds to a single event or multiple events in the video. For datasets such as dense video captioning Caba Heilbron et al. (2015); Liu et al. (2024b), temporal activity detection Sigurdsson et al. (2016, 2018), grounded video QA Wang et al. (2024a), and moment localization Anne Hendricks et al. (2017), each caption or query typically describes a single event. In these cases, the query is paired with the corresponding video segment and the activation span is defined using the annotated start and end timestamps of the event. For datasets involving multiple actions or procedural steps Sigurdsson et al. (2016, 2018); Zhou et al. (2018); Liu et al. (2024b), a single query may correspond to multiple events in the video. For action recognition tasks, the action label itself is used as the query, while for procedural datasets we use the original question or caption as the query and construct activation spans for each relevant event. To prevent repeated activation for events that have already occurred, we sample a random time point between the end of the previous event and the start of the current event and set all activation positions before that point to 0. This encourages the model to ignore previously completed events and focus on the target span. 


Table 6 Training dataset statistics. Videos: number of source videos; Items: total annotation entries; Single: single-event entries; Multi: multi-event entries (average event count in parentheses); Segs: total training segments.


<table><tr><td>Dataset</td><td>Videos</td><td>Items</td><td>Single</td><td>Multi</td><td>Segs</td></tr><tr><td>ActivityNet-Captions Caba Heilbron et al. (2015)</td><td>10,009</td><td>37,421</td><td>37,421</td><td>-</td><td>37,421</td></tr><tr><td>LITA Huang et al. (2024b)</td><td>10,000</td><td>32,489</td><td>32,489</td><td>-</td><td>32,489</td></tr><tr><td>YouCook2 Zhou et al. (2018)</td><td>1,333</td><td>11,480</td><td>10,337</td><td>1,143 (×7.7)</td><td>19,174</td></tr><tr><td>ET-Instruct Liu et al. (2024b)</td><td>91,121</td><td>136,072</td><td>71,966</td><td>64,106 (×5.1)</td><td>398,609</td></tr><tr><td>Charades Sigurdsson et al. (2016)</td><td>7,811</td><td>48,684</td><td>48,067</td><td>617 (×2.1)</td><td>49,374</td></tr><tr><td>CharadesEgo Sigurdsson et al. (2018)</td><td>6,158</td><td>61,575</td><td>57,828</td><td>3,747 (×2.0)</td><td>65,488</td></tr><tr><td>DiDeMo Anne Hendricks et al. (2017)</td><td>8,208</td><td>22,911</td><td>22,911</td><td>-</td><td>22,911</td></tr><tr><td>Grounded-VideoLLM Wang et al. (2024a)</td><td>17,096</td><td>61,812</td><td>61,812</td><td>-</td><td>61,812</td></tr><tr><td>Total</td><td>151,736</td><td>412,444</td><td>342,831</td><td>69,613</td><td>687,278</td></tr></table>

Overall, the training set contains 141.7K videos with 379.9K annotations, including 310.3K single-event and 69.6K multi-event samples, resulting in 654.7K training segments. For multi-event samples, the average number of events per sample is reported in parentheses 

## A.2 Training Hyperparameters

To ensure reproducibility, we report the full set of hyperparameters used for training STRIDE (Qwen3-VL 2B and 4B) in Table 7. We process the input video stream at 1 FPS, accommodating up to 256 frames with a maximum spatial resolution of 512×512. Correspondingly, the temporal activation window size (W ) is set to 256. The model is trained using the AdamW optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ with a global batch size of 256 and no weight decay. We apply diferential learning rates: $3 \times 1 0 ^ { - 5 }$ for the language head and $1 \times 1 0 ^ { - 5 }$ for the language backbone, with a linear warmup of 512 steps followed by cosine decay. We use a gradient clipping threshold of 1.0, bfloat16 precision, and DeepSpeed ZeRO-2. 


Table 7 Detailed training hyperparameters for STRIDE.


<table><tr><td>Config</td><td>Value</td></tr><tr><td>Input frames</td><td>1 FPS</td></tr><tr><td>LR scheduler</td><td>Linear warm-up with cosine decay</td></tr><tr><td>Warmup steps</td><td>512</td></tr><tr><td>Optimizer</td><td>AdamW (<eq>\beta_1=0.9, \beta_2=0.999</eq>)</td></tr><tr><td>Global Batch size</td><td>256</td></tr><tr><td>Learning rate (lang. head)</td><td><eq>3 \times 10^{-5}</eq></td></tr><tr><td>Learning rate (lang. backbone)</td><td><eq>1 \times 10^{-5}</eq></td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Training precision</td><td>bfloat16</td></tr><tr><td>DeepSpeed</td><td>ZeRO-2</td></tr><tr><td>Input Resolution</td><td>Upto <eq>512 \times 512</eq></td></tr><tr><td>Act. Window size (W)</td><td>256</td></tr></table>

For each training sample, the visual context window is constructed with a length uniformly sampled between max(L, 8) and min(L, 256) seconds, where L denotes the source video length. This window is then randomly positioned along the video timeline, ensuring that the target event may or may not fall within the observable window. To allow the model to attend only to the current event of interest while disregarding previously completed events within the same window, the fixed inactive positions from multi-event samples (section A.1) are overridden onto the masked activation sequence after masking corruption. This ensures that these positions that these positions remain as 0 regardless of the applied mask. When the event of interest lies entirely outside the context window, the entire activation se quence is set to 0, training the model not to trigger. Conversely, all activation positions that temporally overlap with the target event are set to 1, enabling the downstream model to be invoked at the appropriate time. 

## B Detailed Architecture and Inference of STRIDE

## B.1 Masked Diffusion Formulation for Activation Span Modeling

In STRIDE, activation span prediction is formulated as a masked difusion process over a discrete activation sequence defined along the video timeline. Given a video and a query describing an event of interest, the supervision signal is represented as a binary activation sequence ${ \mathsf { a } } _ { 0 } = ( a _ { 0 } ^ { 1 } , \ldots , a _ { 0 } ^ { W } )$ ) of length $W$ , where $a _ { 0 } ^ { i } \in \{ 0 , 1 \}$ indicates whether the queried event is active at position i. Positions inside the annotated event span are labeled as 1 (active), while all other positions are labeled as 0 (inactive). 

## B.1.1 Forward Corruption Process.

To enable iterative refinement during training, we apply a masked difusion corruption process to the activation sequence. Starting from the ground-truth sequence ${ \tt a } _ { \mathrm { 0 } }$ , the forward process progressively masks tokens using a special symbol [M]. At noise level $t \in [ 0 , 1 ]$ , tokens are replaced by [M] with probability $t ,$ producing a partially corrupted sequence $\mathbf { a } _ { t } .$ . The corruption process factorizes across positions: 

$$
q (\mathbf {a} _ {t} \mid \mathbf {a} _ {0}) = \prod_ {i = 1} ^ {W} q (a _ {t} ^ {i} \mid a _ {0} ^ {i}), \quad q (a _ {t} ^ {i} \mid a _ {0} ^ {i}) = \left\{ \begin{array}{l l} 1 - t & \text { if } a _ {t} ^ {i} = a _ {0} ^ {i}, \\ t & \text { if } a _ {t} ^ {i} = [ \mathsf {M} ]. \end{array} \right.\tag{3}
$$

Here, it is important to note that STRIDE does not apply token-wise independent masking in practice. Instead, we use the proposed structured masking strategy (section 3.2.1) during training, where masking patterns are constructed to preserve contiguous temporal context along the video timeline while selectively hiding portions of the activation sequence. This encourages the model to infer coherent activation spans rather than predicting isolated token states. 

## B.1.2 Reverse Denoising Process.

The forward corruption process admits a reverse denoising process that reconstructs the clean activation sequence from a corrupted sequence. Given a partially masked activation sequence $\mathbf { a } _ { t } ,$ the model predicts the original activa tion token at each masked position conditioned on the observed context. During the reverse process, tokens that have already been revealed remain unchanged, while masked positions are progressively unmasked according to the model’s predictions or kept masked for further refinement. Through this iterative denoising process, the model gradually recovers the activation sequence and refines activation predictions across the entire video timeline. 

## B.2 Training Objective for Activation Diffusion

The model is trained by minimizing a cross-entropy loss over masked activation positions. Following the masked difusion formulation, the objective can be written as: 

$$
\mathcal {L} (\theta) = - \mathbb {E} _ {t \sim U [ 0, 1 ], \mathbf {a} _ {t} \sim q (\mathbf {a} _ {t} | \mathbf {a} _ {0})} \left[ \frac {1}{t} \sum_ {i = 1} ^ {W} \mathbb {1} [ a _ {t} ^ {i} = [ \mathsf {M} ] ] \log p _ {\theta} (a _ {0} ^ {i} | \mathbf {a} _ {t}) \right],\tag{4}
$$

where ${ \sf a } _ { \mathrm { 0 } }$ denotes the ground-truth activation sequence and ${ \sf a } _ { t }$ is the corrupted sequence obtained through the forward masking process. The indicator function $\mathbb { 1 } [ \cdot ]$ restricts the loss to masked positions only, allowing the model to leverage all unmasked tokens as context when predicting activation states. The $1 / t$ weighting normalizes the expected number of masked positions, ensuring that the loss contribution remains balanced across diferent noise levels t. 

## B.3 Inference Procedure for Activation Span Prediction

During inference, STRIDE predicts activation spans through an iterative reverse denoising process over the activation sequence. The process starts from a fully masked activation sequence ${ \bf a } _ { 1 } = ( [ { \sf M } ] , \dots , [ { \sf M } ] )$ of length W . Reverse denoising is then performed through K discrete refinement steps following a noise schedule $1 = t _ { K } > t _ { K - 1 } >$ $\cdots > t _ { 0 } .$ At each step from $t _ { k }$ to $t _ { k - 1 }$ , the model predicts activation tokens for all currently masked positions conditioned on the video and query representations. A fraction $( t _ { k } - t _ { k - 1 } ) / t _ { k }$ of masked positions with the highest confidence scores are revealed, while the remaining positions stay masked for further refinement. Positions that have already been revealed remain unchanged during the remaining denoising steps. Through this iterative predict-andrefine procedure, it progressively reconstructs the activation sequence and produces temporally consistent activation spans across the video timeline. 

To support streaming inference, STRIDE maintains the activation sequence as a sliding window over the most recent W positions of the timeline. When a new frame at time T +1 arrives, the window shifts forward: the oldest position is removed from the window, previously resolved activations are carried forward to their shifted positions, and a new slot corresponding to the incoming frame is appended. To verify whether previously inferred activations remain valid under the updated visual context $\gamma _ { \leq T + 1 }$ , STRIDE performs selective re-masking using a confidence threshold τ . If the confidence of a carried-forward decision exceeds τ , the activation is retained; otherwise, the position is remasked as [M] so that it re-enters the denoising process. The resulting masked set consists of both newly appended positions and low-confidence carried-forward positions, which are then refined through the same K-step denoising procedure described above. After the window is fully resolved, a trigger is issued only when an active span occupies at least a fraction γ of the activation window, where $\gamma$ denotes the span ratio. 

Since the activation model processes the video stream at 1 FPS, the visual context grows linearly with elapsed time. To prevent excessive computational overhead from unbounded context accumulation, we set the maximum context window size to 256 frames. When the accumulated context exceeds this limit, we retain only the most recent 128 frames and rebuild the context window from that point onward, efectively halving the temporal scope while preserving the latest visual evidence. Although cases where no trigger occurs for more than 256 consecutive seconds are rare in our evaluation benchmarks, this sliding window mechanism ensures that STRIDE remains deployable in arbitrarily long streams without memory overflow. 

## C Detailed Benchmark and Evaluation Setup

## C.1 Additional Benchmark Explanation

## C.1.1 OVO-Bench.

OVO-Bench Niu et al. (2025) evaluates temporal awareness in online video understanding by posing questions at specific timestamps during a video stream, rather than after the entire video has been observed. This timestamp conditioned protocol reflects a core challenge of streaming scenarios: the model must reason under partial observability, where future frames are not yet available at query time. The benchmark comprises 644 videos with 2,814 human-curated question-answer pairs across 12 tasks. 

The tasks are organized into three scenarios that capture distinct temporal reasoning patterns. Backward Tracing requires the model to recall and reason about past events, covering tasks such as EPM (Episodic Memory), ASI (Action Sequence Identification), and HLD (Hallucination Detection). Real-Time Visual Perception tests understanding of what is currently happening at the query timestamp, with six tasks spanning spatial understanding (STU), object recognition (OJR), attribute recognition (ATR), action recognition (ACR), optical character recognition (OCR), and future prediction (FPD). Forward Active Responding is the most distinctive scenario: the model receives a question whose answer depends on events that have not yet occurred, and must actively decide to wait rather than respond prematurely. This includes REC (Repetition Event Count), SSR (Sequential Steps Recognition), and CRR (Clues Reveal Responding). Backward Tracing and Real-Time Visual Perception tasks adopt a multiple-choice format with accuracy as the evaluation metric. The Forward Active Responding scenario employs both accuracy-based and score-based evaluation metrics through a multiple-triggering evaluation pipeline that densely queries models along the temporal axis. This scenario is directly relevant to proactive streaming, as it requires the model to judge when suficient evidence has been gathered, a capability closely aligned with activation timing. 

## C.1.2 StreamingBench.

StreamingBench Lin et al. (2024b) is designed to evaluate streaming comprehension by presenting questions at diferent temporal positions within a video, simulating how a user might interact with a model during real-time playback. The benchmark contains 900 videos with 4,500 human-curated QA pairs (five per video), evaluated across 18 tasks under three dimensions. Real-time Visual Understanding (10 tasks) covers a broad range of perceptual abilities including object perception (OP), causal reasoning (CR), clips summarization (CS), attribute perception (ATP), event understanding (EU), text-rich understanding (TR), prospective reasoning (PR), spatial understanding (SU), action per ception (ACP), and counting (CT). These tasks collectively test whether the model can track and interpret visual changes as the stream progresses. Omni-Source Understanding (4 tasks) requires integrating audio and visual sig nals, with tasks on emotion recognition (ER), scene understanding (SCU), source discrimination (SD), and multimodal alignment (MA). Contextual Understanding (4 tasks) evaluates higher-level reasoning over accumulated context, in cluding misleading context understanding (MCU), anomaly context understanding (ACU), sequential question answering (SQA), and proactive output (PO). The PO task is notable in that the model must determine the appropriate moment to respond without receiving an explicit user query, directly testing proactive timing capabilities. A response is considered correct only if the diference between the actual output timestamp and the ground-truth timestamp is less than two seconds. All tasks follow a multiple-choice format with accuracy as the primary metric. Each question is evaluated on the video segment from the beginning to the timestamp when the question is asked, approximating streaming conditions. 

## C.1.3 ET-Bench.

ET-Bench Liu et al. (2024b) is a large-scale benchmark for event-level video understanding that emphasizes finegrained temporal localization over multi-event videos. The full benchmark spans 7,300 samples across 12 tasks with 7,000 videos (251.4 hours) covering 8 domains. In our work, we evaluate on a subset of five tasks that directly measure temporal boundary prediction quality, which serves as a proxy for activation timing accuracy independent of the endto-end streaming pipeline. 

The five tasks we adopt are as follows. Temporal Video Grounding (TVG) requires localizing the temporal segment that matches a given text description within a video. Episodic Memory (EPM) extends this to egocentric scenarios, where the model must locate the moment relevant to a natural-language question (e.g., “Where did I put my keys?”). Temporal Action Localization (TAL) involves detecting all segments containing a specified action category, testing the model’s ability to identify repeated events with accurate boundaries. Dense Video Captioning (DVC) requires jointly segmenting a video into events and generating a caption for each, evaluating both localization and description quality. Step Localization and Captioning (SLC) is similar but targets instructional videos, where the model must identify and describe sequential procedural steps. 

These tasks span grounding and dense captioning capabilities under the ET-Bench taxonomy, sharing the common requirement of precise event boundary detection. To evaluate temporal localization performance, we report the F1 score as the evaluation metric for all five tasks. This metric directly measures how accurately the predicted event boundaries align with the ground-truth segments, allowing us to assess the quality of activation timing produced by the model. 

## C.2 Comparison with Autoregressive Baseline

To provide a fair comparison with autoregressive activation modeling, we reproduce an autoregressive baseline (Baseline-AR) following the design described in StreamBridge Wang et al. (2025a). Since the original training code and model parameters are not publicly available, we implement the baseline ourselves and train it under the same experimental setting as STRIDE. In particular, the baseline uses the same backbone, training data, and input configuration, ensuring that the comparison primarily reflects the diference between autoregressive point-wise triggering and the proposed span-level denoising formulation. 

## C.2.1 Architecture.

We adopt Qwen3-VL 2B as the backbone, processing up to 256 frames at 1 FPS to match the input configuration used in STRIDE. Following StreamBridge Wang et al. (2025a), a learnable <ACT> token is appended after the visual embedding of each frame. The token is passed through a lightweight score head that performs binary classification to predict whether a trigger should be issued at the corresponding time step. 

## C.2.2 Training & Inference.

The training data is constructed from the same annotations used for training STRIDE. For each annotated event segment, the last P % of frames within the segment are labeled as active (trigger-positive), while all remaining frames are labeled as inactive. To expose the model to diverse activation patterns, P is randomly sampled from a uniform distribution over [0, 50] for each training sample, following the training protocol of StreamBridge Wang et al. (2025a). 

At inference time, the score head outputs a trigger probability for each frame independently. A fixed threshold of 0.35 is applied to determine whether a trigger should be issued at each time step, following StreamBridge Wang et al. (2025a). This point-wise decision mechanism contrasts with STRIDE, which jointly denoises the activation sequence to produce span-level activation predictions. 

## D Additional Experimental Analysis

## D.1 Sensitivity Analysis for τ

In a streaming setting, the activation window slides as new visual context arrives. To re-evaluate previously resolved decisions against new visual evidence, we apply a confidence-based retention threshold τ (see Selective Re-masking in figure 2). Here, τ = 0 unconditionally retains all prior decisions, while τ = 1 efectively rebuilds the activation window from scratch at every step. To validate the efect of the Selective Re-masking threshold τ (section 3), we evaluate performance on ET-Bench by sweeping τ from 0 to 1. 

![](images/d174435a3986fc08ece704e8d50138d1e8738ce091c85bfbd3dcade05c5536d2.jpg)


Figure 5 Sensitivity of STRIDE to the retention constant τ across five temporal understanding tasks (TVG, EPM, TAL, DVC, SLC) in ET-Bench Liu et al. (2024b). The y-axis shows the score diference relative to each task’s average. Task-wise average scores are shown in the legend

As shown in figure 5, unconditional inheritance (τ = 0) results in the lowest scores, with a particularly large drop of −19.7 pt on TVG. Performance peaks broadly in the range $\tau \in [ 0 . 7 5 , 0 . 8 5 ]$ across most tasks, after which tightening the retention criterion causes a gradual decline. Based on these observations, τ = 0.75 is used in all evaluations. 

## D.2 Scalability Analysis for Activation Backbone

To examine how STRIDE scales with diferent size activation backbones, we conduct an additional experiment by replacing the default Qwen3-VL 2B activation backbone with a larger size Qwen3-VL 4B model. The 4B variant is trained nder the same data and training configuration as the 2B model and evaluated across three downstream 

Video-LLMs Team et al. (2025); Wang et al. (2025b); Bai et al. (2025a) on OVO-Bench (table 8) and StreamingBench (table 9). 

Across all downstream backbones, STRIDE-4B consistently achieves higher overall scores than STRIDE-2B, confirming that the activation backbone benefits from increased model capacity and that the improvement transfers regardless of the downstream Video-LLM, supporting the scalability of the proposed plug-in design. 


Table 8 Scalability analysis on activation backbone scale (STRIDE-2B vs. 4B) on OVO-Bench Niu et al. (2025) across multiple downstream Video-LLMs.


<table><tr><td rowspan="2">Method</td><td colspan="7">Real-Time Visual Perception</td><td colspan="4">Backward Tracing</td><td colspan="4">Fwd. Act. Responding</td><td>Overall</td></tr><tr><td>OCR</td><td>ACR</td><td>ATR</td><td>STU</td><td>FPD</td><td>OJR</td><td>Avg.</td><td>EPM</td><td>ASI</td><td>HLD</td><td>Avg.</td><td>REC</td><td>SSR</td><td>CRR</td><td>Avg.</td><td>Avg.</td></tr><tr><td>Gemma3-4B Team et al. (2025)</td><td>65.8</td><td>48.6</td><td>56.0</td><td>36.0</td><td>66.3</td><td>50.0</td><td>53.78</td><td>44.4</td><td>41.9</td><td>3.2</td><td>29.83</td><td>14.4</td><td>61.4</td><td>52.5</td><td>42.77</td><td>42.13</td></tr><tr><td>+ STRIDE-2B</td><td>73.2</td><td>60.6</td><td>64.7</td><td>39.3</td><td>71.3</td><td>56.5</td><td>60.93</td><td>47.8</td><td>52.0</td><td>4.8</td><td>34.87</td><td>42.6</td><td>64.6</td><td>60.0</td><td>55.73</td><td>50.51</td></tr><tr><td>+ STRIDE-4B</td><td>75.2</td><td>56.9</td><td>67.2</td><td>40.4</td><td>67.3</td><td>56.5</td><td>60.58</td><td>51.5</td><td>48.0</td><td>4.3</td><td>34.60</td><td>46.5</td><td>66.2</td><td>60.0</td><td>57.57</td><td>50.92</td></tr><tr><td>InternVL3-8B Wang et al. (2025b)</td><td>65.8</td><td>52.3</td><td>68.1</td><td>51.1</td><td>71.3</td><td>62.0</td><td>61.77</td><td>58.9</td><td>66.9</td><td>9.7</td><td>45.17</td><td>36.6</td><td>64.1</td><td>43.3</td><td>48.00</td><td>51.64</td></tr><tr><td>+ STRIDE-2B</td><td>75.8</td><td>54.1</td><td>80.2</td><td>56.7</td><td>74.3</td><td>65.2</td><td>67.72</td><td>58.9</td><td>65.5</td><td>11.3</td><td>45.23</td><td>40.1</td><td>67.7</td><td>66.2</td><td>58.00</td><td>56.98</td></tr><tr><td>+ STRIDE-4B</td><td>78.5</td><td>56.9</td><td>75.0</td><td>54.5</td><td>71.3</td><td>63.6</td><td>66.63</td><td>59.6</td><td>67.6</td><td>16.1</td><td>47.77</td><td>44.4</td><td>68.4</td><td>59.2</td><td>57.33</td><td>57.24</td></tr><tr><td>Qwen3-VL-8B Bai et al. (2025a)</td><td>69.8</td><td>59.6</td><td>73.3</td><td>57.3</td><td>71.3</td><td>58.7</td><td>65.00</td><td>55.6</td><td>63.5</td><td>12.9</td><td>44.00</td><td>37.7</td><td>60.8</td><td>40.4</td><td>46.30</td><td>51.77</td></tr><tr><td>+ STRIDE-2B</td><td>76.5</td><td>64.2</td><td>79.3</td><td>61.2</td><td>73.3</td><td>63.6</td><td>69.68</td><td>57.2</td><td>72.3</td><td>14.0</td><td>47.83</td><td>46.4</td><td>63.1</td><td>69.6</td><td>59.70</td><td>59.07</td></tr><tr><td>+ STRIDE-4B</td><td>75.2</td><td>64.2</td><td>76.7</td><td>62.9</td><td>73.3</td><td>66.3</td><td>69.77</td><td>59.3</td><td>74.3</td><td>12.4</td><td>48.67</td><td>57.1</td><td>64.6</td><td>65.4</td><td>62.37</td><td>60.27</td></tr></table>


Table 9 Scalability analysis on activation backbone scale (STRIDE-2B vs. 4B) on StreamingBench Lin et al. (2024b) across multiple downstream Video-LLMs.


<table><tr><td rowspan="2">Method</td><td colspan="11">Real-Time Visual Understanding</td><td colspan="5">Omni-Source Understanding</td><td colspan="5">Contextual Understanding</td><td>Overall</td></tr><tr><td>OP</td><td>CR</td><td>CS</td><td>ATP</td><td>EU</td><td>TR</td><td>PR</td><td>SU</td><td>ACP</td><td>CT</td><td>Avg.</td><td>ER</td><td>SCU</td><td>SD</td><td>MA</td><td>Avg.</td><td>ACU</td><td>MCU</td><td>SQA</td><td>PO</td><td>Avg.</td><td>Avg.</td></tr><tr><td>Gemma3-4B Team et al. (2025)</td><td>63.8</td><td>69.5</td><td>68.8</td><td>54.4</td><td>60.2</td><td>65.1</td><td>59.3</td><td>40.7</td><td>62.7</td><td>21.8</td><td>57.49</td><td>28.8</td><td>31.2</td><td>30.4</td><td>46.4</td><td>34.20</td><td>34.4</td><td>31.2</td><td>38.8</td><td>12.4</td><td>29.20</td><td>40.30</td></tr><tr><td>+ STRIDE-2B</td><td>66.8</td><td>71.9</td><td>66.6</td><td>57.2</td><td>66.5</td><td>70.7</td><td>60.2</td><td>43.1</td><td>65.0</td><td>23.8</td><td>60.00</td><td>35.6</td><td>31.6</td><td>36.0</td><td>44.0</td><td>36.80</td><td>33.6</td><td>35.2</td><td>44.8</td><td>41.6</td><td>38.80</td><td>45.20</td></tr><tr><td>+ STRIDE-4B</td><td>66.2</td><td>63.3</td><td>68.1</td><td>58.1</td><td>67.1</td><td>71.7</td><td>63.0</td><td>41.5</td><td>66.7</td><td>21.2</td><td>59.93</td><td>32.8</td><td>30.4</td><td>36.0</td><td>46.4</td><td>36.40</td><td>35.2</td><td>36.8</td><td>48.0</td><td>44.0</td><td>41.00</td><td>45.78</td></tr><tr><td>InternVL3-8B Wang et al. (2025b)</td><td>74.9</td><td>82.0</td><td>75.7</td><td>61.2</td><td>72.0</td><td>67.6</td><td>74.1</td><td>66.7</td><td>78.1</td><td>34.2</td><td>68.71</td><td>40.4</td><td>27.6</td><td>38.8</td><td>45.6</td><td>38.10</td><td>38.0</td><td>26.0</td><td>36.8</td><td>31.2</td><td>33.00</td><td>46.60</td></tr><tr><td>+ STRIDE-2B</td><td>74.9</td><td>78.9</td><td>76.7</td><td>68.6</td><td>77.0</td><td>77.3</td><td>77.8</td><td>71.5</td><td>83.0</td><td>33.2</td><td>72.45</td><td>39.6</td><td>22.4</td><td>44.0</td><td>50.8</td><td>39.20</td><td>34.0</td><td>35.2</td><td>43.2</td><td>42.8</td><td>38.80</td><td>50.15</td></tr><tr><td>+ STRIDE-4B</td><td>77.7</td><td>81.2</td><td>79.5</td><td>70.0</td><td>75.2</td><td>77.3</td><td>73.1</td><td>72.4</td><td>84.3</td><td>37.8</td><td>73.82</td><td>38.8</td><td>25.6</td><td>43.6</td><td>55.6</td><td>40.90</td><td>39.2</td><td>35.2</td><td>44.4</td><td>44.8</td><td>40.90</td><td>51.87</td></tr><tr><td>Qwen3-VL-8B Bai et al. (2025a)</td><td>62.7</td><td>68.0</td><td>69.7</td><td>53.3</td><td>67.5</td><td>65.1</td><td>67.6</td><td>48.0</td><td>68.0</td><td>40.9</td><td>60.88</td><td>36.8</td><td>18.0</td><td>32.8</td><td>34.0</td><td>30.40</td><td>25.6</td><td>23.6</td><td>31.2</td><td>32.4</td><td>28.20</td><td>39.83</td></tr><tr><td>+ STRIDE-2B</td><td>77.1</td><td>75.0</td><td>77.3</td><td>72.8</td><td>76.4</td><td>77.9</td><td>76.9</td><td>69.9</td><td>84.3</td><td>46.1</td><td>74.24</td><td>42.8</td><td>24.0</td><td>45.2</td><td>53.2</td><td>41.30</td><td>32.0</td><td>38.4</td><td>46.4</td><td>42.8</td><td>39.90</td><td>51.81</td></tr><tr><td>+ STRIDE-4B</td><td>80.7</td><td>78.9</td><td>79.8</td><td>73.1</td><td>78.9</td><td>84.1</td><td>77.8</td><td>69.9</td><td>84.0</td><td>42.5</td><td>76.01</td><td>42.8</td><td>21.2</td><td>41.6</td><td>54.4</td><td>40.00</td><td>36.0</td><td>36.8</td><td>40.8</td><td>46.0</td><td>39.90</td><td>51.97</td></tr></table>

## D.3 Qualitative Examples of STRIDE

We provide qualitative examples of activation span prediction on OVO-Bench, StreamingBench, and ET-Bench in figures 6 to 17. Each example visualizes the temporal timeline of the video together with the query arrival time, the ground-truth event span, and the activation span predicted by STRIDE. These timelines illustrate how the model progressively identifies the relevant event segment and aligns its activation predictions with the ground-truth temporal boundaries under streaming conditions. 

## E Limitation and Discussion

## E.1 Failure Cases and Discussion

Although STRIDE improves temporal stability of activation decisions, several practical limitations remain in streaming deployments. First, the activation model operates on sparsely sampled frames (1 FPS) and relies on downstream Video LLMs whose streaming interfaces typically process visual tokens at relatively low frame rates. As a result, extremely short-lived events or rapid visual transitions may not be fully captured by the activation window, since the visual evidence may disappear before suficient temporal context is accumulated. figure 18 illustrates such a case, where a brief visual event occurs between sampled frames and therefore cannot be reliably localized by the activation model. 

Another challenging scenario arises when queries refer to broad or loosely defined events rather than a single well localized moment. In such cases, multiple candidate segments may partially satisfy the query semantics, leading to dispersed or multi-span activations. figure 19 presents an example where the model encounters several visually plausible moments corresponding to the query, which may introduce ambiguity in determining the most appropriate triggering point. These observations suggest that proactive activation remains sensitive to both temporal sampling granularity and query specificity, highlighting directions for future improvements in streaming perception and query grounding. 

Question: Find event: "a shark is swimming underwater" 

![](images/7caf47ff8118673823767c180806a4c7c45089faef4499d001ce688c3dba1bf0.jpg)


![](images/584843e6713d6307ab37ca820d9c3a972e8e7abc42cb73d495bb016bae896095.jpg)



Figure 6 Qualitative example from ET-Bench (TVG).


Question: Find event: "a family is being recorded while having dinner" 

![](images/e0148496ac258ecabab3ca7c5bf53641af9bd9a1dccaf9065044e6aae7e264c1.jpg)



Figure 7 Qualitative example from ET-Bench (TVG).


![](images/0903bc57a611adcf30e09d94337e5d160df039ee4383c2aac3de62940613827e.jpg)


![](images/52b1dde828738b6eedda722426ee2f7e7c7bfcb2c2097add0c13e0d2f454a246.jpg)



Figure 8 Qualitative example from ET-Bench (EPM).


![](images/932292206d093bd16910f2bd415b6ee6150297344022032c10684f3f46650605.jpg)



Figure 9 Qualitative example from ET-Bench (TAL).


Question: Dense captioning: "making tomato soup" 

![](images/39e8619d8cbc1fc1b421be07632e6c2c9139421c8d661550e66298c7414fd5fe.jpg)



Figure 10 Qualitative example from ET-Bench (DVC).


Question: Step localization: "make a latte" 

![](images/9cc3396822138c16d895077c13b937525aa76b83e70835779414cee1dc20fd5f.jpg)



Figure 11 Qualitative example from ET-Bench (SLC).


## Question:

what color was the phone? [0:38] 

Options: 

• A. blue 

• B. white 

• C. black 

• D. red 

![](images/6a573eba2611dbc23faf8c783c393a83f4aed6d52a211e6ca836f317c05eb670.jpg)


![](images/16871b7614902ab6691af3fc39f1a1ea79810f1c51448f16bdaf358d57df6357.jpg)



Figure 12 Qualitative example from OVO-Bench (EPM).


## Question:

What is the state of the person’s hand shown? [10:40] 

## Options:

• A. The person’s hand is shown with a bandage. 

• B. The person’s hand is shown with a tattoo. 

• C. The person’s hand is shown with a glove. 

• D. The person’s hand is shown with cuts and dirt. 

![](images/ee1c79652bfdb14454f73c4d2a3521e6b843cf8648e8236819db646f6b31eea6.jpg)



Figure 13 Qualitative example from OVO-Bench (OJR).


## Question:

What are these two people holding? [15:22] 

Options: 

• A. Black Camera 

• B. Silver round medal 

• C. Football 

• D. Platinum round medal 

![](images/7e64e09b1c6c9fef0897488fe1c7df280fe12649c1a58102b511d7ee8af482ef.jpg)



Figure 14 Qualitative example from OVO-Bench (OJR).


## Question:

What material are the stairs made of? [8:03] 

## Options:

• A. Metal. 

• B. Wood. 

• C. Marble. 

• D. Concrete. 

![](images/8af786c17b5359632a8c6be8865e58c13750097ba0e8e437ca2eb95092a1ae95.jpg)


![](images/8db8e862581d587e53c8e340644c49de327dab79fc5e628c8ec9d0be756223de.jpg)



Figure 15 Qualitative example from StreamingBench (Attribute Recognition).


## Question:

What was the name of the café shown just now? [3:04] 

## Options:

• A. The Green Café. 

• B. The Fun Café. 

• C. Power Up Café. 

• D. Stage 52 Café. 

![](images/f1bedb4e7c6e513f58603e8e2e3fa7aec8c826b276a917c9dc3fa5e7a0a93a5f.jpg)


![](images/07a89c1cf9b4364d808b8354c66c79215696993672eb6b0285871ebbefac7dc5.jpg)



Figure 16 Qualitative example from StreamingBench (Object Recognition).


## Question:

What graphics card models are shown on the benchmarking results right now? [5:09] Options: 

• A. 4080 FE, 4090 Suprim X, 4090 Matrix. 

• B. 4090 FE, 4080 Suprim X, 4090 Matrix. 

• C. 4090 FE, 4090 Suprim X, 4080 Matrix. 

• D. 4090 FE, 4090 Suprim X, 4090 Matrix. 

![](images/b1ccbfb8ed9175bbaabbf6a350af1f0f0575750197ca42f70d130889698503d9.jpg)



Figure 17 Qualitative example from StreamingBench (Text-Rich Understanding).


![](images/8b4d855f3731d826f256636d91dbdab3922f19546d5045d3b7b040dbad08f4c0.jpg)


![](images/583de50213293c0c2e184340f609f66657bba248ba8f8d084e9e20851364d7f9.jpg)



Figure 18 Failure Case from StreamingBench (Proactive Output).


515s 

## Question:

Please describe the scene that just occurred in the video. [7:43] 

## Options:

• A. A panda wearing pants meditated in front of a cherry blossom tree and said, ’Hey there,’, I’m Po the Dragon Warrior. 

• B. A panda wearing pants closed its eyes and meditated in front of the cherry blossom tree, saying ’Breathe’ 

• C. A panda wearing pants closed its eyes and meditated in front of the cherry blossom tree, then took a deep breath and said ’Out through the mouth’ before exhaling 

• D. A panda wearing pants closed its eyes and meditated in front of an apple tree, then took a deep breath and said ’Out through the mouth’ before exhaling 

![](images/7f80128924447f5d69e9f9daa3ad63c2aff3f6a95f6b61ffacb27013fc57380b.jpg)



Figure 19 Failure Case from StreamingBench (Scene Understanding).
