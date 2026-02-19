# OpenVLA Paper Deep Dive
- Created: 2026-02-15
- Last updated: 2026-02-15
- Tags: OpenVLA, VLA, vision-language-action, Prismatic, Llama2, SigLIP, DINOv2, action-tokenization, LIBERO, Open-X-Embodiment

## Current state
- OpenVLA is a 7B-parameter open-source VLA built on the Prismatic-7B VLM (SigLIP+DINOv2 fused vision encoder + Llama 2 7B LLM).
- Input: single 224x224 image + natural language instruction. Output: 7-dim robot actions tokenized as 7 discrete integers in [0..255].
- Actions discretized per-dimension into 256 bins using 1st/99th quantile bounds; tokens mapped to 256 least-used slots in Llama tokenizer.
- Trained on 970k episodes from Open X-Embodiment for 27 epochs (64 A100s, 14 days, batch 2048, lr=2e-5 fixed).
- Loss: next-token prediction cross-entropy on action tokens only.
- Vision encoder is fine-tuned (not frozen) — critical for VLA performance.
- Outperforms RT-2-X (55B) by 16.5% absolute on 29 tasks despite 7x fewer parameters.
- LoRA fine-tuning (r=32, 1.4% params) matches full fine-tuning on downstream tasks.
- 4-bit quantization preserves performance while halving memory (7 GB VRAM).
- On LIBERO simulation benchmark: 76.5% avg success rate (best among tested methods) via LoRA fine-tuning.
- Key limitations: single-image only, ~6 Hz inference, no action chunking, <90% reliability.

## Questions asked
- Q1: What is the overall OpenVLA architecture?
- Q2: What are the input/output specifications?
- Q3: How does the vision encoder work?
- Q4: How are actions represented and predicted?
- Q5: What is the training procedure?
- Q6: What are the LIBERO benchmark evaluation details?

## Q1: Overall OpenVLA Architecture
### Answer (short)
Three-component VLM architecture: fused vision encoder (SigLIP + DINOv2, ~600M params) → 2-layer MLP projector → Llama 2 7B LLM backbone. Total ~7B parameters. Built on the Prismatic-7B VLM from Karamcheti et al. 2024.

### Details
- **Vision Encoder**: Two-part, consisting of pretrained SigLIP (semantic features) and DINOv2 (spatial features). Input image patches are passed separately through both encoders; resulting feature vectors concatenated channel-wise. Total ~600M parameters.
- **Projector**: Small 2-layer MLP that maps concatenated visual patch embeddings into the input space of the language model.
- **LLM Backbone**: Llama 2 7B-parameter language model.
- **Backbone origin**: Prismatic-7B VLM, which was itself fine-tuned on the LLaVA 1.5 data mixture (~1M image-text and text-only samples) on top of SigLIP, DINOv2, and Llama 2.
- The "patch-as-token" approach treats visual patch features as tokens projected into the LLM input space — same approach as LLaVA, PaLI-3, etc.
- Prismatic was chosen over LLaVA and IDEFICS-1 after ablations showed it improved absolute success rate by ~10% on both single-object and multi-object tasks, attributed to the fused SigLIP-DINOv2 spatial reasoning.

### Evidence
- Architecture figure described in Section 3 (Fig 2): "three key components: (1) a vision encoder that concatenates DINOv2 and SigLIP features, (2) a projector that maps visual features to the language embedding space, and (3) the LLM backbone, a Llama 2 7B-parameter large language model."
- Ablation: OpenVLA-Bridge (with DINOv2) outperforms OpenVLA-Bridge-SigLIP (without DINOv2) by ~5% absolute success rate.

---

## Q2: Input/Output Specification
### Answer (short)
Input: one 224×224 RGB image + one natural language instruction string. Output: N discrete action tokens (typically N=7 for 7-DoF end-effector control), each an integer in [0, 255].

### Details
- **Image input**: Single third-person camera image, resized to 224×224 pixels. Higher resolution (384×384) was tested but gave no performance improvement while taking 3x longer to train.
- **Language input**: Natural language task instruction (e.g., "put the carrot on the plate").
- **Action output**: 7-dimensional robot control actions (end-effector control). Each dimension discretized independently into one of 256 bins → 7 tokens autoregressively generated.
- **No proprioceptive input**: Unlike Diffusion Policy or Octo, OpenVLA uses only a single image — no proprioceptive state, no observation history.
- **No action chunking**: Predicts one action per inference step (unlike Diffusion Policy which uses receding horizon control).
- **Control frequency**: ~6 Hz on RTX 4090 in bfloat16 (without compilation or speculative decoding).

---

## Q3: Vision Encoder Details
### Answer (short)
Fused dual encoder: SigLIP (ViT-SO) for high-level semantic features + DINOv2 for low-level spatial features. Patches processed separately through each encoder, features concatenated channel-wise, then projected via 2-layer MLP into LLM token space. Resolution: 224×224. Encoder is fine-tuned (not frozen) during VLA training.

### Details
- **SigLIP**: Sigmoid loss variant of CLIP. Captures semantic understanding — object categories, language grounding.
- **DINOv2**: Self-supervised ViT. Captures spatial/geometric features — object positions, edges, fine-grained layout.
- **Fusion**: Channel-wise concatenation of patch features from both encoders. This creates a richer representation than either encoder alone.
- **Projection**: 2-layer MLP maps concatenated visual features into the Llama 2 input embedding space. Visual tokens are then interleaved with text tokens as input to the LLM.
- **Resolution choice**: 224×224 px chosen over 384×384 px — no performance difference found for robot control tasks, but 3x training compute reduction. (Note: higher resolution does help on standard VLM benchmarks, but not yet for VLAs.)
- **Fine-tuning vs. freezing**: Unlike standard VLM practice (freeze encoder), VLA training requires fine-tuning the vision encoder. Frozen encoder policies showed dramatically worse performance (e.g., 25% vs 90% for LLaVA backbone). Hypothesis: pretrained vision features don't capture sufficient fine-grained spatial detail for precise robotic control.
- **Ablation evidence**: Removing DINOv2 (SigLIP-only) drops success rate by ~5% on Bridge tasks. The low-level spatial features from DINOv2 aid generalization in some but not all cases.

---

## Q4: Action Representation and Prediction
### Answer (short)
Continuous robot actions are discretized per-dimension into 256 bins (using 1st/99th percentile bounds from training data). Each bin maps to one of 256 tokens that overwrite the least-used tokens in the Llama tokenizer. The model autoregressively predicts N action tokens (one per action dimension). Loss is cross-entropy on action tokens only.

### Details
- **Discretization scheme**: Following RT-2, each of N action dimensions is independently discretized into 256 bins. For a 7-DoF action, this yields 7 integers ∈ [0, 255].
- **Bin boundaries**: Uniform division of the interval [1st percentile, 99th percentile] of actions in training data. Using quantiles (not min-max like RT-2) avoids outlier actions expanding the interval and reducing effective granularity.
- **Token mapping**: The Llama tokenizer only has 100 reserved special tokens (too few for 256 action tokens). Solution: overwrite the 256 *least used* tokens in the existing vocabulary (the last 256 tokens) with action tokens. Same approach as RT-2.
- **Autoregressive generation**: The LLM backbone generates action tokens one at a time, left to right, for all N dimensions.
- **De-tokenization**: Map predicted token index back to the corresponding bin center value for each action dimension.
- **Loss function**: Standard next-token prediction cross-entropy, but evaluated only on the predicted action tokens (not on image or language tokens).
- **No-op filtering**: Critical data cleaning step — filtering out all-zero actions (especially the first timestep in BridgeData V2) to prevent the model from learning to predict no-op actions and freezing during deployment. This was a key advantage over RT-2-X which trained on unfiltered data.

---

## Q5: Training Procedure
### Answer (short)
Fine-tune Prismatic-7B VLM on 970k robot episodes from Open X-Embodiment. 64 A100 GPUs, 14 days, batch size 2048, fixed lr 2e-5, 27 epochs (~150k iterations). All parameters fine-tuned (including vision encoder). Cross-entropy loss on action tokens only.

### Details
- **Base model**: Prismatic-7B VLM, pre-trained on LLaVA 1.5 data mixture (~1M samples). This gives the model strong vision-language understanding before robot training.
- **Training data**: 970k episodes from Open X-Embodiment (curated subset):
  - Filtered to: manipulation datasets only, at least one 3rd-person camera, single-arm end-effector control.
  - Mixture weights follow Octo's heuristic: down-weight less diverse datasets, up-weight diverse ones.
  - 26+ individual datasets including Fractal (12.7%), Kuka (12.7%), Bridge (13.3%), BC-Z (7.5%), FMB (7.1%), Language Table (4.4%), etc.
  - DROID included at 10% weight initially but removed for last third of training due to slow learning.
- **Hyperparameters**:
  - 64 A100 GPUs, 14 days total (21,500 A100-hours)
  - Batch size: 2048
  - Learning rate: 2e-5 (fixed, same as VLM pretraining; no warmup)
  - 27 epochs through the dataset, ~150k iterations
  - All parameters fine-tuned (including vision encoder — crucial)
- **Key training insights**:
  - Many more epochs needed than typical LLM/VLM training (27 vs 1-2). Real robot performance keeps improving until action token accuracy >95%.
  - Fine-tuning vision encoder is crucial (opposite of VLM best practice).
  - Learning rate warmup provides no benefit.
  - 224×224 resolution sufficient (384×384 gives no improvement).
- **Infrastructure**: PyTorch with AMP, FlashAttention, FSDP. Released as open-source codebase.
- **Fine-tuning for new tasks**:
  - Full fine-tuning: 8 A100s, 5-15 hours per task.
  - LoRA (r=32): 1 A100, 10-15 hours. Matches full fine-tuning with only 1.4% of parameters.
  - Sandwich fine-tuning: vision encoder + embedding + last layer. Reasonable but worse than LoRA.
  - Freezing vision encoder during fine-tuning: bad performance.

---

## Q6: LIBERO Benchmark Evaluation
### Answer (short)
OpenVLA fine-tuned via LoRA (r=32) on LIBERO achieves 76.5% average success rate across 4 task suites (best among tested methods), beating Diffusion Policy from scratch (72.4%) and Octo fine-tuned (75.1%). Margins are tighter than real-world experiments due to sim-to-real domain gap.

### Details
- **Benchmark**: LIBERO (Liu et al. 2024) — four task suites for lifelong learning in robotic manipulation:
  - LIBERO-Spatial: same objects, different layouts → spatial reasoning (OpenVLA: 84.7%, best)
  - LIBERO-Object: same layouts, different objects → object understanding (OpenVLA: 88.4%, 2nd)
  - LIBERO-Goal: same objects+layouts, different goals → task behaviors (OpenVLA: 79.2%, 2nd)
  - LIBERO-Long: long-horizon diverse tasks (OpenVLA: 53.7%, best)
- **Setup**: 10 tasks per suite, 50 demos each. Each policy trained independently per suite.
- **Data modifications for OpenVLA**:
  1. Regenerated demos at 256×256 resolution (original was 128×128).
  2. Filtered no-op actions (near-zero magnitude).
  3. Rotated third-person images 180° (upside-down rendering issue).
  4. Filtered failed demonstrations (68-121 removed per suite).
  5. Used only third-person camera (no wrist camera) for fair comparison.
- **Evaluation**: 500 trials per suite, averaged over 3 random seeds (1500 total trials per statistic).
- **Results**: OpenVLA achieves highest average success rate (76.5%) and best average rank (1.5 across suites).
- **Key insight**: Margins between methods are tighter in simulation than in real-world experiments. Attributed to domain gap — OpenVLA was pretrained on purely real-world data with no simulation data in pretraining mixture.
- **Compared methods**: Diffusion Policy from scratch (72.4%, rank 2.5), Octo fine-tuned (75.1%, rank 2.0).

---

## Archive
(Empty — first entry)
