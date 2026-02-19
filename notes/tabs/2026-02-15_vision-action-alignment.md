# Vision-Action Alignment: From Platonic Representation Hypothesis to VLA

- Created: 2026-02-15
- Last updated: 2026-02-17 (round 5)
- Tags: platonic-representation-hypothesis, alignment, mutual-knn, openvla, vision-action, libero, VLA, kernel-alignment, embedding-extraction, action-tokenization, pipeline-diagram, search-prompts

## Current state

- The Platonic Representation Hypothesis (PRH) shows that vision and language models are converging to a shared statistical model of reality, measured via mutual k-nearest-neighbor (mutual_knn) alignment on paired image-caption datasets.
- The platonic-rep codebase implements this: extract per-layer features from vision (CLS token) and language (avg-pool) models on paired data (WIT 1024), remove outliers at 95th percentile, L2-normalize, compute mutual_knn(k=10) across all layer pairs, take the max.
- OpenVLA is a 7B VLA built on Prismatic-7B (fused SigLIP + DINOv2 vision encoder + Llama 2 backbone). It discretizes continuous 7-DoF actions into 256 bins and predicts them autoregressively.
- We want to adapt the PRH framework to measure vision-action alignment in VLA models running on Libero, replacing the "language embedding" with an "action embedding."
- **Q1 decision (vision embedding):** Option A (CLS token per ViT layer) is the primary approach; Option B (avg-pooled patch tokens) is secondary. Options C/D dropped.
- **Q2 decision (action embedding):** NOT YET DECIDED. A full pipeline diagram of how OpenVLA generates, decodes, and applies actions has been added. Five candidate extraction points are annotated. A pilot experiment (raw 7D actions vs shuffled baseline) is proposed to guide the decision.

## Questions asked

- Q1: How does the Platonic paper define "concept" and measure vision-language alignment?
- Q2: How is the alignment metric computed in the platonic-rep codebase?
- Q3: What is the OpenVLA model structure (architecture, I/O, training)?
- Q4: How can we measure vision-action alignment, imitating the Platonic Representation Hypothesis?
- Q5: Standalone idea description and search prompts for related work

---

## Q1: How does the Platonic paper define "concept" and measure vision-language alignment?

### Answer (short)

The paper does not define "concept" as a formal term. Instead it formalizes reality as a sequence of discrete **events** Z = [z_1, ..., z_T], each observable through bijective observation functions (e.g., images, text). A "concept" is implicitly an event z whose different observations (image x, caption y) are paired. Alignment is measured by how much the k-nearest-neighbor structure is preserved across modalities.

### Details

**Events and observations (Section 5.1):**

The world consists of discrete events $Z$ sampled from an unknown distribution $P(Z)$. Each event can be observed via a bijective, deterministic function $obs: Z \to X$. An image is $obs_{image}(z)$, a caption is $obs_{text}(z)$. Because the mapping is bijective over discrete variables, the statistical structure (specifically the PMI kernel) is preserved across all observation modalities.

**Formal definitions (Section 2):**

- A **representation** is a function f: X -> R^n assigning a feature vector to each input.
- A **kernel** K(x_i, x_j) = <f(x_i), f(x_j)> characterizes pairwise similarity.
- A **kernel-alignment metric** m: K x K -> R measures similarity between two kernels.

**Primary alignment metric: Mutual k-Nearest Neighbor ($m_{NN}$)**

Given two representations $f$ and $g$ producing features $\phi_i = f(x_i)$ and $\psi_i = g(y_i)$ on $N$ paired samples:

1. Compute k-NN index sets $S(\phi_i)$ and $S(\psi_i)$ within each embedding space.
2. For each sample $i$, compute overlap: $m_{NN}(\phi_i, \psi_i) = \frac{1}{k} |S(\phi_i) \cap S(\psi_i)|$.
3. Average over all samples.

Higher mutual_knn = higher alignment. This metric focuses on local neighborhood structure rather than global geometry (unlike CKA, which the authors found too noisy to reveal alignment trends).

**Why not CKA:**

CKA (Centered Kernel Alignment) is sensitive to all pairwise distances globally. The authors found CKA showed "a very weak trend" while mutual_knn captured alignment clearly. They also propose CKNNA (Centered Kernel Nearest-Neighbor Alignment) which restricts CKA to mutual nearest neighbors, bridging the two approaches.

**Practical details (Appendix C):**

- Vision models: extract CLS token from each transformer layer.
- Language models: average-pool hidden states across tokens for each layer.
- Layer selection: compute pairwise alignment across all (layer_i, layer_j) combinations; take the maximum score (inspired by BrainScore).
- L2-normalization applied to features before computing neighbors.
- Elements above 95th percentile of absolute value are clamped (to handle transformer outlier activations).
- Cross-modal alignment uses 1024 paired image-caption samples from WIT (Wikipedia Image Text).
- Default: k=10 for mutual_knn.

**The PMI kernel convergence result (Section 5.2):**

Contrastive learners with NCE objectives converge to the pointwise mutual information (PMI) kernel:

$$
\langle f_X(x_a), f_X(x_b) \rangle = K_{PMI}(x_a, x_b) + c_X
$$

where $K_{PMI}(x_a, x_b) = \log\left(\frac{P_{co}(x_a, x_b)}{P(x_a) \cdot P(x_b)}\right)$. Because observations are bijective over discrete events: $K_{PMI}(x_a, x_b) = K_{PMI}(z_a, z_b) = K_{PMI}(y_a, y_b)$. All modalities converge to the same kernel.

**Key empirical results:**

- Among 78 vision models, those solving more VTAB tasks are significantly more aligned with each other ("Anna Karenina effect").
- Cross-modal: linear relationship between LLM performance (1 - bits-per-byte) and vision-language alignment score.
- CLIP models (trained with language supervision) show highest alignment. Fine-tuning CLIP on ImageNet *reduces* alignment.
- LLM alignment with DINOv2 correlates with downstream reasoning tasks (Hellaswag, GSM8K).
- Denser captions produce higher alignment scores.
- Absolute alignment scores are low (~0.16 out of 1.0). The paper acknowledges this is an open question.
- The paper notes that robotics is a domain where convergence may not yet hold due to data scarcity.

### Evidence / references

- Paper: Section 2 (formal definitions), Section 5.1 (events and observations), Section 5.2 (PMI kernel), Appendix A (mutual_knn definition), Appendix C (practical details).
- Figures: Fig 2 (vision-vision convergence), Fig 3 (cross-modal convergence), Fig 5 (downstream prediction), Fig 6 (color experiment).

---

## Q2: How is the alignment metric computed in the platonic-rep codebase?

### Answer (short)

The pipeline: (1) load a paired image-caption dataset from HuggingFace, (2) extract per-layer features from vision models (CLS token via TIMM feature extraction) and language models (avg-pool of hidden states via HuggingFace), (3) for each pair of layers, remove outliers, L2-normalize, compute mutual_knn with k=10, (4) report the max score and which layer pair achieved it.

### Details

**1. Data loading (`data.py`, `platonic/alignment.py`):**

- Dataset: `minhuh/prh` on HuggingFace, subset `wit_1024` (1024 paired image-caption samples from WIT).
- Each sample has an `image` field (PIL Image) and a `text` field (list of captions; index 0 = main caption).
- The `Alignment` class downloads pre-computed features for reference models (e.g., `openllama_7b`, `dinov2_g`, `clip_h`) from URLs, stored as `.pt` files containing `{"feats": tensor}`.

**2. Embedding extraction (`extract_features.py`):**

Vision models (ViT via TIMM):
```
# For each ViT block, extract output at block.add_1 (residual connection output)
return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

# Extract CLS token (index 0) from each layer
feats = [v[:, 0, :] for v in lvm_output.values()]  # list of [batch, embed_dim]
feats = torch.stack(feats).permute(1, 0, 2)  # [batch, num_layers, embed_dim]
```

Language models (HuggingFace):
```
# Model returns hidden_states for all layers (including embedding layer)
llm_output = language_model(input_ids=..., attention_mask=...)

# Average-pool across sequence length, masked by attention_mask
feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)  # [batch, num_layers, seq_len, dim]
mask = attention_mask.unsqueeze(-1).unsqueeze(1)
feats = (feats * mask).sum(2) / mask.sum(2)  # [batch, num_layers, dim]
```

Saved as: `{"feats": [batch, num_layers, dim], "num_params": int, ...}`

**3. Alignment score calculation (`measure_alignment.py`, `metrics.py`):**

Step-by-step in `compute_score()`:

```python
def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True):
    # x_feats: [N, L_x, D_x], y_feats: [N, L_y, D_y]
    # Split into per-layer: x_feats[i] is [N, D_x] for layer i
    
    for i, x in enumerate(x_feats):  # iterate over layers of model X
        for j, y in enumerate(y_feats):  # iterate over layers of model Y
            x_aligned = F.normalize(x, p=2, dim=-1)  # L2 normalize
            y_aligned = F.normalize(y, p=2, dim=-1)
            score = AlignmentMetrics.measure("mutual_knn", x_aligned, y_aligned, topk=10)
            # track best (score, (i, j))
    return best_score, best_layer_pair
```

The `prepare_features()` function first removes outliers:
```python
def prepare_features(feats, q=0.95, exact=False):
    feats = remove_outliers(feats.float(), q=q, exact=exact)  # clamp at 95th percentile
    return feats.cuda()
```

`remove_outliers()` computes the q-th percentile of absolute values and clamps:
```python
q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
return feats.clamp(-q_val, q_val)
```

**4. Mutual k-NN metric (`metrics.py`):**

```python
def mutual_knn(feats_A, feats_B, topk):
    # feats_A, feats_B: [N, D] -- same N samples, different embedding spaces
    knn_A = compute_nearest_neighbors(feats_A, topk)  # [N, topk] index sets
    knn_B = compute_nearest_neighbors(feats_B, topk)
    
    # Binary masks: lvm_mask[i, j] = 1 iff j is in kNN of i in space A
    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0
    
    # Overlap: fraction of shared neighbors
    acc = (lvm_mask * llm_mask).sum(dim=1) / topk
    return acc.mean().item()

def compute_nearest_neighbors(feats, topk=1):
    # feats: [N, D]
    knn = (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    return knn  # [N, topk]
```

**5. Other supported metrics:**

- `cycle_knn`: kNN in A space -> look up corresponding B-space neighbors -> check if cycle-consistent.
- `lcs_knn`: Longest common subsequence of kNN orderings.
- `edit_distance_knn`: Edit distance between kNN orderings.
- `cka`: Linear CKA via HSIC.
- `unbiased_cka`: Unbiased CKA variant.
- `cknna`: Centered Kernel Nearest-Neighbor Alignment (restricts CKA to mutual nearest neighbors).
- `svcca`: Singular Vector CCA.

**6. The `Alignment` class API (`platonic/alignment.py`):**

High-level usage (from `examples/example_vision.py`):
```python
platonic_metric = platonic.Alignment(
    dataset="minhuh/prh", subset="wit_1024",
    models=["openllama_7b", "bloom_560m"]
)
images = platonic_metric.get_data(modality="image")

# ... extract your_model features as [N, num_layers, dim] ...

score = platonic_metric.score(your_feats, metric="mutual_knn", topk=10, normalize=True)
# Returns: {model_name: (best_score, (layer_i, layer_j))}
```

### Key design patterns

- Features are always stored as `[N, num_layers, D]` tensors (or lists of `[N, D]` tensors per layer).
- The best alignment is found by exhaustive search over all layer pairs.
- Normalization (L2) and outlier removal (95th percentile clamping) are critical preprocessing steps.

---

## Q3: What is the OpenVLA model structure?

### Answer (short)

OpenVLA is a 7B-parameter VLA built on Prismatic-7B. It has three main components: (1) a fused dual vision encoder (SigLIP + DINOv2), (2) a 2-layer MLP projector that maps visual features to the LLM embedding space, and (3) a Llama 2 7B language model backbone that autoregressively generates discretized action tokens. Input: one 224x224 image + one language instruction. Output: 7 action tokens (7-DoF end-effector control), each an integer in [0, 255].

### Details

**Architecture components:**

| Component | Model | Details |
|---|---|---|
| Vision Encoder | Fused SigLIP (ViT-SO400M) + DINOv2 (ViT-L/14) | ~600M params, features concatenated channel-wise |
| Projector | 2-layer MLP (FusedMLPProjector: Linear -> GELU -> Linear -> GELU -> Linear) | Maps fused vision features to LLM embedding dim |
| LLM Backbone | Llama 2 7B | 7B params, autoregressive generation |

**Vision encoder details:**

- Both SigLIP and DINOv2 process the same 224x224 image independently.
- Features are extracted from the **second-to-last transformer layer** (not the final layer):
  ```python
  self.featurizer.forward = unpack_tuple(
      partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
  )
  ```
- Output: patch features `[batch, num_patches, embed_dim]` (e.g., 256 patches for 16x16 grid).
- The two encoders' features are concatenated, then projected through the MLP.
- **Vision encoder is fine-tuned** during VLA training (opposite of VLM best practice; freezing it drops success rate from ~90% to ~25%).

**Forward pass flow:**

1. Image -> vision backbone -> `[batch, num_patches, vision_dim]`
2. Projected patches -> projector MLP -> `[batch, num_patches, llm_dim]`
3. Text instruction tokenized -> text embeddings -> `[batch, text_len, llm_dim]`
4. Concatenate: `[BOS, projected_patches, text_tokens]` -> LLM
5. LLM autoregressively generates action tokens

**Action tokenization (action_tokenizer.py):**

- Each continuous action dimension is independently discretized into 256 bins over [-1, 1] using `np.digitize()`.
- Bin boundaries: `np.linspace(-1, 1, 256)`.
- Token mapping: `token_id = vocab_size - discretized_bin_index` (overwrites last 256 tokens of Llama vocabulary).
- De-tokenization: `bin_index = vocab_size - token_id`, then map to bin center, then unnormalize using dataset q01/q99 statistics.
- Action normalization during training uses 1st and 99th percentile (not min-max).

**Training:**

- Pre-trained on Prismatic-7B (LLaVA 1.5 data, ~1M image-text pairs).
- Fine-tuned on 970k episodes from Open X-Embodiment (26+ manipulation datasets).
- 64 A100 GPUs, 14 days, 27 epochs, batch size 2048, LR 2e-5 fixed.
- Loss: next-token cross-entropy on action tokens only.
- All parameters fine-tuned (including vision encoder).

**Libero evaluation:**

- Fine-tuned via LoRA (r=32) on each Libero suite separately.
- Results: 84.7% (Spatial), 88.4% (Object), 79.2% (Goal), 53.7% (Long). Average 76.5%.
- Uses only third-person camera (no wrist), 256x256 images, filtered no-op actions.
- Prompt: "What action should the robot take to {task}?"

### Key files

- Model architecture: `prismatic/models/vlas/openvla.py`, `prismatic/models/vlms/prismatic.py`
- Vision encoder: `prismatic/models/backbones/vision/base_vision.py`
- Action tokenizer: `prismatic/vla/action_tokenizer.py`
- HF models: `prismatic/extern/hf/modeling_prismatic.py`
- Projector: `prismatic/util/nn_utils.py`
- Libero eval: `experiments/robot/libero/run_libero_eval.py`
- Inference: `experiments/robot/openvla_utils.py`
- Config: `prismatic/conf/models.py`, `prismatic/extern/hf/configuration_prismatic.py`

---

## Q4: How can we measure vision-action alignment, imitating the Platonic Representation Hypothesis?

### Answer (short)

We propose extracting vision embeddings from OpenVLA's dual vision encoder (SigLIP+DINOv2) and "action embeddings" from the LLM backbone's hidden states at the positions where action tokens are generated. We then compute mutual_knn alignment on paired (observation, action) data collected from Libero demonstrations. Multiple design choices and open questions remain.

### Detailed proposal

#### Core analogy with the Platonic paper

| Platonic paper | Our proposal |
|---|---|
| Vision observation of an event z | Robot camera image at timestep t |
| Language observation of an event z | Action taken at timestep t (conditioned on the image + instruction) |
| Vision embedding f_img(x_i) | Vision encoder features for image_t |
| Language embedding f_text(y_i) | Action-related features for action_t |
| Paired dataset {(x_i, y_i)} | Libero demonstrations {(image_t, action_t)} |
| mutual_knn measures if similar images -> similar text | mutual_knn measures if similar visual scenes -> similar actions |

The philosophical difference: in the Platonic paper, vision and language are two independent observations of the same underlying event, and alignment emerges because both converge to the same statistical kernel. In our setting, the action is not an independent observation of reality -- it is a *function* of the visual observation and the task instruction. This causal dependency (vision -> action) means we are measuring something different: whether the vision encoder's neighborhood structure predicts the action space's neighborhood structure.

#### Sub-question Q1: Which vision embedding to extract?

**Decision: Option A is the primary approach, Option B is the secondary.**

**Option A (primary) -- CLS token from each ViT layer (closest to PRH):**

Extract the CLS token from each transformer block of both SigLIP and DINOv2 (before fusion), exactly as the platonic-rep code does for TIMM ViTs:

```python
return_nodes = [f"blocks.{i}.add_1" for i in range(num_blocks)]
feats = [output[:, 0, :] for output in block_outputs]  # CLS token per layer
```

Then search over all layers for best alignment (same as PRH).

- Directly comparable to PRH methodology -- results can be placed side-by-side with the paper's vision-language alignment numbers.
- CLS token is a global summary -- may miss spatial details critical for action prediction, but this is the principled starting point.

**Option B (secondary) -- Patch tokens (spatial features):**

Instead of just CLS, use all patch tokens from a specific layer and average-pool them into a single vector per layer. For a 224x224 image with 16x16 patches, this gives 256 spatial tokens of dimension $D$; after average pooling we get one vector of dimension $D$ per layer.

- Actions are spatially grounded (e.g., "pick up the bowl" requires knowing where the bowl is), so spatial patch information may be more relevant than the global CLS summary.
- Average pooling follows the same pattern as PRH's treatment of LLM tokens (masked mean over sequence).
- Use this to check whether spatial features improve or degrade alignment relative to Option A.

#### Sub-question Q2: What should the action embedding be?

**Decision: Not yet made.** The core difficulty is understanding exactly where in the OpenVLA pipeline an "action embedding" lives, how it is generated, and how it maps back to physical robot motion. The diagram and walkthrough below are meant to clarify this before choosing.

##### How OpenVLA generates, decodes, and applies actions -- full pipeline diagram

```
+============================================================================+
|                    OPENVLA  INFERENCE  PIPELINE                             |
+============================================================================+
|                                                                            |
|  INPUT                                                                     |
|  -----                                                                     |
|  Camera image (256x256 RGB)          Task instruction (string)             |
|       |                                    |                               |
|       v                                    v                               |
|  Resize to 224x224               Tokenize with LlamaTokenizer              |
|       |                                    |                               |
|       v                                    v                               |
|  +------------------+              +--------------------+                  |
|  | Vision Backbone  |              |  Token Embedding   |                  |
|  | (SigLIP+DINOv2)  |              |  Layer (Llama 2)   |                  |
|  +------------------+              +--------------------+                  |
|       |                                    |                               |
|       | patch_features                     | text_embeddings               |
|       | [1, 256, D_vis]                    | [1, T_text, 4096]             |
|       v                                    |                               |
|  +------------------+                      |                               |
|  | MLP Projector    |                      |                               |
|  | (3-layer, GELU)  |                      |                               |
|  +------------------+                      |                               |
|       |                                    |                               |
|       | projected_patches                  |                               |
|       | [1, 256, 4096]                     |                               |
|       v                                    v                               |
|  +---------------------------------------------------------------+        |
|  | Concatenate: [BOS, projected_patches, text_tokens]             |        |
|  | Shape: [1, 1 + 256 + T_text, 4096]                            |        |
|  +---------------------------------------------------------------+        |
|                          |                                                 |
|                          v                                                 |
|  +---------------------------------------------------------------+        |
|  |                  Llama 2 (7B) LLM                              |        |
|  |                                                                |        |
|  |  32 transformer layers, each producing hidden states           |        |
|  |  hidden_states[layer_l] shape: [1, seq_len, 4096]             |        |
|  |                                                                |        |
|  |  Autoregressive generation (7 steps for 7-DoF):               |        |
|  |                                                                |        |
|  |  Step 1: logits -> sample -> action_token_1 (token ID)        |        |
|  |  Step 2: logits -> sample -> action_token_2 (token ID)        |        |
|  |  ...                                                           |        |
|  |  Step 7: logits -> sample -> action_token_7 (token ID)        |        |
|  +---------------------------------------------------------------+        |
|                          |                                                 |
|                          | 7 predicted token IDs                           |
|                          | e.g. [31888, 31743, 31801, 31900, ...]          |
|                          v                                                 |
+============================================================================+
|                    DETOKENIZATION  (action_tokenizer.py)                    |
+============================================================================+
|                                                                            |
|  Step 1: Token ID -> Bin index                                             |
|          bin_index = vocab_size - token_id                                 |
|          e.g. vocab_size=32000, token_id=31888 => bin_index=112            |
|                                                                            |
|  Step 2: Bin index -> Normalized continuous value in [-1, 1]               |
|          bins = np.linspace(-1, 1, 256)  # 256 boundary points            |
|          bin_centers = (bins[:-1] + bins[1:]) / 2  # 255 centers          |
|          normalized_value = bin_centers[clip(bin_index - 1, 0, 254)]       |
|          e.g. bin_index=112 => normalized_value ~= -0.126                  |
|                                                                            |
|  Step 3: Unnormalize using dataset statistics                              |
|          action = 0.5 * (normalized + 1) * (q99 - q01) + q01              |
|          where q01, q99 are per-dimension quantiles from training data     |
|          e.g. for dx: q01=-0.05, q99=0.05                                 |
|               action_dx = 0.5 * (-0.126 + 1) * (0.05 - (-0.05)) + (-0.05)|
|                         = 0.5 * 0.874 * 0.1 - 0.05                        |
|                         = -0.0063 meters                                   |
|                                                                            |
|  Result: 7-dim continuous action vector                                    |
|          [dx, dy, dz, droll, dpitch, dyaw, gripper]                        |
|          e.g. [-0.006, 0.012, -0.003, 0.001, 0.002, -0.015, 1.0]         |
|                                                                            |
+============================================================================+
|                    ROBOT  EXECUTION  (Libero env)                           |
+============================================================================+
|                                                                            |
|  The 7-dim vector is a DELTA end-effector command:                         |
|                                                                            |
|    dx, dy, dz     = translational displacement (meters)                    |
|    droll, dpitch,                                                          |
|    dyaw           = rotational displacement (radians)                      |
|    gripper         = gripper command (0=open, 1=close)                     |
|                                                                            |
|  Applied at each control step (~6 Hz):                                     |
|    new_pose = current_pose + delta                                         |
|    env.step(action) -> next_observation, reward, done                      |
|                                                                            |
|  One action per camera frame. No action chunking. Loop continues           |
|  until task success or max_steps (220-520 depending on suite).             |
|                                                                            |
+============================================================================+
```

##### Where could we extract "action embeddings"? -- annotated on the pipeline

The diagram above shows five distinct locations where an action-related representation exists. Each corresponds to one of the original options:

```
                         FORWARD PASS
                             |
  [Image] --> Vision Backbone ---> patch_features      <-- (Q1: vision embedding)
                             |          |
                             |     MLP Projector
                             |          |
                             |     projected_patches
                             |          |
                             v          v
                     +---------------------------+
                     |       Llama 2 LLM          |
                     |                           |
                     |  Layer 0 hidden states    | --.
                     |  Layer 1 hidden states    |   |
                     |  ...                      |   |--- Option A: LLM hidden states
                     |  Layer 31 hidden states   |   |    at action-token positions
                     |                           | --'    [N, 32, 4096]
                     |  output logits            |
                     +---------------------------+
                             |
                     action token IDs             ----- Option E: look up these IDs
                     [31888, 31743, ...]                 in the LLM embedding table
                             |                          [N, 7, 4096]
                             v
                     detokenize to bins
                             |
                     normalized [-1,1]
                             |
                     unnormalize
                             |
                             v
               7-dim continuous action             ----- Option B: raw action vector
               [dx,dy,dz,dr,dp,dy,grip]                 [N, 7]
                             |
                    (window of T steps)            ----- Option C: trajectory window
                    [dx...grip]_t to                     [N, 7*T]
                    [dx...grip]_{t+T-1}
                             |
                    (learned encoder)              ----- Option D: trained encoder
                    MLP/Transformer                      [N, D_learned]
                             |
                             v
                     Robot executes delta
```

##### Analysis of each option with respect to the pipeline

**Option A -- LLM hidden states at action-token positions:**

These are the richest representations (4096-dim per layer, 32 layers). But there is a critical confound: at the point in the sequence where action tokens are generated, the LLM's hidden state has already "seen" the projected vision patches (they appear earlier in the sequence: `[BOS, patches, text, action_tokens]`). So the hidden state at action position $t$ is a function of the entire visual input. Measuring alignment between the vision encoder's output and these hidden states would partly measure "how well vision information propagates through the LLM" rather than "how well vision structure matches action structure."

However, the hidden states also encode the model's *decision* -- its prediction of which action to take. In the later layers, the representation is progressively transformed from a "perception" representation to a "decision" representation. The layer-search mechanism (from PRH) would naturally find the layer where this distinction matters most.

**Option B -- Raw continuous action vectors:**

The simplest option. The 7D vector `[dx, dy, dz, droll, dpitch, dyaw, gripper]` is what actually moves the robot. This directly tests: "do visually similar scenes produce physically similar motions?"

Concern: kNN in 7D works fine mathematically, but the action dimensions have very different semantics (translation vs rotation vs binary gripper). L2-normalize after per-dimension standardization to make distances meaningful.

**Option C -- Action trajectory windows:**

Concatenating a window of $T$ future actions gives a $7T$-dimensional vector. This captures the short-term plan rather than an instantaneous command. A single frame where the robot is approaching a bowl might have action `[0, 0.01, -0.005, 0, 0, 0, 0]` (move forward and down), but the next 10 frames describe the full grasp approach.

**Option D -- Learned action encoder:**

Requires a separate training phase, making the alignment measurement no longer "model-free." Deferred unless Options A-C prove inadequate.

**Option E -- Action token embeddings from LLM vocabulary:**

Each of the 7 action token IDs is looked up in the LLM's embedding table (the same table that maps English words to vectors). These 256 action token slots were originally the least-used tokens in Llama 2's vocabulary, then overwritten during VLA fine-tuning. After fine-tuning, these embeddings have been updated by gradient descent to represent action bins. However, the LLM embedding table is a relatively shallow representation (it is the input layer, not the output of deep processing). These embeddings may not carry much semantic structure about what the actions *mean* physically.

##### Decision status: open -- needs pilot experiment

The options are not mutually exclusive. A reasonable first experiment:

1. Start with **Option B** (raw continuous 7D actions) because it is confound-free and directly interpretable.
2. In parallel, extract **Option A** (LLM hidden states at action positions, all 32 layers) to compare.
3. If Option B shows meaningful variation in alignment across tasks/checkpoints, it may be sufficient. If alignment scores are near chance, move to Option C (trajectory windows) to get a richer action representation.

Key question to resolve: does the 7D action space have enough structure for kNN to be meaningful? A quick sanity check: compute mutual_knn(k=10) between vision features and raw actions on 1000 samples from a single Libero task, then compare against a shuffled baseline (randomly permute the action-image pairing). If the gap is significant, Option B is viable.

#### Sub-question Q3: Designing the vision-action alignment metric

**Direct adaptation of mutual_knn:**

Given $N$ paired samples $\{(image_i, action_i)\}$ from Libero demonstrations:

1. Extract vision features: $\phi_i = f_{vision}(image_i)$, shape $[N, D_v]$
2. Extract action features: $\psi_i = f_{action}(action_i)$, shape $[N, D_a]$
3. ($D_v$ and $D_a$ can differ -- kNN operates within each space independently)
4. L2-normalize both sets of features.
5. Compute $kNN(\phi, k)$ and $kNN(\psi, k)$ within each space.
6. Compute $mutual\_knn$ overlap as in PRH.

**Key difference from PRH: the role of instructions**

In PRH, the dataset pairs images and captions describing the *same* scene. In Libero, the action depends on both the image AND the instruction. Two identical scenes with different instructions should produce different actions (e.g., "pick up the red bowl" vs "pick up the blue bowl").

Approaches:
- **Per-task alignment:** Compute alignment separately for each Libero task. Within a single task, the instruction is constant, so vision-action alignment is purely about visual state -> action mapping. This is the cleanest setup.
- **Cross-task alignment:** Pool data across tasks but condition on instruction. This requires some way to account for instruction variation.
- **Instruction-conditioned vision features:** Instead of raw vision features, use the LLM's hidden states after processing both image and instruction (but before generating actions). This "conditioned vision" representation already incorporates task context.

**Recommendation:** Start with per-task alignment (simplest, cleanest interpretation). Each Libero task provides ~50 demonstrations x ~200 timesteps = ~10,000 paired samples, which is sufficient for mutual_knn with k=10.

**Multi-layer search:**

If using per-layer features (Option A for vision, Option A for action), search over all layer pairs (layer_v, layer_a) and report the maximum alignment score, exactly as PRH does.

**Proposed experimental variations:**

1. **Baseline: random model.** Measure alignment using a randomly initialized VLA. This gives the "chance" level of alignment.
2. **Pre-trained vs fine-tuned.** Compare alignment of the base OpenVLA (pre-trained on OXE) vs the Libero-fine-tuned version. Does fine-tuning increase vision-action alignment?
3. **Frozen vs unfrozen vision encoder.** The paper notes that freezing the vision encoder hurts VLA performance. Does it also reduce vision-action alignment?
4. **Cross-task generalization.** Extract features from one task, compute alignment on another. Does alignment transfer?
5. **Alignment vs success rate.** Across multiple fine-tuning checkpoints or model variants, does higher vision-action alignment correlate with higher task success rate?
6. **Scale.** If multiple VLA model sizes are available, does alignment increase with scale (as PRH predicts)?

#### Sub-question Q4: Technical details for Libero

**Data collection:**

- Libero provides 4 task suites: Spatial (10 tasks), Object (10 tasks), Goal (10 tasks), Long (10 tasks).
- Each task has ~50 human demonstrations.
- Each demonstration is a sequence of (image, action, proprioceptive_state) tuples at ~20 Hz.
- Images: 256x256 RGB from third-person camera.
- Actions: 7-DoF (delta xyz, delta euler angles, gripper open/close).

**Feature extraction pipeline (proposed):**

```
For each Libero task:
    For each demonstration episode:
        For each timestep t:
            image_t = env_observation["agentview_image"]  # 256x256 RGB
            action_t = demonstration_action[t]             # 7-DoF continuous
            
            # Resize image to 224x224 for OpenVLA
            # Run through OpenVLA vision encoder
            # Extract CLS token from each ViT layer (or chosen representation)
            
            # Store (vision_features_t, action_t) pair
    
    # After collecting all pairs for this task:
    # Compute mutual_knn alignment
```

**Practical considerations:**

1. **Temporal correlation:** Consecutive timesteps within an episode are highly correlated. This violates the i.i.d. assumption implicit in kNN metrics. Mitigation: subsample every K-th frame (e.g., K=5 or K=10), or sample one random frame per episode.

2. **Sample size:** With ~50 demos x ~200 steps per task, we have ~10,000 samples. After subsampling by 10x, ~1,000 samples -- comparable to the 1,024 used in PRH. k=10 should work.

3. **GPU memory:** Storing per-layer features for all ViT blocks for thousands of images requires significant memory. May need to batch feature extraction and save to disk.

4. **Normalization:** Actions in Libero are unnormalized continuous values with different scales per dimension. Should normalize each dimension to [-1, 1] using the q01/q99 statistics (same as OpenVLA's training normalization) before computing kNN.

5. **Gripper dimension:** The gripper open/close is nearly binary (0 or 1). This dimension may dominate kNN in action space. Consider excluding it or treating it separately.

6. **Stationary frames:** Many frames in demonstrations are near-stationary (action ~= 0). These will cluster together and inflate alignment scores artificially. Filter out frames where ||action|| < threshold.

### Open questions and uncertainties

**OQ1 (Fundamental): Is the "action" truly an independent observation of the same event?**

In the Platonic paper, images and text are two independent modalities observing the same reality. But in robotics, the action is the OUTPUT of a policy conditioned on the visual observation. The action is not an independent observation -- it is a function of the image (and instruction). This changes the interpretation: we are no longer measuring "convergence of two modalities to a shared kernel" but rather "how well the vision encoder's neighborhood structure predicts the policy's action structure." This is closer to measuring the "functional alignment" of the vision encoder with the downstream task.

Is this still meaningful? Arguably yes -- if the vision encoder produces features where visually similar scenes have similar actions, it means the encoder captures task-relevant visual structure. But the theoretical framework from the Platonic paper (PMI kernel convergence via bijective observations) does not directly apply.

**OQ2 (Dimensionality mismatch): How to handle 7D actions vs 1024D vision features?**

The kNN metric operates within each space independently, so the actual dimensionality difference does not break the method. However, kNN behavior is very different in low vs high dimensions (curse of dimensionality). In 7D, kNN is well-behaved but captures different "similarity" than in 1024D. May need to:
- Use different k values for the two spaces.
- Investigate how the metric behaves when one space is very low-dimensional.

**OQ3 (Confound): Is alignment trivially high if using LLM hidden states for actions?**

If we extract action embeddings from the LLM hidden states, these states have already processed the projected vision features. So there is a direct information flow: image -> vision encoder -> projector -> LLM hidden state. The LLM hidden states at action positions "contain" vision information by construction. This could make alignment trivially high and uninformative.

Mitigation: Use raw continuous actions (Option B) as the primary action representation. LLM hidden states (Option A) can be a secondary analysis, but results should be interpreted cautiously.

**OQ4 (Instruction confound): How to disentangle instruction from visual state?**

Within a single Libero task (constant instruction), the only varying inputs are the visual observation and the timestep within the episode. But across tasks, the instruction changes the desired action for identical scenes. Need to decide whether to analyze per-task only, or develop an instruction-aware alignment metric.

**OQ5 (Temporal structure): Should alignment be measured at the frame level or trajectory level?**

A single frame + action pair may be too noisy. Aggregating over short trajectory windows might give more stable and meaningful alignment. But this complicates the analogy with PRH, which uses single (image, caption) pairs.

**OQ6 (What does alignment measure for fine-tuning?)**

If we fine-tune OpenVLA on a Libero task, the vision encoder is updated to produce features useful for action prediction. This should increase vision-action alignment by construction (the model is optimized to map vision -> action). So measuring alignment before vs after fine-tuning tells us something about how much the visual representation was adapted for the task, but it is not surprising that alignment increases. The more interesting question is whether *cross-task* alignment also increases -- does fine-tuning on one task improve the vision encoder's alignment with actions on a different task?

**OQ7 (Benchmark): What constitutes a "strong" alignment score?**

In the Platonic paper, the best models achieve ~0.16 on mutual_knn (k=10) for cross-modal alignment. For within-modality (vision-vision), scores are much higher (~0.4-0.7). What should we expect for vision-action? If actions are very low-dimensional, kNN neighborhoods may be large, leading to high overlap by chance. Need to compute chance-level alignment (e.g., by shuffling the pairing between images and actions) as a baseline.

**OQ8 (Proprioception): Should we include proprioceptive state?**

OpenVLA does not use proprioceptive state, but the robot's joint positions and velocities are another "observation" of the same event z. Including proprioception could provide a richer action-side representation. However, this would make the setting less comparable to the PRH paper, which focuses on two modalities.

**OQ9 (Multiple cameras): What about wrist camera views?**

Libero provides both agentview (third-person) and wrist camera images. Using both could provide richer visual features. The question is whether to measure alignment separately for each camera view or fuse them.

**OQ10 (Connection to representation learning): What actionable insights does this give?**

If we find that vision-action alignment is high for successful policies and low for failed ones, this could be used as:
- A diagnostic tool for VLA training (monitor alignment during fine-tuning).
- A reward signal for RL-based VLA training (optimize for vision-action alignment).
- A model selection criterion (choose the checkpoint with highest alignment).
- An analysis tool for understanding what the vision encoder learns during VLA training.

---

## Q5: Standalone idea description and search prompts

### Standalone paragraph (no reference to PRH)

We propose a method for quantifying the structural alignment between a robot policy's visual representations and its action outputs. Given a Vision-Language-Action model deployed on manipulation tasks, we extract the internal embeddings from the vision encoder at each transformer layer and pair them with the corresponding continuous robot actions collected from demonstration data. We then measure alignment using mutual k-nearest-neighbor overlap: for each observation-action pair in the dataset, we independently find the k nearest neighbors in the vision embedding space and in the action space, and compute the fraction of neighbors shared between the two sets, averaged over all samples and maximized over all layer combinations. A high score indicates that the vision encoder has learned a representation whose neighborhood geometry mirrors the structure of the physical action space -- visually similar scenes produce physically similar motions. We study how this vision-action alignment varies across training stages (pre-trained vs fine-tuned), across tasks within a benchmark suite, and across model configurations (frozen vs fine-tuned vision encoder), and investigate whether it correlates with downstream task success rate and generalizes across tasks.

### Search prompts (5 versions -- focused)

**Version 1**

We want to measure whether a vision-based robot manipulation policy maps visually similar observations to similar actions. Given paired (image, action) data from demonstrations of a single manipulation task, we compare the neighborhood structure of the policy's visual embeddings against the neighborhood structure of its output actions: if two images are nearest neighbors in the vision feature space, are the corresponding actions also nearest neighbors in the action space? We use the overlap between these two neighbor sets as an alignment score.

Search instructions: Find papers (2023-2026) that measure or study the correspondence between visual representations and action outputs within a single robot manipulation task. Focus on work that asks whether visual similarity implies action similarity in learned visuomotor policies.

---

**Version 2**

In robot manipulation, a policy's vision encoder maps camera observations to features, and the policy head maps those features to actions. We ask: does the vision encoder learn a feature space whose similarity structure matches the similarity structure of the action space for a given task? We measure this by computing nearest-neighbor overlap between vision embeddings and actions on paired demonstration data. A high overlap means the encoder organizes scenes by action-relevance -- observations that require similar motions are embedded nearby.

Search instructions: Find papers (2023-2026) on the relationship between visual feature similarity and action similarity in robot policies. Include work on vision-action alignment, action-conditioned visual representations, and whether visual encoders in visuomotor policies learn action-relevant structure.

---

**Version 3**

We study vision-action alignment in robot manipulation policies: given a trained policy and a set of demonstrations for a task, we extract the policy's internal visual features and the corresponding output actions, then measure whether observations that are close in visual feature space also produce close actions. This directly tests whether the vision encoder captures the structure that matters for control -- not object recognition or scene understanding in general, but the specific visual distinctions that lead to different physical motions.

Search instructions: Find papers (2023-2026) that analyze whether robot policy visual encoders capture action-relevant structure. Include work on probing visuomotor representations, measuring vision-to-action correspondence, and evaluating whether learned visual features reflect motor or control structure rather than generic perceptual features.

---

**Version 4**

For a given robot manipulation task, we collect paired (observation, action) samples and ask: is the neighborhood structure of the policy's vision embeddings aligned with the neighborhood structure of the action space? Two scenes that look similar to the encoder should produce similar robot motions. We quantify this alignment using nearest-neighbor overlap and study how it changes across model checkpoints, task difficulty, and whether the vision encoder is frozen or fine-tuned.

Search instructions: Find papers (2023-2026) that study the alignment or correspondence between visual observation spaces and action spaces in imitation learning or visuomotor policy learning. Focus specifically on manipulation tasks and on metrics that compare the geometry of visual features to the geometry of action outputs.

---

**Version 5**

We measure vision-action representational alignment in robot manipulation. Given demonstration data from a task, we extract the vision encoder's internal features for each observation and pair them with the corresponding continuous robot actions. We then check: do nearest neighbors in the visual feature space share nearest neighbors in the action space? This metric tells us whether the visual encoder has organized observations by their control-relevant similarity -- not by appearance in general, but by what the robot should do.

Search instructions: Find papers (2023-2026) on representational alignment between vision and action in robotics. Include work on measuring whether pretrained or fine-tuned visual encoders learn representations that are structured according to action similarity, and studies that compare visual feature geometry with action-space geometry in manipulation policies.

---

## Archive

(No archived content yet.)
