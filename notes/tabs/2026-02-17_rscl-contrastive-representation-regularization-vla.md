# Contrastive Representation Regularization for Vision-Language-Action Models (RS-CL)

- Created: 2026-02-17
- Last updated: 2026-02-17 (round 2)
- Tags: RS-CL, contrastive-learning, VLA, proprioceptive-alignment, CKNNA, GR00T-N1.5, representation-regularization, flow-matching, view-cutoff, RoboCasa, LIBERO, kernel-alignment, Platonic-Representation-Hypothesis
- Paper: Kim, Lee, Koo, Kim, Lee, Kim, Seo, Shin (KAIST / UC Berkeley / RLWRLD), ICLR 2026

## Current state

- The paper introduces Robot State-aware Contrastive Loss (RS-CL), an auxiliary contrastive objective added to VLA training that explicitly aligns VLM condition embeddings with robot proprioceptive states.
- Core observation: pre-trained VLM embeddings cluster by visual appearance such as background and distractor objects, not by control-relevant signals such as robot pose or next action. This misalignment harms action prediction.
- RS-CL uses a soft-weighted InfoNCE loss where weights come from Euclidean distances between proprioceptive states in the training batch; embeddings for similar robot poses are pulled together.
- Three components: (1) a learnable summarization token appended to VLM output, (2) proprioceptive-state-based soft weights, and (3) view cutoff, a representation-level augmentation that masks one camera view's feature slice.
- The VLM backbone is frozen; only a lightweight adapter, projector, and the action decoder are trained. RS-CL adds negligible overhead.
- Validated on GR00T N1.5 (fine-tuning) and from-scratch training with Qwen2.5-VL-3B/7B, SigLIP2, and GR00T N1.5 VLM backbones. Consistent gains across all settings.
- Key result: RoboCasa-Kitchen pick-and-place jumps from 30.8% to 41.5% (+11.2pp). Real-robot tasks jump from 45.0% to 58.3% (+13.3pp).
- RS-CL outperforms VLMs further-trained on curated robotics datasets (RoboBrain, VeBrain, Cosmos-Reason, NORA) as conditioning models.
- CKNNA is used post-hoc as an analysis metric to verify that RS-CL increases representation-proprioception alignment relative to the baseline.
- feats_A in CKNNA = adapter output h, of shape N x d_model. This is not an action representation; it is the vision-language condition representation that the action decoder reads. It encodes what the robot sees and what instruction it received, processed through the frozen VLM and trainable adapter.
- feats_B in CKNNA = raw proprioceptive state vector q, 10-dimensional: end-effector xyz + 6D rotation + gripper. This is a sensor reading, not a learned representation.
- The VLM backbone (Eagle 2.5 / Qwen2.5-VL / SigLIP2) is fully frozen. Only the adapter f_phi, action decoder D_theta, summarization token u, and projector g_psi are trained.
- The CKNNA experiment in this paper directly tests the same core question as our vision-action alignment note: "does the VLA's internal representation's neighborhood structure match the neighborhood structure of the robot's physical state?" The key difference is that this paper uses proprioceptive state rather than actions as the physical-reality anchor, which avoids the causal-dependence confound identified in OQ1.

## Questions asked

- Q1: What is the core motivation, method, and findings of this paper?
- Q2: How does this paper use CKNNA to measure alignment between condition embeddings and proprioceptive states?
- Q3: What exactly are feats_A (adapter output embeddings)? Are they action representations? Dimensionality? Origin in architecture?
- Q4: What exactly are feats_B (proprioceptive state vectors)? Contents and construction?
- Q5: What exactly is frozen, how is the decoder attached, how does RS-CL interact with the action prediction loss?
- Q6: What is the relation between the CKNNA experiment and the vision-action alignment idea in the earlier note?

---

## Q1: Core motivation, method, and findings

### Motivation

Vision-Language-Action (VLA) models condition an action decoder on representations from a pre-trained Vision-Language Model (VLM). The VLM was trained on internet-scale vision-language data and has never been exposed to robotic modalities such as low-level control actions or proprioceptive states. As a result, VLM embeddings are dominated by visual appearance cues such as which objects are present, what the background looks like, and the color palette, rather than control-relevant factors such as the robot's current end-effector pose or the contact state of the gripper.

The authors demonstrate this concretely by visualizing VLM embeddings of robot trajectories using t-SNE, for trajectories performing the same manipulation task ("Open the microwave / cabinet") across different kitchen environments in RoboCasa-Kitchen. The pre-trained VLM clusters these trajectories by visual environment such as background texture and distractor objects, not by task-progress or robot configuration. Two frames from different environments at the same manipulation stage, for example both reaching toward the handle, land far apart in VLM embedding space. This is a problem: the action decoder receives a representation that does not encode "where the robot is and what it should do next," forcing the decoder to learn this mapping entirely on its own from limited demonstration data.

Existing approaches to close this gap require extra training stages such as embodied reasoning pre-training, spatial grounding, or discretized action prediction, or carefully curated datasets. The authors want a method that: (a) operates in a single end-to-end training stage alongside the standard action prediction loss, (b) requires no additional datasets, (c) adds minimal computational overhead, and (d) is compatible with any VLA architecture that uses a frozen VLM plus adapter.

### Method: Robot State-aware Contrastive Loss (RS-CL)

The method adds a contrastive auxiliary loss to the standard VLA training objective (flow-matching for action prediction). The total loss is:

    L = L_FM + lambda * L_RS-CL

where lambda is initialized to 1.0 and cosine-decayed to 0 over training, placing early emphasis on representation shaping and late focus on action accuracy.

**Architecture (three added components):**

1. **Learnable summarization token.** The full VLM output h is a sequence of N tokens, where N can be hundreds to thousands. Applying contrastive learning directly on this full sequence is computationally expensive and dilutes the learning signal. Instead, a single learnable token u is appended to the VLM output and processed by the adapter f_phi. The adapter's output at this position, w, is a compact summary of the entire input. A lightweight 2-layer MLP projector g_psi maps w to a 128-dimensional embedding z = g_psi(w) used for contrastive learning. Crucially, the original h, excluding the summarization token's output, still feeds the action decoder unchanged.

2. **Proprioceptive-state-based soft weights.** Standard InfoNCE uses binary positive/negative pairs. RS-CL replaces this with soft weights derived from the Euclidean distance between proprioceptive states q_i and q_j (end-effector position xyz, 6D rotation, gripper state). The weight w_ij = softmax(-||q_i - q_j||_2 / beta) over j, with temperature beta = 1.0. In a batch of B samples, each sample i distributes its positive mass across all other samples proportionally to how close their proprioceptive states are. Samples with similar robot poses are weighted heavily as positives, while samples with distant poses act as negatives. There are no hard positives or negatives; the weighting forms a continuous spectrum.

   The contrastive loss is then:

       L_RS-CL = - sum_i sum_j w_ij * log( exp(sim(z_i, z_tilde_j)/tau) / sum_k exp(sim(z_i, z_tilde_k)/tau) )

   where tau = 0.2 is the cosine-similarity temperature, and z_tilde is the augmented version of z.

3. **View cutoff (representation-level augmentation).** VLA models typically receive observations from multiple camera views, such as 2 exterior and 1 wrist cameras. The VLM encodes all views into a single token sequence. View cutoff randomly selects one view and zeros out the corresponding slice of the VLM output before passing through the adapter. This generates an alternative embedding z_tilde without requiring a second VLM forward pass; only the lightweight adapter and projector re-process the masked representation. This simulates viewpoint variation and promotes view-invariant representations, which is especially useful when one camera becomes occluded during execution, for example when the wrist camera is blocked by a grasped object.

**Training setup:**

- Baseline: GR00T N1.5 (NVIDIA's VLA using Eagle 2.5 VLM, DiT action decoder with flow-matching).
- The VLM backbone is frozen; the adapter f_phi, projector g_psi, summarization token u, and action decoder D_theta are trained.
- Projector: 2-layer MLP, hidden dim 2048, output dim 128.
- The proprioceptive state q consists of end-effector position (x, y, z), 6D rotation, and gripper state. In real-world joint-space experiments, absolute joint positions of 7 DoF are used instead.

### Key findings

**Fine-tuning on pre-trained VLA (GR00T N1.5):**

- RoboCasa-Kitchen (24 tasks):
  - 30 demos: 48.2% -> 53.0% (+4.8pp overall), pick-and-place: 30.8% -> 41.5% (+11.2pp)
  - 100 demos: 63.9% -> 67.2% (+3.3pp)
  - 300 demos: 65.7% -> 69.7% (+4.0pp)
- LIBERO (4 suites, 40 tasks total): 95.7% -> 96.4% (+0.7pp)
- Real-robot (Franka FR3, 5 tasks, 60 demos): 45.0% -> 58.3% (+13.3pp)

**From-scratch training (VLM + randomly initialized decoder):**

- RS-CL consistently improves all tested VLM backbones: Qwen2.5-VL-3B, Qwen2.5-VL-7B, SigLIP2, GR00T N1.5 VLM.
- RS-CL outperforms VLMs that were further-trained on curated robotics datasets (RoboBrain, VeBrain, Cosmos-Reason, NORA) when used as conditioning models.
- RS-CL gains stack on top of further-trained VLMs: applying RS-CL to RoboBrain/VeBrain/NORA gives additional improvement beyond either technique alone.
- SigLIP2 with unfrozen backbone: RS-CL jumps from 4.0% to 14.1% (+10.1pp), indicating larger gains when more parameters are trainable.

**Ablations:**

- Plain InfoNCE (no proprioceptive weights) already helps over baseline on RoboCasa, showing that contrastive representation regularization itself is valuable. But RS-CL's soft proprioceptive weights add further improvement.
- Using next-action distance as soft labels performs worse than plain InfoNCE. The authors attribute this to the fact that the next action is already the prediction target, making it an unreliable alignment signal. Proprioceptive state is a stable, independent signal.
- View cutoff outperforms other representation-level augmentations such as token cutoff and span cutoff from the NLP cutoff paper. The multi-view structure of robotic observations is key.

---

## Q2: How does this paper use CKNNA to measure alignment between condition embeddings and proprioceptive states?

### Answer (short)

CKNNA (Centered Kernel Nearest-Neighbor Alignment) is used as a post-hoc diagnostic metric, not as part of training. After training a VLA model with or without RS-CL, the authors extract the adapter's output embeddings and the corresponding proprioceptive states from a large sample of transitions, then compute CKNNA between these two representation spaces. Higher CKNNA means the condition embeddings' neighborhood structure is better aligned with proprioceptive-state similarity. The result confirms that RS-CL significantly increases this alignment relative to the baseline trained only with action prediction loss.

### What is CKNNA?

CKNNA is a metric proposed by Huh et al. in the Platonic Representation Hypothesis paper. It bridges two classical representation comparison approaches:

1. **CKA (Centered Kernel Alignment):** Compares two representations by computing the Hilbert-Schmidt Independence Criterion (HSIC) between their Gram matrices (kernel matrices K = XX^T, L = YY^T), normalized: CKA = HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L)). CKA is sensitive to all pairwise distances globally, which means it can be dominated by large-scale structure and may miss local neighborhood patterns.

2. **Mutual k-NN:** Compares two representations by checking how much their k-nearest-neighbor sets overlap. It is a purely local metric, caring only about whether nearby points in one space are also nearby in the other. The Platonic paper found mutual k-NN to be much more sensitive to convergence trends than CKA.

CKNNA combines the two ideas: it restricts CKA to operate only within the mutual k-nearest-neighbor set. Concretely, the algorithm works as follows:

**Step 1: Compute Gram (kernel) matrices.**
Given N data points, let feats_A be the N x d_A matrix of condition embeddings and feats_B be the N x d_B matrix of proprioceptive states, both L2-normalized. Compute linear kernel matrices:
- K = feats_A @ feats_A^T (N x N)
- L = feats_B @ feats_B^T (N x N)

**Step 2: Identify k-nearest neighbors in each space.**
For each row i in K, with the diagonal set to -inf to exclude self-similarity in the unbiased variant, find the top-k indices. These are sample i's k nearest neighbors in the condition embedding space. Similarly, for each row i in L, find the top-k indices in the proprioceptive state space.

**Step 3: Compute mutual nearest-neighbor masks.**
For each pair (i,j), create binary masks:
- mask_K[i,j] = 1 if j is among i's k-NN in condition embedding space
- mask_L[i,j] = 1 if j is among i's k-NN in proprioceptive state space
- mask[i,j] = mask_K[i,j] * mask_L[i,j], their intersection, set to 1 only if j is a k-NN of i in both spaces

**Step 4: Apply masked CKA.**
Instead of computing HSIC on the full kernel matrices K and L, compute it on the masked versions (mask * K) and (mask * L). This means only the kernel entries for pairs that are mutual nearest neighbors contribute to the alignment score. The HSIC computation follows the unbiased estimator from Song et al. (2012):

    HSIC_unbiased(K_tilde, L_tilde) = [ sum(K_tilde * L_tilde^T) + sum(K_tilde)*sum(L_tilde)/((m-1)(m-2)) - 2*sum(K_tilde @ L_tilde)/(m-2) ] / (m*(m-3))

where K_tilde and L_tilde have their diagonals zeroed.

**Step 5: Normalize.**
CKNNA(feats_A, feats_B) = HSIC(mask*K, mask*L) / sqrt( HSIC(mask_K*K, mask_K*K) * HSIC(mask_L*L, mask_L*L) )

The self-alignment terms in the denominator use the respective space's own mask (not the mutual mask), providing proper normalization.

### How the RS-CL paper applies CKNNA

**Data collection:**
The authors randomly sample 10 trajectories per task from the 24 tasks in RoboCasa-Kitchen, yielding 240 trajectories. Each trajectory is processed with a window size of 16 timesteps. This produces 4,415 transitions total.

**Embedding extraction:**
For each transition, two things are extracted:
1. **Condition embeddings (feats_A):** The output of the adapter module f_phi, specifically the representation h that gets fed as conditioning input to the action decoder. This is the representation that RS-CL is designed to shape. It is not the raw VLM output; it is the VLM output after being processed by the learnable adapter. For the GR00T N1.5 architecture, the VLM output comes from an intermediate layer (layer 12 of 36 for the 3B variant), and the adapter further transforms it.
2. **Proprioceptive states (feats_B):** The raw robot proprioceptive state vector q at that timestep (end-effector position xyz, 6D rotation, gripper state).

Both are L2-normalized before CKNNA computation.

**Metric computation:**
They follow the Huh et al. (Platonic Representation Hypothesis) implementation with k = 10. The CKNNA score ranges from 0 to 1, where higher values indicate that the local neighborhood structure of the condition embeddings matches the local neighborhood structure of the proprioceptive states more closely.

**What CKNNA measures in this context:**
For a given transition i, its k=10 nearest neighbors in condition-embedding space are the 10 transitions whose VLM-adapter outputs are most similar. Its k=10 nearest neighbors in proprioceptive-state space are the 10 transitions where the robot was in the most similar physical configuration. CKNNA asks whether these two neighbor sets are the same, and whether overlapping pairs show correlated similarity values in their kernel entries.

If the condition embeddings perfectly encoded the robot's physical state, the neighborhood sets would perfectly overlap and the kernel values would be correlated, producing high CKNNA. If the condition embeddings cluster by visual appearance instead, as the pre-trained VLM does, then visually similar frames from different robot configurations would be neighbors in embedding space but not in proprioceptive space, producing low CKNNA.

**Results (Figure 8 in the paper):**
The paper presents a plot titled "Alignment to proprioceptive states" comparing CKNNA scores for:
- The baseline VLA model (trained only with flow-matching action prediction loss L_FM)
- The RS-CL model (trained with L_FM + lambda * L_RS-CL)

RS-CL produces substantially higher CKNNA scores, confirming that the contrastive objective successfully reshapes the embedding space toward capturing control-relevant signals. The figure appears to show CKNNA measured across different training stages or adapter layers; the exact x-axis is described as "condition representations inside trained VLA models".

### Why CKNNA rather than mutual k-NN or plain CKA?

The choice of CKNNA over simpler alternatives is deliberate:

1. **CKA alone** is sensitive to global structure and can be dominated by a few large-magnitude directions. If the embedding space has one principal component that correlates with proprioception but everything else is noise, CKA might still report decent alignment even though the local neighborhood structure is poor. CKA also showed "very weak trends" in the Platonic paper's convergence analysis.

2. **Mutual k-NN alone** only measures binary overlap of neighbor sets. It throws away the actual similarity magnitudes. Two representations could have the same neighbor sets but very different distance profiles within those sets.

3. **CKNNA** gets the best of both: it restricts attention to local neighborhoods like mutual k-NN, but preserves the kernel value information like CKA. It measures whether points that are mutual neighbors also have correlated similarity patterns. This makes it sensitive to fine-grained alignment in the local neighborhood while being robust to global distributional differences.

### Connection to the Platonic Representation Hypothesis

The Platonic Representation Hypothesis (Huh et al., 2024) originally used CKNNA among other metrics to measure whether different neural network models, trained on different modalities such as vision and language, are converging toward a shared representation of reality. The core claim is that as models scale, their internal representations become more aligned because they are all approximating the same underlying statistical structure of the world.

This RS-CL paper repurposes the CKNNA metric for a different question: instead of asking whether two separately-trained models are converging, it asks whether the training objective RS-CL has successfully aligned the condition embeddings of a single VLA model with the robot's proprioceptive states. The conceptual link is that proprioceptive state represents a physical ground truth about the robot's configuration, and a good condition representation should reflect this physical reality, echoing the Platonic hypothesis's theme that good representations should converge toward the structure of reality.

### Practical details and interpretation caveats

- The 4,415 transitions come from 240 trajectories, 10 per task across 24 tasks, windowed at size 16. This is a modest sample size for CKNNA. The metric can be noisy at small N.
- k = 10 means each sample's neighborhood is about 0.2% of the dataset (10/4415). This is a very local measure.
- The condition embedding is the adapter output, not the summarization token z used in the contrastive loss. This means CKNNA measures whether the contrastive regularization has influenced the main conditioning pathway h, not just the contrastive projection head. This is important because it shows the representation improvement transfers from the auxiliary loss path to the actual action-conditioning path.
- Proprioceptive state is low-dimensional at 10D, comprising 3D position, 6D rotation, and 1D gripper, while condition embeddings are high-dimensional at d_model = 2048 or 3584 for a sequence of N tokens, though it is unclear whether the paper average-pools or uses a specific token. The dimensionality mismatch is handled by CKNNA's kernel-based formulation, which operates on N x N Gram matrices regardless of feature dimensionality.

---

## Q3: What exactly are feats_A (condition embeddings)? Are they action representations?

### Answer (short)

feats_A are not action representations. They are the condition embeddings, specifically the output of the adapter module f_phi applied to the frozen VLM's intermediate hidden states. These embeddings encode what the robot sees and what instruction it received. They are the input that the action decoder reads to decide what action to produce. Calling them action representations would be a mischaracterization. They are vision-language representations that have been adapted through the trainable adapter to better serve action prediction.

### Detailed explanation of what feats_A are

The information flow in the RS-CL architecture is:

```
Camera images (V views) + text instruction
        |
        v
  [Frozen VLM backbone]
  (e.g., Eagle 2.5 for GR00T N1.5, or Qwen2.5-VL, or SigLIP2)
  The VLM processes images and text jointly.
  Output is taken from an INTERMEDIATE layer -- not the final layer.
    - For Qwen2.5-VL-3B and GR00T N1.5 backbone: layer 12 of 36
    - For Qwen2.5-VL-7B: layer 18 of 28
    - For SigLIP2: final hidden layer
  This intermediate output is a token sequence of shape [N, d_model_vlm].
        |
        v
  [Trainable adapter f_phi]
  A lightweight module that transforms the VLM's intermediate output.
  Output: h = f_phi(VLM(...)) with shape [N, d_model]
        |
        +---------> This is feats_A (for CKNNA measurement)
        |
        v
  [Action decoder D_theta]  (receives h as conditioning, plus proprioceptive state q)
  A 16-layer DiT (Diffusion Transformer) with 0.5B parameters.
  Generates action chunk A_t via flow-matching (iterative denoising).
```

So feats_A = h, the adapter output. It is a sequence of N token embeddings, each of dimension d_model. The tokens correspond to the VLM's input tokens: image patch tokens from each camera view, plus text instruction tokens. There is nothing action-related in these tokens, as they have not yet passed through the action decoder.

### Dimensionality

The paper states: h in R^{N x d_model}, where:

- N = number of input tokens to the VLM. For a multi-view setup with 3 cameras (RoboCasa-Kitchen: 2 exterior + 1 wrist), N includes all image patch tokens from all views plus text tokens. For a Qwen2.5-VL style model, this could be hundreds to over a thousand tokens.
- d_model = the hidden dimension of the adapter output:
  - Qwen2.5-VL-3B: d_model = 2048
  - Qwen2.5-VL-7B: d_model = 3584
  - GR00T N1.5: likely 2048 (based on the 3B backbone)

For the CKNNA measurement, the paper states that they extract the embeddings from the adapter module f_phi, which are used as conditioning inputs to the action decoder. The paper does not specify how they reduce the N x d_model sequence to a single vector for CKNNA. Two plausible options are: (a) average-pool across the N tokens to get a single d_model-dimensional vector per transition, or (b) flatten to get a single N*d_model vector. Option (a) is far more practical and standard, since CKNNA requires computing an N_samples x N_samples Gram matrix and thus the per-sample representation must be a single vector, not a sequence.

### Why feats_A is not an action representation

The critical distinction is that feats_A sits upstream of the action decoder. It is the conditioning signal that tells the decoder what the world looks like and what the robot should do in natural language. The action decoder then uses this conditioning, combined with the proprioceptive state q, to produce the actual action via flow-matching denoising.

A useful analogy: if the action decoder is like a chef, feats_A is the recipe and description of available ingredients, not the food itself. The RS-CL paper's whole point is that the recipe, meaning the VLM condition embedding, was written in a language the chef cannot read well, describing visual appearance rather than control-relevant signals. RS-CL rewrites the recipe to include information about the robot's physical state.

### What feats_A DOES encode

After RS-CL training, feats_A encodes a mixture of:
1. Visual scene understanding (from the frozen VLM backbone)
2. Task instruction semantics (from the frozen VLM backbone)
3. Proprioceptive-state-aligned structure (injected by RS-CL via the trainable adapter)

The adapter f_phi is the only trainable component in the conditioning path since the VLM is frozen. So all the reshaping that RS-CL achieves is concentrated in this adapter. The CKNNA measurement on feats_A specifically tests whether the adapter has successfully restructured the VLM output to reflect proprioceptive signals.

---

## Q4: What exactly are feats_B (proprioceptive state vectors)?

### Answer (short)

feats_B is the raw robot proprioceptive state vector q at each timestep. It is not a learned representation; it is a direct sensor reading from the robot. It contains the end-effector position (x, y, z), a 6D rotation representation, and the gripper state.

### Contents of the proprioceptive state vector

The paper states: "For proprioceptive inputs, we primarily use the end-effector position (x, y, z), 6D rotation, and gripper state."

This gives a 10-dimensional vector:
- 3 dimensions: end-effector position (x, y, z) in Cartesian coordinates (meters)
- 6 dimensions: end-effector orientation in 6D rotation representation, the continuous representation from Zhou et al. "On the Continuity of Rotation Representations in Neural Networks", consisting of the first two columns of the rotation matrix and giving 6 floats instead of Euler angles or quaternions
- 1 dimension: gripper state (open/close, likely normalized between 0 and 1)

Total: 10D.

In real-world experiments with the Franka Research 3 using joint-space control, the paper uses absolute joint positions of the 7-DoF manipulator instead, giving a 7D vector of one angle per joint.

### How proprioceptive state enters the model

The proprioceptive state q is used in two places in the RS-CL architecture:

1. **As input to the action decoder D_theta.** The decoder receives both the condition embedding h and the proprioceptive state q: D_theta(h, A_t^s, q). This is standard in flow-matching VLA architectures; the decoder needs to know where the robot currently is to predict the next action. The paper follows pi0 and GR00T N1 conventions here.

2. **As the supervision signal for RS-CL soft weights.** The Euclidean distance ||q_i - q_j|| between two proprioceptive states in a training batch determines how strongly their condition embeddings should be pulled together at high weight w_ij or pushed apart at low weight w_ij. The proprioceptive state is not processed by any encoder for this purpose; the raw vector is used directly.

### How feats_B is constructed for CKNNA

For the CKNNA measurement: the paper extracts q at each transition and L2-normalizes it. No learned transformation is applied. The 10D vector is used as-is. CKNNA operates on Gram matrices of inner products, so the actual dimensionality of feats_B at 10 vs feats_A at 2048 does not cause a problem, since both produce 4415 x 4415 Gram matrices.

### Why proprioceptive state and not actions?

The paper explicitly ablates using next-action distance as the soft-weight target and finds it performs worse than plain InfoNCE. Their explanation: the action is the prediction target itself, so using it as an alignment signal creates a circular dependency, causing the model to align its input representation with its own output. Proprioceptive state is different: it is an independent observation of the robot's physical configuration that exists before the action decoder runs. It is stable, deterministic since it comes from joint encoders, and directly describes where the robot is, making it a clean alignment target.

---

## Q5: What exactly is frozen, how is the decoder attached, and how does RS-CL interact with the action loss?

### What exactly is frozen in the VLM backbone?

The entire VLM backbone is frozen. Every parameter in the pre-trained VLM is frozen, including all transformer layers, the vision encoder such as ViT, the text embedding layer, attention weights, feedforward networks, and layer norms. Zero gradients flow into any VLM parameter.

The paper states (Section 2.1): "In practice, we train a lightweight adapter module f_phi upon the VLM and freeze the VLM, following GR00T N1.5."

Furthermore, the VLM output used as the starting point is not even from the final layer. It is from an intermediate layer:
- GR00T N1.5 / Qwen2.5-VL-3B: layer 12 of 36
- Qwen2.5-VL-7B: layer 18 of 28
- SigLIP2: final layer

This means for the Qwen variants, the top 24 or 10 layers of the VLM are not even executed. The VLM is treated as a feature extractor up to a mid-level representation, and the adapter takes over from there.

The one exception is SigLIP2, where the paper experiments with both frozen and unfrozen settings. When unfrozen, RS-CL gives larger gains from 4.0% to 14.1%, versus 2.0% to 2.6% when frozen, suggesting that allowing the backbone to adapt amplifies the contrastive signal.

### How is the action decoder attached?

The action decoder D_theta is a 16-layer Diffusion Transformer (DiT) with approximately 0.5B parameters. It is connected to the VLM through the adapter:

```
Frozen VLM (layers 1..12) --> adapter f_phi --> h (condition embedding)
                                                  |
                                                  v
                                         Action decoder D_theta
                                         (DiT, 16 layers, 0.5B params)
                                         Inputs: h, A_t^s (noised action), q (proprioception)
                                         Output: predicted velocity field (epsilon - A_t)
```

The DiT architecture uses cross-attention or adaptive layer norm (adaLN) to condition its denoising process on h. At each denoising step, the decoder receives:
- h: the condition embedding from the adapter, encoding vision and language information
- A_t^s: the current noised version of the action chunk, interpolated between noise and ground truth at flow-matching timestep s
- q: the current proprioceptive state

The decoder is trained with the flow-matching objective: it predicts the "velocity" that interpolates between noise epsilon and the clean action A_t. After training, at inference time, it iteratively denoises starting from pure Gaussian noise to produce an action chunk of H future actions.

The connection is end-to-end differentiable through the adapter. Gradients from L_FM flow back through D_theta and f_phi but stop at the frozen VLM boundary. The action decoder's attention dimension matches d_model of the backbone, being 2048 for 3B variants and 3584 for 7B. The paper found that omitting a dimensionality-reduction projection before conditioning actually improves performance.

### How RS-CL interacts with the action prediction loss

The total training objective is:

    L = L_FM + lambda * L_RS-CL

These two losses operate on different outputs of the adapter but share the same adapter parameters f_phi:

**L_FM (flow-matching, action prediction):**
- Operates on: h (the N-token condition embedding from the adapter)
- Trainable parameters: theta (action decoder), phi (adapter)
- Purpose: predict correct actions

**L_RS-CL (robot-state-aware contrastive):**
- Operates on: z = g_psi(w), where w is the adapter's output at the summarization token position
- Trainable parameters: phi (adapter), psi (projector), u (summarization token)
- Purpose: reshape the representation space so that embeddings for similar proprioceptive states are nearby

The key interaction mechanism is that both losses backpropagate through the same adapter f_phi. The adapter must simultaneously:
(a) produce h that enables accurate action decoding via L_FM
(b) produce w that, when projected to z, reflects proprioceptive-state similarity via L_RS-CL

Since the summarization token u attends to all other tokens in the adapter through self-attention, shaping z also implicitly shapes h, as the adapter's internal representations are shared. This is the mechanism by which the contrastive loss on z transfers to the conditioning representation h.

The lambda schedule (cosine decay from 1.0 to 0) implements a curriculum:
- Early training: strong RS-CL signal reshapes the representation space
- Late training: RS-CL fades out, L_FM dominates, fine-tuning for accurate actions

This prevents the contrastive objective from interfering with final action accuracy while ensuring it has shaped the representation during the critical early learning phase.

---

## Q6: Relation between the CKNNA experiment and the vision-action alignment idea

### The vision-action alignment idea (from the earlier note)

The idea in `notes/tabs/2026-02-15_vision-action-alignment.md` proposes adapting the Platonic Representation Hypothesis (PRH) framework to measure alignment between a VLA model's vision encoder embeddings and action embeddings. The core question is whether the vision encoder learns a feature space whose similarity structure matches the similarity structure of the action space. If two observations are nearest neighbors in vision-feature space, are the corresponding actions also nearest neighbors in action space? This is measured using mutual k-NN or CKNNA on paired vision_features and actions from demonstration trajectories.

The idea identified several open questions:
- OQ1: Actions are causally dependent on vision (not independent observations), breaking the theoretical framework of PRH
- OQ2: Dimensionality mismatch (7D actions vs 1024D vision)
- OQ3: Using LLM hidden states might give trivially high alignment
- OQ8: Should proprioceptive state be included as another modality?
- OQ10: What actionable insights does alignment measurement give?

### How the RS-CL CKNNA experiment connects

The RS-CL paper's CKNNA experiment is essentially a realized version of the vision-action alignment idea, with two crucial design choices that address the open questions:

**Choice 1: Proprioceptive state replaces action as the "physical-reality anchor."**

Our note proposed measuring alignment between vision embeddings and actions. The RS-CL paper instead measures alignment between condition embeddings and proprioceptive state. This is a significant improvement because it resolves OQ1, the causal-dependence confound:

- In our proposal, actions are the output of the policy, which is a function of the visual observation. Measuring vision-action alignment partly measures how well the model's own decision-making pipeline works rather than how well the vision encoder captures physical structure. The action depends on the image by construction.
- In RS-CL, proprioceptive state is an independent sensor reading. The robot's joint positions and end-effector pose exist regardless of what the camera sees. Two different visual scenes can have the same proprioceptive state if the robot pose is the same but the background differs. Proprioceptive state is thus a genuinely independent modality, closer to the PRH setup where images and text are independent observations of the same event.

This also resolves OQ8, which asked whether proprioceptive state should be included. The RS-CL paper shows that yes, proprioceptive state is actually the better choice compared to actions.

**Choice 2: Condition embeddings (adapter output) replace raw vision encoder features.**

Our note proposed extracting CLS tokens from each ViT layer. The RS-CL paper extracts the adapter output h instead. The adapter sits between the frozen VLM and the action decoder, making it the last representation before actions are generated. This has a practical advantage: it measures alignment at the exact point where representation quality matters for action prediction. If the adapter output is aligned with proprioception, the action decoder receives control-relevant conditioning. If the raw ViT CLS token is aligned but the adapter destroys this structure, it would not help.

However, this choice means the RS-CL paper measures a different thing from our proposal. We proposed measuring the vision encoder's intrinsic alignment: does the pre-trained encoder already capture action-relevant structure? The RS-CL paper measures the trained adapter's alignment: has the training method successfully added proprioceptive structure? These are complementary questions.

**What the CKNNA experiment confirms (relevant to our idea):**

1. The core hypothesis is validated: it is possible to measure meaningful alignment between a VLA's internal representations and physical-state signals using k-NN-based metrics. CKNNA produces discriminative scores; RS-CL models score higher than baselines.

2. The hypothesis that VLM representations lack control-relevant structure is quantitatively confirmed. The baseline (trained only with L_FM) has lower CKNNA scores, meaning its condition embeddings do not naturally reflect proprioceptive-state similarity. RS-CL fixes this.

3. The measurement is actionable, addressing OQ10. Higher CKNNA correlates with better task success rates across all their experiments. This supports our note's speculation that alignment could serve as a diagnostic tool for VLA training and a model selection criterion.

**Key differences between the RS-CL experiment and our proposal:**

| Aspect | Our proposal | RS-CL CKNNA experiment |
|--------|-------------|----------------------|
| Side A | Vision encoder CLS tokens (per ViT layer) | Adapter output h (post-VLM, post-adapter) |
| Side B | Raw 7D actions (or LLM hidden states) | Raw 10D proprioceptive state |
| Metric | mutual_knn (as in PRH) | CKNNA (a kernel-weighted variant) |
| Purpose | Measure intrinsic vision-action alignment | Verify that RS-CL training reshapes representations |
| Model | OpenVLA (SigLIP + DINOv2 + Llama 2) | GR00T N1.5 (Eagle 2.5 VLM + DiT decoder) |
| Benchmark | Libero | RoboCasa-Kitchen |
| Layer search | Yes (all layer pairs) | No (single adapter output) |
| Causal confound | Present (actions depend on vision) | Avoided (proprioception is independent) |

The RS-CL paper's CKNNA experiment is the closest published work to the vision-action alignment idea. It validates the core approach of using k-NN kernel alignment to measure representation-physical-state correspondence in VLAs, while making a design choice of using proprioception instead of actions that sidesteps the most fundamental theoretical concern we identified. It also goes further by not just measuring alignment but actively improving it through training, showing that increased alignment leads to better manipulation performance.
