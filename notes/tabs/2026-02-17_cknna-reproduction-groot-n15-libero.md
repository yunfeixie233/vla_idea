# CKNNA Reproduction: GR00T N1.5 on LIBERO

- Created: 2026-02-17
- Last updated: 2026-02-17 (round 4 -- Platonic interpretation)
- Tags: CKNNA, GR00T-N1.5, LIBERO, proprioceptive-alignment, reproduction, Figure-8, RS-CL, platonic-rep, feature-extraction, Eagle-2.5, backbone-features, flow-matching, platonic-interpretation, visual-kernel, proprioceptive-kernel

## Current state

- Goal: reproduce the CKNNA measurement from Figure 8 of the RS-CL paper (Kim et al., ICLR 2026), but for a single fixed pretrained GR00T N1.5 model evaluated on LIBERO.
- **Implementation complete.** Three scripts in `cknna/`:
  - `load_libero_data.py` (Phase 1): loads HDF5 demos, saves images + feats_B (proprioceptive states)
  - `extract_features_groot.py` (Phase 2): runs GR00T backbone on each transition, saves feats_A at two extraction points
  - `compute_cknna.py` (Phase 3): computes CKNNA between any feats_A and feats_B
- **Prerequisite not yet met**: LIBERO HDF5 demo files must be downloaded before running.
- Design follows the Platonic Representation Hypothesis codebase pattern: separate data loading, feature extraction, and metric computation. This allows fair multi-model comparison (same transitions for all models).
- **Platonic interpretation (Q2)**: CKNNA compares two kernels -- K (visual-language similarity from the VLM adapter) and L (proprioceptive-state similarity from robot sensors). Both are partial observations of the same underlying events z (the robot-and-scene state). Higher CKNNA means the visual representation's neighborhood structure matches the proprioceptive structure, i.e., the VLM is closer to the "platonic ideal" of capturing physical reality rather than superficial visual appearance.
- All six open questions resolved:
  - OQ1: Extract at BOTH points (raw backbone_features AND after process_backbone_output) and compare.
  - OQ2: LIBERO agent_pos is 8D: [eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)].
  - OQ3: REVISED to forward pass on LIBERO demo dataset (not rollouts). Rollouts are unfair for multi-model comparison.
  - OQ4: Mean pool across tokens (masked by attention_mask), matching the Platonic paper's LLM treatment.
  - OQ5: Same as paper -- window=16, stride=1.
  - OQ6: Raw unnormalized proprioceptive state.

## Questions asked

- Q1: Detailed implementation plan for CKNNA reproduction
- Q2: Platonic interpretation -- what two distributions does CKNNA compare, and what does higher CKNNA mean?
- OPEN QUESTIONS: see section at bottom

---

## Q1: Detailed implementation plan for CKNNA reproduction

### Step 0: Understanding what the RS-CL paper actually measures in Figure 8

The paper's Appendix C.3 ("CKNNA measurement") states:

> We randomly sample 10 trajectories per task in RoboCasa-Kitchen, totaling 240 trajectories.
> Each trajectory is processed with a window size of 16, yielding 4415 transitions.
> We extract the embeddings from the adapter module f_phi (used as conditioning inputs to the action decoder) along with the corresponding proprioceptive states.
> We follow the implementation of [Huh et al.] and report results with k=10.

Key details:
- feats_A: adapter output h, the condition embedding that feeds the action decoder. Shape per transition: unknown pooling -> single vector.
- feats_B: raw proprioceptive state vector (10D for RoboCasa: EE xyz + 6D rotation + gripper).
- Both are L2-normalized before CKNNA computation.
- CKNNA is computed on the full set of 4415 transitions at once (one N x N Gram matrix where N = 4415).
- k = 10 for nearest-neighbor selection.

### Step 1: Identify the feature extraction points in GR00T N1.5

**Architecture flow** (from lerobot codebase):

```
Images + text instruction
    |
    v
[Eagle 2.5 VLM] (frozen, intermediate layer hidden states)
    |
    v  eagle_output.hidden_states[select_layer]  (B, N_tokens, 2048)
    |
[eagle_linear]  (2048 -> 1536)   # this is the "adapter" in GR00T N1.5
    |
    v  backbone_features  (B, N_tokens, 1536)
    |
[FlowmatchingActionHead.process_backbone_output]:
    |   vlln (LayerNorm on last dim, 1536)
    |   vl_self_attention (SelfAttentionTransformer)
    |
    v  processed backbone_features  (B, N_tokens, 1536)
    |
[DiT cross-attention decoder]
    |   takes: sa_embs = [state_enc, future_tokens, action_enc]
    |          encoder_hidden_states = processed backbone_features
    |   flow matching denoising loop
    |
    v  predicted actions  (B, H, action_dim)
```

**feats_A (condition embedding) extraction point:**

The RS-CL paper defines feats_A as "the output of the adapter module f_phi (used as conditioning inputs to the action decoder)." In the GR00T N1.5 lerobot codebase, this maps to:

- **Option A**: `backbone_features` immediately after `EagleBackbone.forward()` -- shape (B, N_tokens, 1536). This is after eagle_linear but BEFORE the action head's LayerNorm + self-attention.
- **Option B**: The processed backbone_features after `FlowmatchingActionHead.process_backbone_output()` -- shape (B, N_tokens, 1536). This is AFTER LayerNorm + self-attention, and is what the DiT decoder actually cross-attends to.

Option B is closer to the RS-CL paper's definition ("conditioning inputs to the action decoder"), since in their architecture the adapter output directly conditions the DiT. In our GR00T N1.5, the action head applies an extra LN + self-attention before cross-attention, so Option B captures the final conditioning signal.

**OPEN QUESTION OQ1**: Which extraction point to use? Option A (raw backbone_features) or Option B (after process_backbone_output)? Option B is more faithful to "what the decoder actually sees," but Option A is closer to "adapter output."

**feats_B (proprioceptive state) extraction point:**

The proprioceptive state enters the model via `action_input.state` (shape B, 1, max_state_dim=64, zero-padded). Before padding, this is the raw state from the environment, min-max normalized to [-1, 1].

In LIBERO with `obs_type=pixels_agent_pos`, the state is "agent_pos" -- the robot's proprioceptive state. The actual dimensions depend on LIBERO's observation spec.

**RESOLVED OQ2**: `agent_pos` in LIBERO is **8-dimensional**:
- Dims 0-2: End-effector position (x, y, z) -- 3D
- Dims 3-5: End-effector orientation as axis-angle -- 3D (converted from quaternion via LiberoProcessorStep._quat2axisangle)
- Dims 6-7: Gripper joint positions -- 2D

Source: `lerobot/src/lerobot/processor/env_processor.py`, lines 75, 104, 114-153. The raw LIBERO observations contain robot0_eef_pos (3D), robot0_eef_quat (4D quaternion -> 3D axis-angle), and robot0_gripper_qpos (2D). Joint positions (7D) and velocities (7D) are available but NOT included in agent_pos.

Comparison with RS-CL paper: they use 10D (EE xyz + 6D rotation + 1D gripper) on RoboCasa. Our LIBERO state is 8D (EE xyz + 3D axis-angle + 2D gripper). The core information is similar (end-effector pose + gripper), but the representation differs (6D rotation vs 3D axis-angle, 1D gripper vs 2D gripper qpos).

### Step 2: Data source -- forward pass on LIBERO demo dataset

**Decision (revised)**: Option A -- load LIBERO demonstrations from HDF5 and run model backbone forward passes offline.

**Why not rollouts (Option B)**: Different models produce different rollout trajectories, making cross-model CKNNA comparison unfair. The RS-CL paper samples from the demo dataset. The rollout approach also ties feature extraction to each model's eval loop, making it non-reusable.

**Architecture** (follows the Platonic Representation Hypothesis codebase design):

```
Phase 1: Data loading  (model-independent, run once)
    LIBERO HDF5 -> list of (images, agent_pos, task_description) per transition
    Save: feats_B = agent_pos vectors  (N, 8)

Phase 2: Feature extraction  (one small function per model)
    For model X:
        Load data from Phase 1
        Model-specific preprocessing (e.g., Eagle encode for GR00T)
        Run backbone forward pass (no action decoding)
        Pool to single vector per transition
        Save: feats_A_modelX.pt  (N, D_model)

Phase 3: CKNNA computation  (model-independent)
    Load feats_A_modelX and feats_B
    L2-normalize both
    Compute CKNNA(feats_A_modelX, feats_B, k=10)
```

**LIBERO HDF5 structure** (confirmed from code investigation):
```
data/demo_N/obs/agentview_rgb    -- (T, 128, 128, 3) uint8
data/demo_N/obs/eye_in_hand_rgb  -- (T, 128, 128, 3) uint8
data/demo_N/obs/ee_pos           -- (T, 3) float32
data/demo_N/obs/ee_ori           -- (T, 3) float32 axis-angle
data/demo_N/obs/gripper_states   -- (T, 2) float32
data/demo_N/obs/joint_states     -- (T, 7) float32
data/demo_N/actions              -- (T, action_dim) float32
```

**Constructing feats_B from HDF5** (no model needed):
```python
agent_pos = np.concatenate([ee_pos[t], ee_ori[t], gripper_states[t]])  # (8,)
```

**GR00T N1.5 feature extraction from HDF5** (model-specific preprocessing):
1. Load images from HDF5, flip 180 degrees (matching LiberoProcessorStep convention)
2. Construct video tensor: (B, 1, 2, C, H, W) from agentview + eye_in_hand
3. Get task description from LIBERO task metadata
4. Run through GrootEagleEncodeStep + GrootEagleCollateStep (or call Eagle processor directly)
5. Run EagleBackbone.forward() -> backbone_features (B, N_tokens, 1536)
6. Run FlowmatchingActionHead.process_backbone_output() -> processed features (for point B)
7. Mean-pool across tokens using attention_mask -> (B, 1536)

**Important preprocessing detail**: The lerobot LiberoProcessorStep flips images 180 degrees (both H and W axes) before feeding to the model. When loading from HDF5, we must apply the same flip to match what the model was trained on.

### Step 3: Determine the pooling strategy for backbone_features

backbone_features has shape (B, N_tokens, 1536) where N_tokens varies (it includes image patch tokens from all camera views + text instruction tokens). CKNNA requires a single vector per data point for the Gram matrix.

The RS-CL paper does not specify how they pool from the token sequence to a single vector. The most plausible and standard approach is **mean pooling** across the token dimension (masked by attention_mask to exclude padding tokens). This gives a single 1536-dimensional vector per transition.

Alternative: use only the first token (CLS-like position), or use the last token. Mean pooling is far more standard for this type of analysis.

**Decision (tentative)**: Mean pool backbone_features across the token dimension (using attention_mask), yielding shape (N_transitions, 1536).

**OPEN QUESTION OQ4**: Do you agree with mean pooling, or do you have a preference for a different pooling strategy? The paper does not specify this detail.

### Step 4: Sampling strategy for transitions

The RS-CL paper's setup on RoboCasa:
- 24 tasks, 10 trajectories per task = 240 trajectories
- Window size 16, yielding 4415 transitions total

For LIBERO, the eval command targets:
- libero_spatial (10 tasks)
- libero_object (10 tasks)
- libero_10 (10 tasks)
- Total: 30 tasks

If we sample 10 trajectories per task: 300 trajectories. With varying trajectory lengths and a window size of 16, this would yield a similar order of magnitude of transitions.

**What "window size of 16" means**: The RS-CL paper uses an action horizon H=16. Each transition consists of a single observation frame plus the model's 16-step action chunk. So "processing with window size 16" likely means: for each starting frame t in a trajectory, one transition = (observation_t, proprioceptive_state_t, action_chunk_{t:t+16}). Consecutive transitions are obtained by sliding the window by 1 step. A trajectory of length T yields approximately T - 16 + 1 transitions.

For CKNNA, only the observation and proprioceptive state matter (not the action chunk). So the "window size" primarily affects how many transitions are extracted from each trajectory.

**OPEN QUESTION OQ5**: Should we use the same windowing (stride 1, window 16) as the paper, or simply use every timestep as a separate transition (which is equivalent to window=1)?

### Step 5: Implement feature extraction (dataset-based, 3-phase design)

The code is organized into three independent phases, following the Platonic Representation Hypothesis codebase pattern.

**Phase 1: load_libero_data.py** (run once, model-independent)

```python
import h5py
import numpy as np
import os

def load_libero_transitions(hdf5_paths, task_descriptions, n_demos_per_task=10, window=16):
    """Load transitions from LIBERO HDF5 files.
    
    Returns:
        images: list of dicts, each with "agentview" and "eye_in_hand" uint8 arrays
        agent_pos: np.ndarray (N, 8)  -- raw proprioceptive state
        metadata: list of dicts with task/demo/timestep info
    """
    all_images = []
    all_agent_pos = []
    all_meta = []

    for hdf5_path, task_desc in zip(hdf5_paths, task_descriptions):
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            selected = demos[:n_demos_per_task]

            for demo_name in selected:
                grp = f["data"][demo_name]
                T = grp.attrs["num_samples"]
                if T < window:
                    continue

                agentview = grp["obs/agentview_rgb"][()]      # (T, 128, 128, 3)
                eye_in_hand = grp["obs/eye_in_hand_rgb"][()]  # (T, 128, 128, 3)
                ee_pos = grp["obs/ee_pos"][()]                # (T, 3)
                ee_ori = grp["obs/ee_ori"][()]                # (T, 3)
                gripper = grp["obs/gripper_states"][()]        # (T, 2)

                # Flip images 180 degrees (matching LiberoProcessorStep)
                agentview = agentview[:, ::-1, ::-1, :]
                eye_in_hand = eye_in_hand[:, ::-1, ::-1, :]

                for t in range(T - window + 1):
                    all_images.append({
                        "agentview": agentview[t].copy(),
                        "eye_in_hand": eye_in_hand[t].copy(),
                    })
                    state = np.concatenate([ee_pos[t], ee_ori[t], gripper[t]])
                    all_agent_pos.append(state)
                    all_meta.append({
                        "task": task_desc,
                        "demo": demo_name,
                        "timestep": t,
                        "hdf5": os.path.basename(hdf5_path),
                    })

    return all_images, np.stack(all_agent_pos), all_meta
```

Output: `feats_B.pt` (N, 8) + `images/` folder or kept in memory + `metadata.json`.

**Phase 2: extract_features_groot.py** (model-specific, one script per model family)

```python
import torch
from lerobot.policies.groot.modeling_groot import GrootPolicy

def extract_groot_features(model_path, images, task_descriptions):
    """Extract backbone features from GR00T N1.5 for a list of transitions.
    
    Returns:
        feats_pointA: torch.Tensor (N, D) -- after EagleBackbone (D=2048 for GR00T-N1.5-3B)
        feats_pointB: torch.Tensor (N, D) -- after LN + SelfAttn
    """
    policy = GrootPolicy.from_pretrained(model_path)
    model = policy._groot_model
    model.eval()

    # Register hooks for point A and point B
    collected_A, collected_B, collected_mask = [], [], []

    def hook_backbone(module, inp, out):
        collected_A.append(out["backbone_features"].detach().cpu())
        collected_mask.append(out["backbone_attention_mask"].detach().cpu())

    def hook_vl_sa(module, inp, out):
        collected_B.append(out.detach().cpu())

    h1 = model.backbone.register_forward_hook(hook_backbone)
    h2 = model.action_head.vl_self_attention.register_forward_hook(hook_vl_sa)

    with torch.inference_mode():
        for i in range(len(images)):
            # Build input dict: video + language + dummy state/embodiment
            # (construct using GrootEagleEncodeStep or Eagle processor directly)
            batch = build_groot_input(images[i], task_descriptions[i], model)
            backbone_inputs, action_inputs = model.prepare_input(batch)
            backbone_outputs = model.backbone(backbone_inputs)
            model.action_head.process_backbone_output(backbone_outputs)
            # Hooks fire automatically

    h1.remove()
    h2.remove()

    # Pool: mean over tokens (masked)
    all_A = torch.cat(collected_A)        # (N, T, 1536)
    all_B = torch.cat(collected_B)        # (N, T, 1536)
    all_mask = torch.cat(collected_mask)  # (N, T)
    m = all_mask.unsqueeze(-1).float()

    feats_A = (all_A * m).sum(1) / m.sum(1)
    feats_B_proc = (all_B * m).sum(1) / m.sum(1)
    return feats_A, feats_B_proc
```

Output: `feats_A_groot_pointA.pt` (N, D), `feats_A_groot_pointB.pt` (N, D).  D=2048 for GR00T-N1.5-3B.

For a different model (e.g., OpenVLA): write `extract_features_openvla.py` with its own preprocessing. The interface is the same: images in -> feats_A out.

**Phase 3: compute_cknna.py** (model-independent, reuses platonic-rep)

```python
import torch
import torch.nn.functional as F
from metrics import AlignmentMetrics  # from platonic-rep

feats_A = torch.load("feats_A_groot_pointA.pt")  # (N, D)
feats_B = torch.load("feats_B.pt")                # (N, 8)

feats_A = F.normalize(feats_A.float(), p=2, dim=-1)
feats_B = F.normalize(feats_B.float(), p=2, dim=-1)

score = AlignmentMetrics.cknna(feats_A.cuda(), feats_B.cuda(), topk=10)
```

This phase works for ANY model's features -- just swap the feats_A file.

### Step 6: Compute CKNNA

The CKNNA function from platonic-rep/metrics.py can be reused directly. Key implementation details:
1. Computes kernel matrices K = feats_A @ feats_A^T, L = feats_B @ feats_B^T
2. For unbiased variant: fills diagonal with -inf, finds top-k per row
3. Creates binary masks mask_K, mask_L from top-k indices
4. Intersection mask = mask_K * mask_L
5. Computes HSIC_unbiased(mask * K, mask * L) for sim_kl
6. Computes HSIC_unbiased(mask_K * K, mask_K * K) for sim_kk (note: self-alignment uses own mask, not mutual)
7. Computes HSIC_unbiased(mask_L * L, mask_L * L) for sim_ll
8. Returns sim_kl / sqrt(sim_kk * sim_ll)

Memory consideration: for N = 4415 transitions, the Gram matrices are 4415 x 4415 = ~19.5M entries, which at float32 is ~78 MB each. This fits comfortably in GPU memory.

### Step 7: Expected output and interpretation

Since we are evaluating the baseline GR00T N1.5 (without RS-CL), the CKNNA score should be at the lower end of what Figure 8 shows. The figure compares:
- Baseline (flow-matching only): lower CKNNA
- RS-CL: higher CKNNA

We should get a single CKNNA number that corresponds to the baseline level. This number quantifies how well the pretrained GR00T N1.5 adapter output aligns with proprioceptive state neighborhood structure.

### Step 8: Potential extensions

- Compute CKNNA per task suite (libero_spatial, libero_object, libero_10) separately
- Vary k (5, 10, 20, 50) to assess sensitivity
- Compare with mutual_knn and CKA as additional metrics
- Extract features from different layers of the backbone (not just the final adapter output) to see if deeper/shallower layers have different alignment

---

## Q2: Platonic interpretation -- what two distributions does CKNNA compare, and what does higher CKNNA mean?

### Context

The Platonic Representation Hypothesis (Huh et al., ICML 2024) posits an underlying reality consisting of events z drawn from P(Z). Different modalities observe z through bijective observation functions. Neural network representations of these observations should converge toward a shared statistical structure -- the PMI kernel of event co-occurrences. CKNNA measures how closely two representation spaces agree in their local neighborhood structure.

### The underlying event z

In our CKNNA measurement on LIBERO, the underlying event z at each transition is the full physical state of the robot-and-scene: the robot's joint configuration, end-effector pose, gripper state, the objects' positions, the task progress, etc. This is the "platonic reality" that both modalities partially observe.

### The two representation spaces (kernels) compared

1. **K = feats_A @ feats_A^T** -- the similarity structure induced by the **visual-language modality**.
   - Observation function: camera images + text instruction -> VLM backbone (Eagle 2.5) -> adapter (eagle_linear) -> feats_A.
   - This kernel encodes "which events look similar according to the VLM."
   - Two events z_i and z_j are close under K if their images and instructions produce similar adapter embeddings.

2. **L = feats_B @ feats_B^T** -- the similarity structure induced by the **proprioceptive modality**.
   - Observation function: robot joint encoders / end-effector sensors -> raw 8D state (ee_pos + ee_ori + gripper_qpos).
   - This kernel encodes "which events have similar physical robot configurations."
   - Two events z_i and z_j are close under L if the robot was in a similar pose.

Both are partial projections of the same underlying events z, through different observation functions -- exactly the Platonic setup. Crucially, proprioception is an **independent** observation of z, not derived from vision. The joint encoders read the physical state directly, regardless of what the camera sees.

### What higher CKNNA indicates

Higher CKNNA means the local neighborhood structure of the visual representation **agrees** with that of the proprioceptive representation:
- The 10 nearest neighbors of transition i in VLM embedding space are also the 10 nearest neighbors in proprioceptive-state space.
- The relative similarity magnitudes within those neighbor sets are correlated (this is the CKA component of CKNNA, beyond simple mutual k-NN overlap).

In Platonic terms: higher CKNNA = the visual representation is **closer to the platonic ideal** -- the shared statistical structure of reality that both modalities should converge toward. The VLM's internal model of "which world-states are similar" agrees with the physical ground truth as measured by proprioception.

Lower CKNNA means the visual representation organizes events by **superficial visual appearance** (background textures, lighting, object colors) rather than by the robot's actual physical configuration. Two frames with the same background but different robot poses are close in K but far in L, dragging CKNNA down.

### Connection to RS-CL results

- Baseline GR00T N1.5 (flow-matching only): lower CKNNA, because the frozen VLM was never exposed to proprioceptive signals. Its kernel K reflects internet visual-language similarity, not physical-state similarity.
- RS-CL-trained model: higher CKNNA, because the contrastive loss explicitly pushes K toward L by pulling embeddings for similar proprioceptive states together.
- The empirical correlation between CKNNA and task success confirms that proximity to the "platonic ideal" (proprioceptive-visual convergence) translates to better manipulation performance.

---

## Resolved Open Questions

### OQ1: Feature extraction point -- RESOLVED: extract at BOTH points

**Decision**: Extract backbone_features at both:
- **(A)** EagleBackbone output: directly after eagle_linear (2048->1536)
- **(B)** After FlowmatchingActionHead.process_backbone_output (LN + SelfAttention)

Compute CKNNA for both and compare. This tells us whether the action head's preprocessing changes alignment structure.

### OQ2: Proprioceptive state in LIBERO -- RESOLVED: 8D

`agent_pos` in LIBERO is **8-dimensional**: [eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)].

Source: `lerobot/src/lerobot/processor/env_processor.py`, LiberoProcessorStep.
- Raw LIBERO obs: robot0_eef_pos (3D), robot0_eef_quat (4D->3D axis-angle), robot0_gripper_qpos (2D).
- Joint positions (7D) and velocities (7D) are available in robot_state but NOT included in agent_pos.

Comparison with RS-CL paper: they use 10D (EE xyz + 6D rotation + 1D gripper) on RoboCasa. Our LIBERO state is 8D (EE xyz + 3D axis-angle + 2D gripper). Core information is similar (end-effector pose + gripper), but representation differs.

### OQ3: Data source -- REVISED: forward pass on LIBERO demo dataset (option A)

**Previous decision**: option B (hook into live eval loop).

**Revised decision after re-analysis**: option A (dataset forward pass). Reasons:

**Why Option B (rollout hooking) is problematic for multi-model comparison:**

1. **Unfair comparison data**. Different models produce different rollout trajectories. A better model reaches further in the task, visiting states that a weaker model never sees. CKNNA measured on different data points is not comparable across models.
2. **Entangled with model-specific eval infrastructure**. Each VLA model has its own policy class, preprocessing pipeline, observation format, and forward method. Hooking into GR00T's eval loop does not transfer to OpenVLA, pi0, or any other model -- you would rewrite the hooks from scratch each time.
3. **Slow and non-deterministic**. Requires running the LIBERO simulator for every model evaluation. Stochasticity from rollouts adds noise.

**Why Option A (dataset forward pass) is better:**

1. **Fair comparison**. All models are evaluated on the exact same set of (observation, proprioceptive_state) pairs from the demo dataset. feats_B is identical for every model.
2. **Clean separation following the Platonic paper's architecture**:
   - Phase 1 (model-independent): load LIBERO HDF5 -> images + proprioceptive states
   - Phase 2 (model-specific, minimal code): for each model, preprocess + run backbone + pool -> save feats_A
   - Phase 3 (model-independent): compute CKNNA(feats_A, feats_B)
3. **Fast**. No simulation needed. Just forward passes through the backbone (no action decoding / denoising).
4. **Matches the RS-CL paper**. They "randomly sample 10 trajectories per task" from the dataset, not from rollouts.
5. **Reusable**. Adding a new model means writing one small function (model-specific preprocessing + backbone forward). Everything else is shared.

**Practical feasibility of Option A (confirmed by code investigation):**

LIBERO HDF5 files contain raw observations per timestep:
- `data/demo_N/obs/agentview_rgb` -- (T, 128, 128, 3) uint8
- `data/demo_N/obs/eye_in_hand_rgb` -- (T, 128, 128, 3) uint8
- `data/demo_N/obs/ee_pos` -- (T, 3) float32
- `data/demo_N/obs/ee_ori` -- (T, 3) float32 axis-angle
- `data/demo_N/obs/gripper_states` -- (T, 2) float32
- `data/demo_N/obs/joint_states` -- (T, 7) float32

The proprioceptive state (feats_B) can be constructed directly:
```
agent_pos = concat(ee_pos, ee_ori, gripper_states)  # (8,)
```

For GR00T N1.5, the backbone forward pass needs Eagle-processed images + text. We can reuse GrootEagleEncodeStep and GrootEagleCollateStep from the existing codebase, or call the Eagle processor directly. One important detail: the lerobot LiberoProcessorStep flips images 180 degrees before feeding to the model. When loading from HDF5, we must apply the same flip.

### OQ4: Pooling strategy -- RESOLVED: mean pool across tokens

The Platonic Representation Hypothesis paper/codebase uses:
- **avg pooling** (over tokens, masked by attention_mask) for LLMs
- **CLS token** (first token) for vision models (ViTs)

Pooling happens during feature extraction. Features are saved as (N, L, D) with one vector per sample per layer.

Since GR00T N1.5 backbone_features are VLM-derived token sequences (Eagle 2.5), **mean pooling over the token dimension** (masked by backbone_attention_mask) is the appropriate strategy. This matches the Platonic paper's treatment of LLMs.

Source: `platonic-rep/extract_features.py` lines 81-84 (LLM avg pooling); paper main.tex: "We extract the class tokens from the vision models and the average-pooled tokens from the language models."

### OQ5: Windowing -- RESOLVED: window=16, stride=1 (same as paper)

Each trajectory of length T yields T - 15 transitions. The "window" refers to the fact that each transition corresponds to one model forward pass that would predict a 16-step action chunk. For CKNNA, only the observation and proprioceptive state at the start of each window matter.

### OQ6: Normalization -- RESOLVED: raw unnormalized state

Use the raw proprioceptive state from the environment (before min-max normalization). L2 normalization before CKNNA handles scale. The RS-CL paper says they use "raw proprioceptive state."

In practice: intercept the state from the environment observation BEFORE it enters GrootPackInputsStep. The LiberoProcessorStep produces agent_pos (8D), and we capture this before it gets min-max normalized and zero-padded.

---

## Implementation checklist (all OQs resolved, 3-phase design)

1. [x] Resolve OQ1-OQ6 (OQ3 revised from B to A)

**Phase 1: Data loading (model-independent)**
2. [x] Locate LIBERO HDF5 files -- NOT on disk, must download first (see prerequisites)
3. [x] Write `cknna/load_libero_data.py`

**Phase 2: Feature extraction (per model)**
4. [x] Write `cknna/extract_features_groot.py`

**Phase 3: CKNNA computation (model-independent)**
5. [x] Write `cknna/compute_cknna.py`

**Remaining**
6. [ ] Download LIBERO HDF5 datasets
7. [ ] Run Phase 1 -> Phase 2 -> Phase 3 end-to-end
8. [ ] Record CKNNA scores in this note

---

## How to evaluate a model's CKNNA (end-to-end instructions)

### Prerequisites

1. Activate the groot_libero conda environment (same one used for GR00T evaluation):

```
eval "$(conda shell.bash hook)" && conda activate groot_libero
```

2. Download the LIBERO HDF5 demo datasets (one-time):

```
cd /home/ubuntu/verl/LIBERO
python benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface
```

This places HDF5 files under `{LIBERO_ROOT}/libero/datasets/{suite_name}/`, e.g.:
```
LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_..._demo.hdf5
LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_..._demo.hdf5
```

3. Install h5py if not already available: `pip install h5py`

### Step 1: Load demo data (run once, model-independent)

```
cd /home/ubuntu/verl

python cknna/load_libero_data.py \
    --dataset_dir /lambda/nfs/verl/Isaac-GR00T/external_dependencies/LIBERO/libero/datasets \
    --suites libero_spatial libero_object libero_goal \
    --n_demos 10 \
    --window 16 \
    --output_dir ./cknna_data
```

Outputs in `./cknna_data/`:
- `feats_B.pt` -- (N, 8) raw proprioceptive states
- `images.pt` -- agentview + eye_in_hand images as uint8 tensors
- `metadata.json` -- per-transition info (task, demo, timestep)

### Step 2: Extract model backbone features (per model)

For GR00T N1.5 (aractingi/groot-libero-latest):

```
python cknna/extract_features_groot.py \
    --model_path aractingi/groot-libero-latest \
    --data_dir ./cknna_data \
    --output_dir ./cknna_data \
    --device cuda
```

Outputs:
- `feats_A_pointA.pt` -- (N, D) after EagleBackbone (D=2048 for GR00T-N1.5-3B; project_to_dim=None means no projection)
- `feats_A_pointB.pt` -- (N, D) after action_head LN + SelfAttention

To evaluate a different GR00T checkpoint, change `--model_path` and `--output_dir`:

```
python cknna/extract_features_groot.py \
    --model_path /path/to/another/groot/checkpoint \
    --data_dir ./cknna_data \
    --output_dir ./cknna_data_other_model
```

### Step 3: Compute CKNNA (model-independent)

```
python cknna/compute_cknna.py \
    --feats_A ./cknna_data/feats_A_pointA.pt ./cknna_data/feats_A_pointB.pt \
    --feats_B ./cknna_data/feats_B.pt \
    --topk 10 \
    --also_mutual_knn \
    --output ./cknna_data/cknna_results.json
```

Output (printed and saved to JSON):
```
--- feats_A_pointA ---
  CKNNA (k=10): 0.XXXXXX
  mutual_knn (k=10): 0.XXXXXX

--- feats_A_pointB ---
  CKNNA (k=10): 0.XXXXXX
  mutual_knn (k=10): 0.XXXXXX
```

### Multi-model comparison

All lerobot VLA models are supported via a single generic extractor `cknna/extract_features.py`.

**Run all models at once:**
```
bash cknna/run_all.sh                  # full run, all models
bash cknna/run_all.sh --test           # smoke test, all models
bash cknna/run_all.sh --models groot pi0fast  # subset
```

**Run one model manually:**
```
python cknna/extract_features.py \
    --model_type pi0fast \
    --model_path lerobot/pi0fast-libero \
    --data_dir ./cknna_data \
    --output_dir ./cknna_data/pi0fast
```

Phase 3 compares all models whose feats_A.pt exist:
```
python cknna/compute_cknna.py \
    --feats_A ./cknna_data/groot/feats_A.pt ./cknna_data/pi0fast/feats_A.pt \
    --feats_B ./cknna_data/feats_B.pt \
    --topk 10 --also_mutual_knn \
    --output ./cknna_data/cknna_results.json
```

**Supported models and hook points:**

| Model | HF Path | VLM Backbone | Hook Module | Feature Dim |
|-------|---------|-------------|-------------|-------------|
| GR00T N1.5 | aractingi/groot-libero-latest | Eagle 2.5 | `_groot_model.backbone` | 2048 |
| Pi0Fast | lerobot/pi0fast-libero | PaliGemma (SigLIP + Gemma) | `model.paligemma_with_expert.paligemma.language_model` | 2048 |
| Pi0.5 | lerobot/pi05_libero_finetuned | PaliGemma (SigLIP + Gemma) | same as Pi0Fast | 2048 |
| SmolVLA | lerobot/smolvla_base | SmolVLM | `model.vlm_with_expert` | config.text_config.hidden_size |
| XVLA | lerobot/xvla-libero | Florence2 | `model.model.vlm.language_model.model.encoder` | projection_dim |

For each model, the hook captures the VLM backbone's output that conditions the action decoder. This is the same semantic quantity across models -- the "condition embedding" -- even though the internal architecture differs. Mean pooling across tokens (masked) produces a single vector per transition.

**Output directory structure:**
```
cknna_data/
  feats_B.pt            # (N, 8) proprioceptive states (shared)
  images.pt             # images (shared)
  metadata.json         # transition metadata (shared)
  groot/feats_A.pt      # (N, 2048) GR00T backbone features
  pi0fast/feats_A.pt    # (N, 2048) Pi0Fast backbone features
  pi05/feats_A.pt       # (N, 2048) Pi0.5 backbone features
  xvla/feats_A.pt       # (N, D) XVLA backbone features
  cknna_results.json    # all CKNNA scores
```

### What the output means

CKNNA score in [0, 1]:
- **Higher** = the backbone's internal representation neighborhood structure better matches
  the proprioceptive state neighborhood structure. When the 10 nearest neighbors of a
  transition in embedding space are also the 10 nearest neighbors in proprioceptive-state
  space, and their kernel similarities agree, CKNNA is high.
- **Lower** = the backbone clusters by visual appearance (background, objects) rather than
  by robot configuration. This is expected for a frozen VLM or a baseline trained without
  proprioceptive alignment.

For the RS-CL paper's Figure 8 context:
- The baseline GR00T N1.5 (flow-matching only) produces lower CKNNA.
- RS-CL-trained models produce higher CKNNA.
- Comparing across VLA families shows which backbone inherently captures more control-relevant structure.

### Code files

| File | Role | Phase |
|------|------|-------|
| `cknna/load_libero_data.py` | Load LIBERO HDF5 demos, save images + feats_B | 1 (model-independent) |
| `cknna/extract_features.py` | Generic: extract backbone features for any VLA model | 2 (model-specific config, shared code) |
| `cknna/extract_features_groot.py` | GR00T-specific: extracts both point A and point B features | 2 (GR00T only, legacy) |
| `cknna/compute_cknna.py` | Compute CKNNA between any feats_A and feats_B | 3 (model-independent) |
| `cknna/run_all.sh` | End-to-end runner for all models with logging | orchestration |