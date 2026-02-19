# Master Index

## Vision-Action Alignment: From Platonic Representation Hypothesis to VLA (file: notes/tabs/2026-02-15_vision-action-alignment.md)
- Last updated: 2026-02-17 (round 5)
- Tags: platonic-representation-hypothesis, alignment, mutual-knn, openvla, vision-action, libero, VLA, kernel-alignment, SigLIP, DINOv2, action-embedding, action-tokenization, pipeline-diagram
- Summary:
  This tab investigates adapting the Platonic Representation Hypothesis (PRH) framework to measure alignment between a VLA model's vision encoder embeddings and action embeddings. Vision embedding extraction is decided: Option A (CLS token per ViT layer from SigLIP/DINOv2, separately) is primary, Option B (average-pooled patch tokens) is secondary. This follows the PRH methodology directly and allows layer-pair search for maximum alignment.

  Action embedding extraction is NOT yet decided. A detailed pipeline diagram of the full OpenVLA inference chain was added -- from camera image through vision encoding, LLM autoregressive generation of 7 action tokens, detokenization (token ID -> bin index -> normalized value -> unnormalized physical delta), and robot execution. Five candidate extraction points for action embeddings are annotated on this pipeline. A pilot experiment is proposed: compute mutual_knn on raw 7D actions vs a shuffled baseline to check if action space has enough structure for kNN to be meaningful.

  Ten open questions remain, most critically: actions are causally dependent on vision (not independent observations as in PRH), dimensionality mismatch between 7D actions and ~1024D vision features, and trivial-alignment confounds when using LLM hidden states.
- Key sections:
  - [Q1: How does the Platonic paper define concept and measure alignment](tabs/2026-02-15_vision-action-alignment.md#q1-how-does-the-platonic-paper-define-concept-and-measure-vision-language-alignment)
  - [Q2: How is alignment computed in the platonic-rep codebase](tabs/2026-02-15_vision-action-alignment.md#q2-how-is-the-alignment-metric-computed-in-the-platonic-rep-codebase)
  - [Q3: OpenVLA model structure](tabs/2026-02-15_vision-action-alignment.md#q3-what-is-the-openvla-model-structure)
  - [Q4: Vision-action alignment proposal](tabs/2026-02-15_vision-action-alignment.md#q4-how-can-we-measure-vision-action-alignment-imitating-the-platonic-representation-hypothesis)
  - [Q4/Q2: Action pipeline diagram and embedding options](tabs/2026-02-15_vision-action-alignment.md#how-openvla-generates-decodes-and-applies-actions----full-pipeline-diagram)
  - [Open questions and uncertainties](tabs/2026-02-15_vision-action-alignment.md#open-questions-and-uncertainties)
  - [Q5: Standalone idea description and search prompts](tabs/2026-02-15_vision-action-alignment.md#q5-standalone-idea-description-and-search-prompts)

## VINN: Representation Learning for Visual Imitation (file: notes/tabs/2026-02-17_vinn-representation-learning-visual-imitation.md)
- Last updated: 2026-02-17 (round 4)
- Tags: VINN, BYOL, self-supervised, locally-weighted-regression, k-NN, non-parametric, visual-imitation, behavior-cloning, decoupling, ResNet-50, door-opening, task-structure-assumption, action-correctness, dataset-design, single-task
- Summary:
  Detailed reading of "The Surprising Effectiveness of Representation Learning for Visual Imitation" (Pari, Shafiullah, Arunachalam, Pinto, NYU). The paper decouples visual imitation into (1) BYOL self-supervised representation learning on raw robot frames (no action labels) and (2) non-parametric Locally Weighted Regression (k-NN weighted average of demonstration actions) at test time. No parametric policy is learned.

  The encoder receives zero action supervision during SSL; actions enter solely through the demonstration database at test-time LWR lookup. The bridge from "visual similarity" to "action correctness" is an unproven task-structure assumption: that visually similar states require similar actions. This is a property of the environment, not something the encoder learns. The "surprising effectiveness" in the title reflects that generic visual SSL features happen to correlate with action-relevant features for manipulation tasks -- an empirical finding, not a formal guarantee.

  Cross-verification with the VINN code (/home/ubuntu/verl/VINN) revealed four specific conditions that make the assumption hold: (1) single task per model -- each k-NN database contains one task only, no instruction conditioning exists in the code; (2) consistent demonstration strategy -- curated single-strategy demos (e.g., stacking always places closest on distant); (3) actions as local translational deltas (small per-frame SfM displacements); (4) low-dimensional action space (3D translation + separate discrete gripper). Removing any condition would break the assumption.
- Key sections:
  - [Q1: BYOL self-supervised training phase](tabs/2026-02-17_vinn-representation-learning-visual-imitation.md#q1-byol-self-supervised-training-phase-in-detail)
  - [Q2: Locally Weighted Regression with example](tabs/2026-02-17_vinn-representation-learning-visual-imitation.md#q2-locally-weighted-regression----how-it-works-supervision-specific-example)
  - [Q3: What bridges visual similarity to action correctness](tabs/2026-02-17_vinn-representation-learning-visual-imitation.md#q3-does-action-correctness-depend-entirely-on-representation-quality-what-bridges-visual-similarity-to-action-correctness)
  - [Q4: How the dataset design justifies the assumption (code-verified)](tabs/2026-02-17_vinn-representation-learning-visual-imitation.md#q4-how-does-the-paper-justify-similar-observation---similar-action-is-it-the-dataset-design)
  - [Q5: Single-task BYOL training, meaning of "goal", evidence for one-to-one mapping](tabs/2026-02-17_vinn-representation-learning-visual-imitation.md#q5-does-byol-train-on-single-task-or-all-task-data-what-does-goal-mean-evidence-for-one-to-one-mapping)

## RS-CL: Contrastive Representation Regularization for VLA Models (file: notes/tabs/2026-02-17_rscl-contrastive-representation-regularization-vla.md)
- Last updated: 2026-02-17 (round 2)
- Tags: RS-CL, contrastive-learning, VLA, proprioceptive-alignment, CKNNA, GR00T-N1.5, representation-regularization, flow-matching, view-cutoff, RoboCasa, LIBERO, Platonic-Representation-Hypothesis, adapter-output, condition-embedding, feats_A, feats_B, frozen-VLM
- Summary:
  Detailed reading of Kim et al. (KAIST / UC Berkeley, ICLR 2026). The paper introduces Robot State-aware Contrastive Loss (RS-CL), an auxiliary objective for VLA training that aligns VLM condition embeddings with robot proprioceptive states. It uses soft-weighted InfoNCE where weights derive from Euclidean distance between proprioceptive states in the batch, plus a view-cutoff augmentation that masks one camera view's features. The VLM stays frozen; only a lightweight adapter, projector, and action decoder are trained.

  Round 2 added detailed clarification of the CKNNA measurement inputs. feats_A is the adapter output h (shape N x d_model, e.g. N tokens x 2048) -- this is the vision-language condition representation, NOT an action representation. It sits upstream of the action decoder and encodes what the robot sees/reads. feats_B is the raw 10D proprioceptive state vector (EE xyz + 6D rotation + gripper). The full architecture is: frozen VLM (intermediate layer output) -> trainable adapter f_phi -> h (condition embedding, = feats_A) -> DiT action decoder (0.5B params, flow-matching). RS-CL's contrastive loss operates on a separate summarization-token branch but shares the adapter, so gradients reshape h indirectly.

  The CKNNA experiment is the closest published realization of our vision-action alignment idea (from the 2026-02-15 tab). Key design choice that improves on our proposal: proprioceptive state replaces actions as the physical-reality anchor, which avoids the causal-dependence confound (OQ1). The experiment confirms alignment is measurable, discriminative, and actionable -- higher CKNNA correlates with higher task success.
- Key sections:
  - [Q1: Core motivation, method, and findings](tabs/2026-02-17_rscl-contrastive-representation-regularization-vla.md#q1-core-motivation-method-and-findings)
  - [Q2: CKNNA measurement details](tabs/2026-02-17_rscl-contrastive-representation-regularization-vla.md#q2-how-does-this-paper-use-cknna-to-measure-alignment-between-condition-embeddings-and-proprioceptive-states)
  - [Q3: What are feats_A -- adapter output, not action representation](tabs/2026-02-17_rscl-contrastive-representation-regularization-vla.md#q3-what-exactly-are-feats_a-condition-embeddings-are-they-action-representations)
  - [Q4: What are feats_B -- proprioceptive state vectors](tabs/2026-02-17_rscl-contrastive-representation-regularization-vla.md#q4-what-exactly-are-feats_b-proprioceptive-state-vectors)
  - [Q5: Frozen VLM, decoder attachment, loss interaction](tabs/2026-02-17_rscl-contrastive-representation-regularization-vla.md#q5-what-exactly-is-frozen-how-is-the-decoder-attached-and-how-does-rs-cl-interact-with-the-action-loss)
  - [Q6: Relation to vision-action alignment idea](tabs/2026-02-17_rscl-contrastive-representation-regularization-vla.md#q6-relation-between-the-cknna-experiment-and-the-vision-action-alignment-idea)

## CKNNA Reproduction: GR00T N1.5 on LIBERO (file: notes/tabs/2026-02-17_cknna-reproduction-groot-n15-libero.md)
- Last updated: 2026-02-17 (round 4 -- Platonic interpretation)
- Tags: CKNNA, GR00T-N1.5, LIBERO, proprioceptive-alignment, reproduction, Figure-8, RS-CL, platonic-rep, feature-extraction, Eagle-2.5, backbone-features, flow-matching, implementation, platonic-interpretation, visual-kernel, proprioceptive-kernel
- Summary:
  Implementation complete for reproducing the CKNNA measurement from Figure 8 of the RS-CL paper (Kim et al., ICLR 2026). Three scripts in `cknna/`: (1) `load_libero_data.py` loads LIBERO HDF5 demos and saves raw proprioceptive states (feats_B, 8D) and flipped images; (2) `extract_features_groot.py` runs GR00T N1.5 Eagle backbone on each transition via forward hooks, mean-pools over tokens, and saves feats_A at two extraction points (after eagle_linear and after LN+SelfAttn); (3) `compute_cknna.py` computes CKNNA (and optionally mutual k-NN) between any feats_A and feats_B.

  The design follows the Platonic Representation Hypothesis codebase pattern, enabling fair multi-model comparison: Phase 1 data is shared across all models, Phase 2 is one small script per model family, Phase 3 is fully generic. OQ3 was revised from rollout hooking to dataset forward pass after recognizing that rollouts produce different data per model.

  Round 4 added a Platonic interpretation (Q2): in the PRH framework, the underlying event z is the full robot-and-scene state. CKNNA compares two kernels -- K (visual-language similarity from VLM adapter, i.e., the visual modality's observation of z) and L (proprioceptive-state similarity from joint encoders, i.e., the proprioceptive modality's observation of z). Higher CKNNA means the visual representation's local neighborhood structure matches the proprioceptive structure, indicating the VLM is closer to the "platonic ideal" of capturing physical reality rather than superficial visual appearance. Lower CKNNA means the VLM clusters by background/lighting/color instead of robot configuration.
- Key sections:
  - [Q1: Detailed implementation plan](tabs/2026-02-17_cknna-reproduction-groot-n15-libero.md#q1-detailed-implementation-plan-for-cknna-reproduction)
  - [Q2: Platonic interpretation of CKNNA](tabs/2026-02-17_cknna-reproduction-groot-n15-libero.md#q2-platonic-interpretation----what-two-distributions-does-cknna-compare-and-what-does-higher-cknna-mean)
  - [How to evaluate a model's CKNNA](tabs/2026-02-17_cknna-reproduction-groot-n15-libero.md#how-to-evaluate-a-models-cknna-end-to-end-instructions)
  - [Resolved open questions](tabs/2026-02-17_cknna-reproduction-groot-n15-libero.md#resolved-open-questions)

## Pi0Fast Environment Setup and CKNNA Test Run (file: notes/tabs/2026-02-17_pi0fast-env-setup-cknna.md)
- Last updated: 2026-02-19
- Tags: pi0fast, conda, pi0fast_env, transformers-fork, gated-model, paligemma, CKNNA, HF-token, forward-hook, GemmaModel, version-conflict, pyproject-conflicts
- Summary:
  Set up conda for all terminals (`~/.bashrc`), analyzed the `cknna/run_all.sh` three-phase pipeline (load LIBERO data, extract per-model features via forward hooks, compute CKNNA), and fixed three issues blocking Pi0Fast: (1) gated HF model access for `google/paligemma-3b-pt-224` via token login, (2) missing custom transformers fork (`fix/lerobot_openpi` branch) resolved by creating a new `pi0fast_env` venv that inherits from `groot_libero` via `--system-site-packages`, (3) forward hook not firing because `language_model.forward()` was called directly -- fixed by hooking on `language_model.norm` instead.

  A critical prefill-only optimization was added: the old code ran full autoregressive generation (257 forward passes per transition), capturing the wrong features (last decode token). The new `_pi0_prefill_only()` runs a single forward pass, giving ~56x speedup (24 it/s vs 0.43 it/s) and correct features. CKNNA jumped from 0.652 to 0.975, confirming the old features were wrong. Full run estimated ~25 min instead of ~23 hours.

  Q6 audit (2026-02-19): Confirmed that **no lerobot source code was modified** -- all pi0/pi0.5 workarounds live in `cknna/` scripts only. The separate `pi0fast_env` is required by an **upstream design limitation**: `pyproject.toml` explicitly declares `pi` as conflicting with every other model family (groot, smolvla, xvla, transformers-dep) and excludes it from `lerobot[all]` with a TODO to update to transformers v5. The fork pins transformers at 4.53.3 (older than the standard 4.57.6). The direct `.forward()` call that bypasses hooks is also upstream code.
- Key sections:
  - [Q2: How run_all.sh loads models](tabs/2026-02-17_pi0fast-env-setup-cknna.md#q2-how-runallsh-loads-models)
  - [Q3: Fix Pi0Fast gated model error](tabs/2026-02-17_pi0fast-env-setup-cknna.md#q3-fix-pi0fast-gated-model-error)
  - [Q4: Prefill-only speedup](tabs/2026-02-17_pi0fast-env-setup-cknna.md#q4-prefill-only-speedup-for-pi0fast-feature-extraction)
  - [Upstream version conflict](tabs/2026-02-17_pi0fast-env-setup-cknna.md#upstream-version-conflict-q6----2026-02-19)

## Repo Restructure and Git Setup (file: notes/tabs/2026-02-18_repo-restructure-git-setup.md)
- Last updated: 2026-02-18
- Tags: repo-structure, git, gitignore, cknna, lerobot, github, path-migration, pytorch-tensors, large-files
- Summary:
  Moved three directories (`cknna/`, `cknna_data/`, `cknna_data_test/`) from `/home/ubuntu/verl/` into `/home/ubuntu/verl/lerobot/`. Updated all Python `sys.path.insert` calls and all shell script absolute paths to reflect the new locations. The key changes were: `extract_features.py` and `extract_features_groot.py` dropped the redundant `lerobot/` path segment; `load_libero_data.py` added an extra `../` to reach the LIBERO repo two levels up; all four shell scripts had their `cd` and `RUN_DIR`/`DATA_DIR` variables updated.

  Extended `.gitignore` with patterns for large binary files: video formats (`*.mp4`, `*.avi`, `*.mov`, `*.webm`, `*.mkv`), PyTorch files (`*.pt`, `*.pth`, `*.ckpt`), runtime artifact dirs (`cknna/runs/`, `cknna/logs/`), data dirs (`cknna_data/`, `cknna_data_test/`), and eval output dirs (`eval_groot_*/`). Existing lerobot `.gitignore` already covered `logs/`, `*.log`, `__pycache__/`.

  Created new public GitHub repo `yunfeixie233/lerobot`. Updated the git remote from the original `huggingface/lerobot` upstream. Committed only source files (8 Python/shell scripts + updated `.gitignore`) and pushed successfully to `main`. Large files (`.pt` tensors, videos, run outputs) are entirely excluded.
- Key sections:
  - [Q1: Path migration details](tabs/2026-02-18_repo-restructure-git-setup.md#q1-move-cknna-directories-into-lerobot-and-fix-all-paths)
  - [Q2: Gitignore additions and what is ignored](tabs/2026-02-18_repo-restructure-git-setup.md#q2-git-setup-gitignore-and-push)

## Multi-Model CKNNA Testing: All LeRobot Models on LIBERO (file: notes/tabs/2026-02-19_multi-model-cknna-testing.md)
- Last updated: 2026-02-19 (round 2 -- hook audit, SmolVLA finetuned, VQ-BeT issue search)
- Tags: CKNNA, multi-model, ACT, Diffusion, VQ-BeT, SAC, HIL-SERL, TDMPC, Pi0.5, SmolVLA, LIBERO, feature-extraction, hook-design, IL, RL, VLA, pipeline-verification, hook-audit
- Summary:
  Extended the CKNNA feature extraction pipeline to cover all 10 supported lerobot models: 5 VLA (GR00T, Pi0Fast, Pi0.5, SmolVLA, XVLA), 3 IL (ACT, Diffusion, VQ-BeT), and 2 RL (SAC/HIL-SERL, TDMPC). Full hook audit completed: all 10 hooks verified correct for CKNNA semantics. Each hook captures the condition embedding for the action decoder (VLA) or the pure visual encoder output before state fusion (IL/RL).

  Round 2 updates: (1) SmolVLA now uses official LIBERO-finetuned checkpoint `HuggingFaceVLA/smolvla_libero`, eliminating 2 of 3 issues (image key mismatch and missing preprocessor); the hook bypass fix remains needed. (2) VQ-BeT `mlp_hidden_dim` config error is NOT a known issue on lerobot GitHub -- it's a silent backward-compatibility break. (3) No LIBERO-finetuned SAC or TDMPC checkpoints exist anywhere. (4) Cross-category CKNNA comparison caveat documented: VLA hooks capture vision+language, IL/RL hooks capture vision only, so scores are not directly comparable.

  Libero simulation eval remains blocked by MuJoCo/osmesa rendering dependencies (system-level).
- Key sections:
  - [Hook audit (Q3)](tabs/2026-02-19_multi-model-cknna-testing.md#q3-hook-audit----what-is-hooked-where-and-does-it-match-cknna)
  - [SmolVLA finetuned checkpoint (Q4)](tabs/2026-02-19_multi-model-cknna-testing.md#q4-smolvla-libero-finetuned-checkpoint)
  - [VQ-BeT config issue search (Q5)](tabs/2026-02-19_multi-model-cknna-testing.md#q5-vq-bet-config-error----is-it-a-known-lerobot-issue)
  - [Test results table](tabs/2026-02-19_multi-model-cknna-testing.md#test-results-20-transition-test-libero_spatial)
  - [Fixes applied](tabs/2026-02-19_multi-model-cknna-testing.md#fixes-applied)

## RoboTwin 2.0 Policy Evaluation Results (file: notes/tabs/2026-02-19_robotwin-eval-results.md)
- Last updated: 2026-02-19
- Tags: RoboTwin, evaluation, ACT, DP3, RDT, benchmark, SAPIEN, beat_block_hammer
- Summary:
  Ran end-to-end policy evaluation on the RoboTwin 2.0 benchmark for all 11 supported policies. Three policies (ACT, DP3, RDT) were evaluated successfully using pre-trained fine-tuned checkpoints from the official HuggingFace dataset (TianxingChen/RoboTwin2.0). The remaining 8 policies (DP, pi0, pi05, OpenVLA-oft, TinyVLA, DexVLA, GO-1, LLaVA-VLA) were skipped because no pre-trained fine-tuned checkpoints are available -- each would require collecting training data and running the full training pipeline first.

  ACT ran end-to-end without code changes. DP3 required three fixes: numpy array casting in encode_obs, a pointcloud-enabled task config (demo_clean has pointcloud:false by default), and pytorch3d installation. RDT required downloading three large base models (T5-v1_1-xxl ~45GB, SigLIP ~3.5GB) plus placing the fine-tuned checkpoint in the correct directory structure. Observed partial success rates: DP3 87.5% (7/8), RDT 60% (3/5) on beat_block_hammer. Full 100-episode runs take 4-8 hours each.

  Environment: conda RoboTwin, Python 3.10, torch 2.4.1+cu121, SAPIEN 3.0.0b1, NVIDIA H100 80GB. SAPIEN rendering required installing libnvidia-gl-570-server for Vulkan ICD on headless servers.
- Key sections:
  - [Evaluated successfully (ACT, DP3, RDT)](tabs/2026-02-19_robotwin-eval-results.md#evaluated-successfully)
  - [Skipped policies table](tabs/2026-02-19_robotwin-eval-results.md#skipped-no-fine-tuned-checkpoint)
  - [Setup steps](tabs/2026-02-19_robotwin-eval-results.md#setup-steps-performed)

## RoboTwin 2.0 Policy Setup/Installation/Eval Guide (file: notes/tabs/2026-02-19_robotwin-policy-setup-guide.md)
- Last updated: 2026-02-19
- Tags: RoboTwin, policy, setup, installation, eval, DP, ACT, DP3, RDT, Pi0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA, GO-1, deploy, benchmark
- Summary:
  Comprehensive reference summarizing setup, installation, checkpoint download, data processing, training, and evaluation instructions for all 11 RoboTwin 2.0 policy integrations: 3 traditional IL (DP, ACT, DP3), 1 diffusion transformer (RDT), 6 VLA/foundation models (Pi0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA, GO-1), plus the generic deploy-your-own-policy template.

  DP/ACT/DP3 are the simplest (pip install into existing RoboTwin env, no pretrained model downloads). RDT requires Python 3.10 and 3 large HF model downloads. Pi0 uniquely uses uv instead of pip. TinyVLA and DexVLA share a training conda env (dexvla-robo). GO-1 uses a server-client eval architecture with two separate conda environments and terminals. OpenVLA-oft requires TFDS dataset registration. LLaVA-VLA has a two-stage training process. All policies share a common eval pattern with results in `eval_result/` and support cross-evaluation (train on demo_clean, test on demo_randomized).

  GPU requirements vary: Pi0 full fine-tuning needs >100GB (2xA100), LoRA needs >46GB. Most others work on a single GPU. Known issues documented for DP3 (pointcloud requirement), RDT (single-GPU DeepSpeed), OpenVLA-oft (diffusers version), and GO-1 (long eval times needing tmux).
- Key sections:
  - [All 11 policies summary](tabs/2026-02-19_robotwin-policy-setup-guide.md#q1-setupinstaleval-summary-for-all-11-robotwin-policies)
  - [Common patterns](tabs/2026-02-19_robotwin-policy-setup-guide.md#common-patterns-across-all-policies)

## GR00T N1.6 LIBERO Evaluation Setup (file: notes/tabs/2026-02-17_gr00t-n16-libero-eval-setup.md)
- Last updated: 2026-02-17
- Tags: GR00T-N1.6, LIBERO, evaluation, setup, Isaac-GR00T, uv, osmesa, server-client, simulation, finetuning, H100
- Summary:
  Full end-to-end setup of the NVIDIA Isaac-GR00T N1.6 repository for LIBERO benchmark evaluation on 1x H100 80GB. The setup uses `uv` (not conda) for environment management. The main GR00T environment was created via `uv sync --python 3.10`, and a separate LIBERO simulation venv was built by `setup_libero.sh`. The LIBERO spatial dataset (432 episodes, 2 camera views) was downloaded from HuggingFace.

  Six errors were encountered and fixed: EGL rendering failure (workaround: osmesa), HF rate limiting (use local paths), incomplete dataset download, missing `libero_panda` modality config in the base model's processor (injected into a patched checkpoint), and missing `libero_sim` -> `LIBERO_PANDA` mapping in `env_utils.py`. The evaluation pipeline was verified end-to-end with 2 episodes on the base (unfinetuned) model -- 0% success as expected.

  Finetuning was also tested and confirmed working at ~1.54 it/s on single GPU (global_batch_size=64). Full 20K-step training takes ~3.6h on 1 H100. To reproduce benchmark numbers (97.65% spatial), finetuning with the original config (8 GPUs, batch 640) is needed.
- Key sections:
  - [Q1: Setup and evaluation pipeline](tabs/2026-02-17_gr00t-n16-libero-eval-setup.md#q1-end-to-end-setup-and-evaluation-pipeline)
