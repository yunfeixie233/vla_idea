# RoboTwin 2.0 Policy Evaluation Results
- Created: 2026-02-19
- Last updated: 2026-02-19
- Tags: RoboTwin, evaluation, ACT, DP3, RDT, benchmark, SAPIEN, beat_block_hammer, DP, pi0, pi05, openvla-oft, TinyVLA, DexVLA, GO1, LLaVA-VLA

## Current state
- Environment: conda `RoboTwin`, Python 3.10, torch 2.4.1+cu121, NVIDIA H100 80GB, SAPIEN 3.0.0b1
- 3 out of 11 supported policies evaluated successfully: ACT, DP3, RDT
- 8 policies skipped: no pre-trained fine-tuned checkpoints available on HuggingFace
- All evals ran on `beat_block_hammer` task, `demo_clean` config, aloha-agilex embodiment
- Full 100-episode evaluations not completed (each takes hours); confirmed end-to-end correctness with partial runs

## Questions asked
- Q1: Which policies have pre-trained checkpoints and can be evaluated?

## Q1: Evaluation results for all 11 RoboTwin 2.0 policies

### Answer (short)
3 policies (ACT, DP3, RDT) ran successfully. 8 skipped due to no pre-trained checkpoints.

### Evaluated successfully

#### ACT
- Checkpoint: `TianxingChen/RoboTwin2.0` -> `act_ckpt/act-beat_block_hammer/demo_clean-50/`
- Model: 83.9M parameters, ResNet-18 backbone, chunk_size=50, temporal aggregation
- Result: Ran multiple episodes (0/2 observed before kill). Eval code ran without errors.
- Command: `python script/eval_policy.py --config policy/ACT/deploy_policy.yml --overrides --task_name beat_block_hammer --task_config demo_clean --ckpt_setting demo_clean --ckpt_dir policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50 --seed 0 --temporal_agg true`

#### DP3
- Checkpoint: `TianxingChen/RoboTwin2.0` -> `DP3_ckpt/beat_block_hammer/3000.ckpt`
- Model: 262.4M parameters, PointNet encoder + diffusion UNet
- Result: 7/8 = 87.5% success rate (early kill at 8 episodes)
- Fixes required:
  1. `encode_obs()` returned lists instead of numpy arrays -- fixed with `np.array(..., dtype=np.float32)`
  2. `demo_clean.yml` has `pointcloud: false` -- created `demo_clean_pcd.yml` with `pointcloud: true`
  3. pytorch3d not installed -- installed from source (`pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation`)
- Command: `python script/eval_policy.py --config policy/DP3/deploy_policy.yml --overrides --task_name beat_block_hammer --task_config demo_clean_pcd --ckpt_setting demo_clean --expert_data_num 50 --seed 0 --policy_name DP3`

#### RDT
- Checkpoint: `TianxingChen/RoboTwin2.0` -> `rdt_ckpt/demo_clean/beat_block_hammer/mp_rank_00_model_states.pt`
- Model: 1.23B diffusion parameters, SigLIP vision encoder, T5-v1_1-xxl text encoder, language-conditioned
- Required base model downloads:
  - `google/t5-v1_1-xxl` (~45GB) -> `policy/weights/RDT/t5-v1_1-xxl/`
  - `google/siglip-so400m-patch14-384` (~3.5GB) -> `policy/weights/RDT/siglip-so400m-patch14-384/`
- Fine-tuned ckpt placed at: `policy/RDT/checkpoints/demo_clean/checkpoint-1/pytorch_model/mp_rank_00_model_states.pt`
- Result: 3/5 = 60.0% success rate (early kill at 5+ episodes). Language instructions varied per episode.
- Command: `python script/eval_policy.py --config policy/RDT/deploy_policy.yml --overrides --task_name beat_block_hammer --task_config demo_clean --ckpt_setting demo_clean --seed 0 --checkpoint_id 1 --policy_name RDT`

### Skipped (no fine-tuned checkpoint)

| Policy | Pre-trained ckpt on HF? | Module imports? | Blocking issue |
|--------|------------------------|-----------------|----------------|
| DP | No | Yes | Need: collect data + train 600 steps |
| pi0 | No | No (needs JAX) | Need: uv env, JAX, LoRA/full finetuning (>46GB for LoRA) |
| pi05 | No | No (needs JAX) | Same as pi0 |
| OpenVLA-oft | No | No (draccus, version conflicts) | Need: RLDS pipeline, torch 2.2.0 env, LoRA merge |
| TinyVLA | No | No (needs vla module) | Need: InternVL3-1B download, config.json edits, training |
| DexVLA | No | No (needs aloha_scripts) | Need: separate dexvla-robo conda env, fine-tuning |
| GO-1 | No | No (needs json_numpy) | Need: separate go1 conda env, server-client arch, AgiBot-World repo |
| LLaVA-VLA | No | Yes | Need: trained model_path + obs_statistics.yaml |

### Setup steps performed
1. Created conda env `RoboTwin` (Python 3.10)
2. Installed: torch 2.4.1+cu121, sapien 3.0.0b1, mplib 0.2.1, toppra, curobo, pytorch3d 0.7.8, flash-attn 2.8.3
3. Applied SAPIEN urdf_loader encoding patch + mplib planner `or collide` patch
4. Installed Vulkan + libnvidia-gl-570-server for headless GPU rendering
5. Downloaded assets from HuggingFace (embodiments, textures, objects) via `_download_assets.sh`
6. Downloaded checkpoints for ACT, DP3, RDT from `TianxingChen/RoboTwin2.0`
7. Downloaded T5-v1_1-xxl + SigLIP base weights for RDT

### Why 8 policies have code but no weights

Investigated via README, leaderboard, HuggingFace, and GitHub issues (#213, #222, #209, #248).

"Policies Support" in the README means **code integration** (deploy_policy.py, eval.sh, training scripts), NOT pre-trained weights. The benchmark is designed as a train+eval framework. Key evidence:

1. The leaderboard explicitly marks DexVLA, TinyVLA, LLaVA-VLA, OpenVLA-oft, Chain-of-Action as **"Pending"** -- contributed teams provided code but no official benchmark numbers yet.
2. HuggingFace only has 3 checkpoint sets: `act_ckpt`, `DP3_ckpt`, `rdt_ckpt`. A commenter on issue #213 explained: "I think others are too large to release."
3. Even DP and Pi0 (which have leaderboard numbers) lack published checkpoints. There is no `DP_ckpt` on HF; the maintainer likely conflated DP with DP3 when closing issue #213.
4. Contributed policies (TinyVLA/DexVLA from Midea, LLaVA-VLA from HKUST, GO-1 from GO-1 Team) are code-only contributions.
5. Issue #248 shows users actively training OpenVLA-oft themselves and encountering low success rates (~10-12% on beat_block_hammer), confirming no ready-to-use weights exist.

### Open items
- Full 100-episode evaluations not run (each takes 4-8 hours)
- DP3 pointcloud fix and encode_obs fix could be upstreamed
- openvla-oft install broke env (downgraded torch 2.4.1 -> 2.2.0); would need separate env
