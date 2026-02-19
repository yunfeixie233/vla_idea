# RoboTwin 2.0 Policy Setup/Installation/Eval Guide
- Created: 2026-02-19
- Last updated: 2026-02-19
- Tags: RoboTwin, policy, setup, installation, evaluation, DP, ACT, DP3, RDT, Pi0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA, GO-1, deploy, benchmark

## Current state
- Fetched and summarized 11 RoboTwin 2.0 policy documentation pages.
- Policies span: 3 traditional IL (DP, ACT, DP3), 1 diffusion transformer (RDT), 6 VLA/foundation models (Pi0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA, GO-1), plus a generic deploy-your-own guide.
- Common pattern: most policies use `process_data.sh` -> `train.sh` -> `eval.sh`, results saved in `eval_result/`.
- DP, ACT, DP3 are simplest (pip install into RoboTwin env, no pretrained weights needed).
- RDT, Pi0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA, GO-1 all require model downloads and more complex env setup.
- TinyVLA and DexVLA share training/eval environments (`dexvla-robo` conda env).
- GO-1 uses a unique server-client eval architecture (two separate terminals/envs).
- Pi0 uses `uv` instead of pip for dependency management.
- Deploy-your-policy provides a template for integrating custom policies.

## Questions asked
- Q1: What are the setup/install/eval instructions for all 11 RoboTwin policies?

## Q1: Setup/Install/Eval summary for all 11 RoboTwin policies

### 1. DP (Diffusion Policy)
- **Env**: RoboTwin conda env, no special Python version.
- **Install**: `cd policy/DP && pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor sympy && pip install -e .`
- **Checkpoints**: None (trains from scratch).
- **Data**: `bash process_data.sh ${task} ${config} ${num}` (Zarr format).
- **Train**: `bash train.sh ${task} ${config} ${num} ${seed} ${action_dim} ${gpu}` (600 steps default, action_dim=14 for aloha-agilex).
- **Eval**: `bash eval.sh ${task} ${task_config} ${ckpt_setting} ${num} ${seed} ${gpu}`

### 2. ACT (Action Chunking Transformer)
- **Env**: RoboTwin conda env.
- **Install**: `cd policy/ACT && pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython && cd detr && pip install -e .`
- **Checkpoints**: None.
- **Data**: `bash process_data.sh ${task} ${config} ${num}`
- **Train**: `bash train.sh` (6000 steps default).
- **Eval**: `bash eval.sh` (same pattern as DP).

### 3. DP3 (3D Diffusion Policy)
- **Env**: RoboTwin conda env.
- **Install**: `cd policy/DP3/3D-Diffusion-Policy && pip install -e . && cd .. && pip install zarr==2.12.0 ...`
- **Checkpoints**: None.
- **CRITICAL**: Must set `data_type/pointcloud=true` during data collection or get ZeroDivisionError.
- **Data/Train/Eval**: Same pattern as DP (3000 steps default).

### 4. RDT (Robotics Diffusion Transformer)
- **Env**: Python 3.10 required. Overwrites RoboTwin env with official RDT env.
- **Install**: torch==2.1.0, flash-attn==2.7.2.post1, requirements.txt.
- **Checkpoints**: 3 HF downloads: t5-v1_1-xxl, siglip-so400m-patch14-384, rdt-1b -> `policy/weights/RDT/`.
- **Data**: `process_data_rdt.sh` -> HDF5, `generate.sh` -> config, copy data to `training_data/${model_name}`.
- **Train**: Edit model_config YAML (set GPU), `bash finetune.sh ${model_name}`.
- **Eval**: `bash eval.sh ${task} ${config} ${model_name} ${ckpt_id} ${seed} ${gpu}`
- **Issue**: Single-GPU doesn't save DeepSpeed model states; need special pretrained_model_name_or_path for continuation.

### 5. Pi0 (OpenPI)
- **Env**: Uses uv, not conda pip. `.venv` created by `uv sync`.
- **Install**: `pip install uv` in conda, `GIT_LFS_SKIP_SMUDGE=1 uv sync`. curobo needed for eval.
- **Data**: process_data_pi0.sh -> HDF5, generate.sh -> LeRobot dataset (large cache in ~/.cache).
- **Config**: Edit `src/openpi/training/config.py` _CONFIGS dict.
- **Train**: compute_norm_stats.py then finetune.sh. LoRA >46GB, Full >100GB.
- **Eval**: `bash eval.sh ${task} ${config} ${train_config} ${model_name} ${seed} ${gpu}`

### 6. OpenVLA-oft
- **Env**: Official openvla-oft env, overwrites RoboTwin env.
- **Install**: Clone openvla-oft, pip install -e ., flash-attn==2.5.5.
- **Data**: preprocess_aloha.sh -> TFDS conversion -> register in configs.py/transforms.py/mixtures.py.
- **Train**: `bash finetune_aloha.sh`, then `merge_lora.sh`.
- **Eval**: `bash eval.sh ${task} ${config} ${ckpt_path} ${seed} ${gpu} ${unnorm_key}`
- **Issue**: May need diffusers==0.33.1.

### 7. TinyVLA
- **Env**: Training: `conda env create -f Train_Tiny_DexVLA_train.yml` (dexvla-robo). Eval: RoboTwin env + requirements.txt.
- **Checkpoints**: InternVL3-1B from HF. Must edit config.json (architectures -> TinyVLA, model_type -> tinyvla).
- **Config**: Add task to aloha_scripts/constants.py.
- **Train**: Edit train_robotwin_aloha.sh, run it.
- **Eval**: Edit deploy_policy.yml, `bash eval.sh`.

### 8. DexVLA
- **Env**: Same as TinyVLA (shared dexvla-robo training env).
- **Checkpoints**: Qwen2-VL-2B-Instruct (edit config.json) + ScaleDP-H or ScaleDP-L pretrained head.
- **Train**: Edit vla_stage2_train.sh (OUTPUT must contain "qwen2"), run it.
- **Eval**: Edit deploy_policy.yml, `bash eval.sh`.
- **Note**: OUTPUT dir naming convention matters (must include "qwen2", optionally "lora").

### 9. LLaVA-VLA
- **Env**: Follow official LLaVA-VLA installation.
- **Checkpoints**: Download from LLaVA-VLA model zoo.
- **Data**: image_extraction.sh -> process_data.sh (with future_chunk param) -> merge_json.py -> yaml_general.py.
- **Train**: Two-stage: pre-training then fine-tuning, both via calvin_finetune_obs.sh.
- **Eval**: Edit deploy_policy.yml, `bash eval.sh ${gpu_id}` (only gpu_id argument).

### 10. GO-1
- **Env**: Two separate envs: RoboTwin (data + eval client) + GO-1 (server). Extra `pip install -r requirements.txt` in RoboTwin env.
- **Data**: robotwin2hdf5.sh (RoboTwin env) -> hdf52lerobot.sh (GO-1 env).
- **Train**: Follow AgiBot-World repo instructions.
- **Eval**: Server-client: `python evaluate/deploy.py` (GO-1 env) + `bash eval.sh` (RoboTwin env). Use tmux/screen.
- **Benchmark**: GO-1 80.25% avg, GO-1 Air 76.75%, Pi0 74%, ACT 51.25%, RDT 49.5%, DP 34.25%.

### 11. Deploy Your Policy (Custom)
- **Files**: deploy_policy.py, deploy_policy.yml, eval.sh.
- **Required functions**: get_model(usr_args), eval(TASK_ENV, model, observation).
- **Optional functions**: encode_obs(obs), reset_model(model), update_obs(obs), get_action(model, obs).
- **Action types**: qpos (joint), ee (end-effector), delta_ee (delta end-effector).
- **Run**: `bash eval.sh ...params...` -> calls `python script/eval_policy.py`.

### Common patterns across all policies
- Results always saved in `eval_result/` under project root.
- `task_config` = eval environment config; `ckpt_setting` = training data config.
- Cross-evaluation supported: train on demo_clean, eval on demo_randomized (or vice versa).
- Most use shell scripts: process_data.sh, train.sh/finetune.sh, eval.sh.
