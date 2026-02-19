# GR00T N1.6 LIBERO Evaluation Setup
- Created: 2026-02-17
- Last updated: 2026-02-17
- Tags: GR00T-N1.6, LIBERO, evaluation, setup, Isaac-GR00T, uv, osmesa, server-client, simulation, finetuning

## Current state
- Full evaluation pipeline verified end-to-end on this machine (1x H100 80GB).
- Isaac-GR00T repo set up with `uv sync --python 3.10` + `uv pip install -e .`
- LIBERO sim venv created via `setup_libero.sh` at `gr00t/eval/sim/LIBERO/libero_uv/.venv/`
- EGL rendering fails due to NVIDIA driver/library mismatch (kernel 570.195 vs userspace 570.207). Workaround: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
- Base model `nvidia/GR00T-N1.6-3B` does not include `libero_panda` modality config. A patched checkpoint at `/tmp/gr00t_n16_libero_base` was created by injecting the config + stats + embodiment_id into symlinked copies of the base model files.
- `gr00t/eval/sim/env_utils.py` was patched to map `libero_sim` env names to `LIBERO_PANDA` embodiment tag (the upstream code does not handle this).
- LIBERO spatial dataset downloaded to `examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/` (432 episodes, 2 camera views).
- Finetuning was tested and works (~1.54 it/s on 1 GPU, global_batch_size=64). 20K steps ~3.6h.
- Evaluation ran 2 episodes: success_rate=0.0% (expected for unfinetuned base model).

## Questions asked

## Q1: End-to-end setup and evaluation pipeline
### Answer (short)
Works. Server-client architecture: `uv run python gr00t/eval/run_gr00t_server.py` (server) + `libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py` (client).

### Details
- **Server command** (Terminal 1):
  ```
  cd Isaac-GR00T
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python gr00t/eval/run_gr00t_server.py \
      --model-path /tmp/gr00t_n16_libero_base \
      --embodiment-tag LIBERO_PANDA \
      --use-sim-policy-wrapper
  ```
- **Client command** (Terminal 2):
  ```
  cd Isaac-GR00T
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 \
    gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
      --n_episodes 2 \
      --policy_client_host 127.0.0.1 \
      --policy_client_port 5555 \
      --max_episode_steps=720 \
      --env_name libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
      --n_action_steps 8 \
      --n_envs 1
  ```
- Video output saved to `/tmp/sim_eval_videos_libero_sim/`
- Each episode takes ~180s (~3 min) with 1 env.

### Errors encountered and fixes
1. **uv not installed**: Installed via `curl -LsSf https://astral.sh/uv/install.sh | sh` (v0.10.3).
2. **EGL rendering failure** (`ImportError: Cannot initialize a EGL device display`): Fixed with `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
3. **HuggingFace rate limiting**: Used local cached model path instead of `nvidia/GR00T-N1.6-3B` hub name.
4. **Missing video files**: Re-ran `huggingface-cli download` to complete partial download (only 100/432 wrist_image videos initially).
5. **KeyError: 'libero_panda'** in processor: Base model processor_config.json and statistics.json don't include LIBERO configs. Created patched checkpoint at `/tmp/gr00t_n16_libero_base` with injected modality config, stats, and embodiment ID.
6. **ValueError: 'libero_sim' is not a valid EmbodimentTag**: `env_utils.py` did not map `libero_sim` env prefix to `LIBERO_PANDA`. Added `is_libero_env()` and corresponding branch in `get_embodiment_tag_from_env_name()`.

### Open items
- To reproduce benchmark numbers (97.65% spatial), finetuning is required: 20K steps, global_batch_size=640 (original uses 8 GPUs). On 1 H100, use global_batch_size=64 with gradient accumulation, takes ~3.6h.
- For a faithful reproduction, match the original batch size of 640 or observe ~5-6% variance as noted in the README.
