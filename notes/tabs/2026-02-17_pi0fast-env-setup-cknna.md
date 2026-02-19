# Pi0Fast Environment Setup and CKNNA Test Run

- Created: 2026-02-17
- Last updated: 2026-02-19
- Tags: pi0fast, conda, pi0fast_env, transformers-fork, gated-model, paligemma, CKNNA, HF-token, forward-hook, GemmaModel, version-conflict, pyproject-conflicts

## Current state
- Conda initialized in `~/.bashrc` for all terminals. Available envs: `groot_libero`, `openvla`, `pla`, and new `pi0fast_env`.
- `pi0fast_env` is a lightweight Python venv at `/lambda/nfs/verl/conda/envs/pi0fast_env/` that inherits all packages from `groot_libero` via `--system-site-packages`, but overlays the custom transformers fork (`git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi`).
- `groot_libero` restored to standard `transformers==4.57.6` (was accidentally modified by an aborted pip install).
- HF token (redacted, user: yunfeixie) is logged in and has access to gated model `google/paligemma-3b-pt-224`.
- `cknna/run_all.sh` updated: per-model Python selection in Phase 2 -- pi0fast/pi05 use `PI0_PYTHON_BIN` (pi0fast_env), other models use `PYTHON_BIN` (groot_libero).
- `cknna/extract_features.py` updated: pi0fast/pi05 hook path changed from `model.paligemma_with_expert.paligemma.language_model` (GemmaModel -- bypassed by direct `.forward()` call) to `model.paligemma_with_expert.paligemma.language_model.norm` (GemmaRMSNorm -- called via `__call__`, hooks fire).
- **Prefill-only optimization (Q4)**: added `_pi0_prefill_only()` forward function. Runs only the encoding/prefill pass, skips the 256-step autoregressive decode loop. Result: ~56x faster (24 it/s vs 0.43 it/s) AND captures correct features (full condition embedding vs single decode token).
- Pi0fast test run (prefill-only): `feats_A.pt` shape=(20, 2048), CKNNA(k=10)=0.975, mutual_knn(k=10)=0.945. Previous (decode-step) values were CKNNA=0.652, mutual_knn=0.455 -- confirming the old code captured wrong features.
- **No modifications to lerobot source code** (`src/`): `git diff` and `git status` on `src/` are clean. All pi0/pi0.5 workarounds live in `cknna/` scripts only.

## Upstream version conflict (Q6 -- 2026-02-19)

The separate `pi0fast_env` is NOT caused by a bug we introduced. It is a **known, intentional upstream design limitation** in the lerobot repo:

1. `pyproject.toml` line 136: `pi = ["transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi", ...]` -- pi needs a custom unreleased fork that adds `adarms_cond` parameter support for PaliGemma/Gemma forward.
2. `pyproject.toml` line 178: `# "lerobot[pi]", TODO(Pepijn): Update pi to transformers v5` -- pi is explicitly excluded from `lerobot[all]`.
3. `pyproject.toml` lines 439-455: explicit `[tool.uv] conflicts` declarations between `pi` and every other model family (`transformers-dep`, `smolvla`, `groot`, `xvla`).
4. The fork resolves to transformers 4.53.3. Standard lerobot requires `>=4.57.1,<5.0.0`. So the fork is actually OLDER than what every other model uses.
5. The direct `.forward()` call in `modeling_pi0_fast.py` (line 262) and `modeling_pi05.py` (lines 447, 459) that bypasses PyTorch hooks is also upstream code, not our modification.

## Questions asked

- Q1: How to set up conda for all terminals
- Q2: How does run_all.sh load models
- Q3: Fix Pi0Fast gated model error and run test

## Q1: Conda setup for all terminals

### Answer (short)
Ran `/home/ubuntu/verl/conda/bin/conda init bash` which appended a conda init block to `~/.bashrc`. Both `/home/ubuntu/verl/conda` and `/lambda/nfs/verl/conda` resolve to the same NFS path.

### Details
- Available envs: `groot_libero` (Python 3.10.19), `openvla`, `pla`
- New terminals automatically source conda

## Q2: How run_all.sh loads models

### Answer (short)
Three-phase pipeline: (1) load LIBERO HDF5 demos -> images.pt, feats_B.pt, metadata.json; (2) for each model, load policy via HF `from_pretrained()`, register forward hook on VLM backbone, run transitions through model, capture hidden states -> feats_A.pt; (3) compute CKNNA between feats_A and feats_B.

### Details
- Model registry maps short names to HF model IDs: groot, pi0fast, pi05, xvla
- Phase 2 uses `extract_features.py` with per-model hook configs (hook_path, extract_fn)
- pi0fast uses `PI0FastPolicy.from_pretrained("lerobot/pi0fast-libero")` which internally loads `google/paligemma-3b-pt-224` (gated)
- Phase 1 data is shared across models; can be reused with `--reuse_data`

## Q3: Fix Pi0Fast gated model error

### Answer (short)
Three issues were identified and fixed:

1. **Gated model access**: Logged into HF with provided token. Access to `google/paligemma-3b-pt-224` confirmed.
2. **Custom transformers fork**: Pi0fast requires `transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi` which adds `transformers.models.siglip.check` module. Created new env `pi0fast_env` to avoid polluting `groot_libero`.
3. **Forward hook not firing**: The code called `language_model.forward(...)` directly (line 262 of modeling_pi0_fast.py), bypassing `__call__` and therefore PyTorch hooks. Changed hook target to `language_model.norm` (GemmaRMSNorm) which IS called via `__call__`.

### Details -- pi0fast_env creation
- `python -m venv --system-site-packages /lambda/nfs/verl/conda/envs/pi0fast_env` (inherits all groot_libero packages)
- `pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"` (shadows inherited transformers)
- groot_libero restored to `transformers==4.57.6` after accidental modification
- pi0fast_env has tokenizers 0.21.4 (satisfies fork's `<0.22` constraint); groot_libero has tokenizers 0.22.2 (needed by 4.57.6)

### Details -- forward hook fix
- Original hook path: `model.paligemma_with_expert.paligemma.language_model` (GemmaModel)
- PI0FastPaliGemma.forward() at line 262 calls `self.paligemma.language_model.forward(...)` -- direct method call bypasses `nn.Module.__call__`, so hooks never fire
- Inside GemmaModel.forward(), `self.norm(hidden_states, adarms_cond)` IS called via `__call__` (hooks fire)
- New hook path: `model.paligemma_with_expert.paligemma.language_model.norm`
- Extract function: `_extract_tuple_first` (norm returns tuple, first element is final hidden states)

### Test run results
- Run dir: `cknna/runs/20260217_193434/`
- Phase 1: 20 transitions from libero_spatial (test mode)
- Phase 2: feats_A.pt shape=(20, 2048), ~2.32s/transition
- Phase 3: CKNNA(k=10) = 0.652167, mutual_knn(k=10) = 0.455000

## Q4: Prefill-only speedup for pi0fast feature extraction

### Problem
`select_action()` runs the full autoregressive generation path: 1 prefill + 256 decode steps = 257 forward passes through GemmaModel per transition. At ~2.32s/transition for 35,828 transitions that is ~23 hours. Additionally, the hook fires 257 times and `_buf["feats"]` gets overwritten each time, so we capture the LAST decode step's (B, 1, 2048) single-token hidden state -- not the condition embedding.

### Solution
Added `_pi0_prefill_only(policy, batch)` in `extract_features.py` with a `forward_fn` key in MODEL_CONFIGS. This function:
1. Calls `policy._preprocess_images(batch)` for image prep
2. Gets language tokens/masks from batch
3. Appends BOS token
4. Calls `model.embed_prefix_fast(images, img_masks, tokens, masks)` to get prefix embeddings
5. Single forward pass: `model.paligemma_with_expert.forward(inputs_embeds=[prefix_embs, None], use_cache=False)`
6. Hook fires exactly once, capturing full (B, seq_len, 2048) condition embedding

### Results comparison (test mode, 20 transitions)
| Metric | old (select_action, decode-step capture) | new (prefill-only) |
|--------|------------------------------------------|-------------------|
| Speed  | 0.43 it/s (~2.32s/it)                    | 24 it/s (~0.04s/it) |
| CKNNA(k=10) | 0.652 | 0.975 |
| mutual_knn(k=10) | 0.455 | 0.945 |

Estimated full-run: ~25 minutes instead of ~23 hours.

### Files modified
- `cknna/run_all.sh`: Added `PI0_PYTHON_BIN`, per-model Python selection in Phase 2 loop
- `cknna/extract_features.py`: Changed pi0fast/pi05 hook_path to `...language_model.norm`, extract_fn to `_extract_tuple_first`, added `forward_fn: _pi0_prefill_only`
