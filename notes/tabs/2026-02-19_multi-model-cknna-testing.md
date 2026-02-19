# Multi-Model CKNNA Testing: All LeRobot Models on LIBERO

- Created: 2026-02-19
- Last updated: 2026-02-19 (round 2 -- hook audit, SmolVLA finetuned, VQ-BeT issue search)
- Tags: CKNNA, multi-model, ACT, Diffusion, VQ-BeT, SAC, HIL-SERL, TDMPC, Pi0.5, SmolVLA, LIBERO, feature-extraction, hook-design, IL, RL, VLA, hook-audit

## Current state

- All 10 supported models work with the CKNNA pipeline (feats A extraction + CKNNA computation).
- Previously tested: Pi0Fast, GR00T N1.5, XVLA.
- Newly tested and passing: Pi0.5, SmolVLA, ACT, Diffusion, VQ-BeT, SAC (HIL-SERL), TDMPC.
- QC-FQL is "coming soon" in lerobot and was not tested.
- SmolVLA now uses the official LIBERO-finetuned checkpoint `HuggingFaceVLA/smolvla_libero` (resolves 2 of 3 issues; hook bypass fix still needed).
- Libero simulation eval is blocked by MuJoCo/osmesa rendering on all conda envs (system issue, not model issue).
- IL/RL models have no LIBERO-finetuned checkpoints; CKNNA scores use out-of-domain or random weights.
- No LIBERO-finetuned SAC or TDMPC checkpoints exist anywhere (HuggingFace Hub, GitHub issues, lerobot docs).
- VQ-BeT `mlp_hidden_dim` config error: no matching GitHub issue found; appears to be a silent breaking change in lerobot. Workaround in place.
- Full hook audit completed (Q3): all 10 hooks verified correct for CKNNA semantics.

## Questions asked

- Q1: Can all models run CKNNA feature extraction on LIBERO data?
- Q2: Can all models run Libero simulation eval?
- Q3: Hook audit -- what is hooked, where, and does it match CKNNA expectations?
- Q4: SmolVLA LIBERO-finetuned checkpoint (`HuggingFaceVLA/smolvla_libero`)
- Q5: VQ-BeT config error -- is it a known issue in lerobot?

---

## Q3: Hook audit -- what is hooked, where, and does it match CKNNA?

### CKNNA expectations

feat A = the "condition embedding" that feeds the action decoder. For VLA models, this is the VLM output (vision+language tokens, mean-pooled). For IL/RL models, this is the pure visual encoder output before state fusion. CKNNA compares the vision kernel K (from feat A) against the proprioception kernel L (from feat B = 8D proprioceptive state).

### VLA models

| Model | Hook Path | Module Class | What Is Captured | Semantic Role | Correct? |
|-------|-----------|--------------|------------------|---------------|----------|
| GR00T N1.5 | `_groot_model.action_head.vl_self_attention` | `SelfAttentionTransformer` | Eagle 2.5 features after eagle_linear (2048->1536) + LayerNorm + SelfAttention. Token sequence (B, N, 1536). | Condition embedding the DiT cross-attends to. Matches RS-CL paper's "adapter output f_phi." | Yes |
| Pi0Fast | `model.paligemma_with_expert.paligemma.language_model.norm` | `GemmaRMSNorm` | Final RMSNorm output of PaliGemma language model. Token sequence (B, N, 2048). | VLM hidden states that feed the expert/action branch for flow-matching denoising. | Yes |
| Pi0.5 | `model.paligemma_with_expert.paligemma.language_model.norm` | `GemmaRMSNorm` | Same as Pi0Fast. Token sequence (B, N, 2048). | Same semantic role. Pi0.5 shares the VLM backbone; only the expert/action head differs. | Yes |
| SmolVLA | `model.vlm_with_expert.vlm.model.text_model.norm` | `Idefics3RMSNorm` | Final RMSNorm of SmolVLM text model. Token sequence (B, N, 960). | VLM branch output after all interleaved attention layers. Expert cross-attends to this. | Yes |
| XVLA | `model.vlm.language_model.model.encoder` | `BartEncoder` (Florence2) | Full encoder output, `BaseModelOutput.last_hidden_state`. Token sequence (B, N, D). | Florence2 encoder output that feeds the SoftPromptedTransformer action head. | Yes |

### IL models (no language)

| Model | Hook Path | Module Class | What Is Captured | Semantic Role | Correct? |
|-------|-----------|--------------|------------------|---------------|----------|
| ACT | `model.backbone` | `IntermediateLayerGetter(ResNet)` | ResNet layer4 feature map dict `{"feature_map": (B, 512, H, W)}`, global avg pooled to (B, 512). | Pure visual features before Transformer encoder mixes with state tokens. | Yes |
| Diffusion | `diffusion.rgb_encoder` | `DiffusionRgbEncoder` | (B, 64) after ResNet -> SpatialSoftmax -> Linear -> ReLU. | Visual condition vector for the UNet denoiser. | Yes |
| VQ-BeT | `vqbet.rgb_encoder` | `VQBeTRgbEncoder` | (B, 64), same architecture as Diffusion encoder. | Visual feature vector that feeds the GPT action predictor. | Yes |

### RL models (no language)

| Model | Hook Path | Module Class | What Is Captured | Semantic Role | Correct? |
|-------|-----------|--------------|------------------|---------------|----------|
| SAC (HIL-SERL) | `actor.encoder.image_encoder` | `DefaultImageEncoder` (4-layer CNN) | CNN feature maps, global avg pooled if 4D, to (B, 32). | Visual encoding before concatenation with state for actor/critic MLPs. | Yes |
| TDMPC | `model._encoder.image_enc_layers` | `nn.Sequential` (4 Conv2d + Flatten + Linear + LN + Sigmoid) | (B, 50) visual latent. | Image encoding before averaging with state encoding in `TDMPCObservationEncoder.forward()`. | Yes |

### Cross-category comparison caveat

VLA hooks capture vision+language (task instruction is part of the token sequence). IL/RL hooks capture vision only (no language). This means CKNNA scores are not directly comparable across VLA and IL/RL categories. VLA feat A encodes richer information, so VLA CKNNA scores may be higher simply because the representation has more capacity, not necessarily because the visual features are more proprioceptively aligned.

---

## Q4: SmolVLA LIBERO-finetuned checkpoint

### Finding

`HuggingFaceVLA/smolvla_libero` is the official LeRobot team's LIBERO-finetuned SmolVLA (by jadechoghari, HF Staff). Base model: `lerobot/smolvla_base`. Finetuned on `HuggingFaceVLA/libero`.

Config shows LIBERO-compatible keys:
- `observation.images.image` at (3, 256, 256)
- `observation.images.image2` at (3, 256, 256)
- `observation.state` shape (8,)
- `action` shape (7,)
- Has `policy_preprocessor.json` with normalizer stats

### Impact on the three SmolVLA issues

| Issue | With `lerobot/smolvla_base` | With `HuggingFaceVLA/smolvla_libero` |
|-------|----------------------------|--------------------------------------|
| (a) Image key mismatch | Yes -- config has camera1/2/3 | **Resolved** -- config has image/image2 |
| (b) No preprocessor on Hub | Yes -- no LIBERO preprocessor | **Resolved** -- has policy_preprocessor.json |
| (c) Hook bypass (.forward vs .__call__) | Yes -- code-level issue | **Still occurs** -- our hook fix handles it |

### Decision

Changed `run_all.sh` to use `HuggingFaceVLA/smolvla_libero` instead of `lerobot/smolvla_base`. Updated `extract_features.py` so the config override only triggers when the model uses non-LIBERO image keys. Verified the finetuned model loads with the Hub preprocessor (no programmatic build needed).

---

## Q5: VQ-BeT config error -- is it a known lerobot issue?

### GitHub issue search results

Searched lerobot GitHub issues for "vqbet", "mlp_hidden_dim", "config deserialization", "DecodingError". **No matching issue found.** The VQ-BeT-related issues that exist are:
- #2829: "Error building my own policy" (general, not VQ-BeT-specific config loading)
- #2221: "Question about pre-trained weights usability" (discusses that base models need finetuning, not config errors)
- Various PRs for processor pipeline, torch.compile, mypy -- none mention `mlp_hidden_dim`

### Root cause analysis

- The field `mlp_hidden_dim` existed in an older version of `VQBeTConfig` but was removed in the current codebase.
- The `lerobot/vqbet_pusht` checkpoint on HuggingFace was saved with the older config and was never re-uploaded.
- The config parser (`draccus`) is strict about unknown fields and raises `DecodingError`.
- This is NOT a `transformers` version issue. VQ-BeT does not depend on `transformers` at all (uses vanilla ResNet + minGPT).
- This is a lerobot-internal backward-compatibility break. It silently broke when the config field was removed without updating the published checkpoint.

### Status

- Not reported on GitHub as of 2026-02-19.
- Our workaround (`_load_with_config_fix()` strips `mlp_hidden_dim` before loading) is correct and complete.
- Anyone loading `lerobot/vqbet_pusht` with current lerobot code will hit this error.

---

## Q1: CKNNA feature extraction -- all models

### Answer (short)
All 10 models pass. Five required code fixes.

### Fixes applied

#### Pi0.5: new prefill function
- Pi0Fast's `_pi0_prefill_only` crashed: `'PI05Pytorch' has no attribute '_paligemma_tokenizer'`.
- Pi0.5 API differs: no BOS token, uses `embed_prefix` (not `embed_prefix_fast`), uses `make_att_2d_masks`.
- Solution: wrote `_pi05_prefill_only()` mirroring Pi0.5's own `sample_actions` prefill path (lines 806-819 of modeling_pi05.py).
- Verified: our function is a line-for-line match of the model's prefill, minus `use_cache=True` (irrelevant for hook capture).

#### SmolVLA: three problems (two resolved by using finetuned checkpoint)
1. **Image key mismatch**: Resolved by using `HuggingFaceVLA/smolvla_libero` (LIBERO-finetuned, correct keys).
2. **No LIBERO preprocessor**: Resolved by using finetuned checkpoint (has `policy_preprocessor.json`).
3. **Hook bypass**: Still occurs. SmolVLA calls `.forward()` directly instead of `.__call__()`. Our fix: hook `vlm.model.text_model.norm` (called via `__call__`) + `_smolvla_prefill_only()`.

#### VQ-BeT: config field removed
- `mlp_hidden_dim` removed from current `VQBeTConfig` but present in saved checkpoint.
- Not a known/reported issue on GitHub.
- Fix: `_load_with_config_fix()` strips unknown keys before loading.

#### SAC + TDMPC: no pretrained checkpoints
- No LIBERO-finetuned SAC or TDMPC checkpoints exist anywhere.
- Fix: `--model_path from_config` creates models with LIBERO-compatible config + random weights.

#### TDMPC: forward path key mismatch
- TDMPC uses `OBS_IMAGE = "observation.images"` in `predict_action_chunk` but `next(iter(config.image_features))` = `"observation.images.image"` inside encoder. Conflicting key conventions.
- Fix: `_tdmpc_encode_only()` bypasses the key-mapping chain, calls CNN directly.

---

## Q2: Libero simulation eval

Blocked by MuJoCo/osmesa rendering dependencies on all conda envs. System issue, not model issue. The `lerobot-eval` command parses and loads correctly; only simulation instantiation crashes. The GR00T N1.6 evaluation used a separate LIBERO simulation venv (built by `setup_libero.sh`).

---

## Test results (20-transition test, libero_spatial)

| Model | Category | feat A dim | CKNNA (k=10) | mutual_knn (k=10) | Hook Path | Weights |
|-------|----------|-----------|-------------|-------------------|-----------|---------|
| Pi0.5 | VLA | 2048 | 1.017 | 0.855 | `model.paligemma_with_expert.paligemma.language_model.norm` | LIBERO-finetuned |
| SmolVLA | VLA | 960 | 1.026 | 0.890 | `model.vlm_with_expert.vlm.model.text_model.norm` | LIBERO-finetuned (`HuggingFaceVLA/smolvla_libero`) |
| ACT | IL | 512 | 1.065 | 0.890 | `model.backbone` | aloha domain (wrong domain) |
| Diffusion | IL | 64 | 0.813 | 0.510 | `diffusion.rgb_encoder` | pusht domain (wrong domain) |
| VQ-BeT | IL | 64 | 0.989 | 0.540 | `vqbet.rgb_encoder` | pusht domain (wrong domain) |
| SAC | RL | 32 | 0.831 | 0.565 | `actor.encoder.image_encoder` | Random weights |
| TDMPC | RL | 50 | 0.717 | 0.535 | `model._encoder.image_enc_layers` | Random weights |

Note: CKNNA values are not comparable across models here because weights differ (LIBERO-finetuned vs wrong-domain vs random). The test validates the pipeline, not the representation quality.

---

## Files changed

| File | Change |
|------|--------|
| `cknna/extract_features.py` | Added all 10 models; SmolVLA config override now conditional; uses Hub preprocessor for finetuned model |
| `cknna/run_all.sh` | SmolVLA uses `HuggingFaceVLA/smolvla_libero`; pi0fast/pi05 use `PI0_PYTHON_BIN` |
