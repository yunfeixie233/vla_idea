# Repo Restructure and Git Setup
- Created: 2026-02-18
- Last updated: 2026-02-18
- Tags: repo-structure, git, gitignore, cknna, lerobot, github, path-migration

## Current state
- Moved `/home/ubuntu/verl/cknna` -> `/home/ubuntu/verl/lerobot/cknna/`
- Moved `/home/ubuntu/verl/cknna_data` -> `/home/ubuntu/verl/lerobot/cknna_data/`
- Moved `/home/ubuntu/verl/cknna_data_test` -> `/home/ubuntu/verl/lerobot/cknna_data_test/`
- All Python sys.path inserts updated to reflect new relative locations
- All shell script absolute paths updated (`cd`, `RUN_DIR`, `FINETUNED_RUN`, `DATA_DIR`)
- `.gitignore` extended with large-file patterns
- New GitHub repo: https://github.com/yunfeixie233/lerobot (public)
- Remote origin updated and first push succeeded (`main` branch)
- Large files (`.pt`, `.mp4`, run artifacts, data dirs) are not tracked by git

## Questions asked
- Q1: Move cknna directories into lerobot and fix all paths
- Q2: Create new git repo and push with gitignore for large files

## Q1: Move cknna directories into lerobot and fix all paths

### Answer (short)
Moved all three directories; updated 2 Python path changes and 5 shell script path changes.

### Details
Path changes made:

**extract_features.py and extract_features_groot.py**
- Old: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lerobot", "src"))`
- New: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))`
- Reason: `__file__` is now inside `lerobot/cknna/`, so `..` already points to `lerobot/`; no need for the extra `lerobot` segment.

**load_libero_data.py**
- Old: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LIBERO"))`
- New: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LIBERO"))`
- Reason: LIBERO lives at `/home/ubuntu/verl/LIBERO`, which is now two levels up from `lerobot/cknna/`.

**run_all.sh and run_base_models.sh**
- `cd /home/ubuntu/verl` -> `cd /home/ubuntu/verl/lerobot`
- `RUN_DIR="/home/ubuntu/verl/cknna/runs/..."` -> `RUN_DIR="/home/ubuntu/verl/lerobot/cknna/runs/..."`

**run_groot_base.sh**
- `FINETUNED_RUN="/home/ubuntu/verl/cknna/runs/20260217_183952"` -> `/home/ubuntu/verl/lerobot/cknna/runs/20260217_183952`

**run_groot_comparison.sh**
- `DATA_DIR="/home/ubuntu/verl/cknna/runs/20260217_183952"` -> `/home/ubuntu/verl/lerobot/cknna/runs/20260217_183952`
- `cd /home/ubuntu/verl` -> `cd /home/ubuntu/verl/lerobot`

## Q2: Git setup, gitignore, and push

### Answer (short)
New repo `yunfeixie233/lerobot` created on GitHub. Gitignore extended. Pushed successfully.

### Gitignore additions (appended to existing `.gitignore`)
```
### Large binary files ###
*.mp4
*.avi
*.mov
*.webm
*.mkv

*.pt
*.pth
*.ckpt

cknna/runs/
cknna/logs/

cknna_data/
cknna_data_test/

eval_groot_*/
```

### What is ignored (not pushed)
- All video files (`*.mp4`, `*.avi`, etc.) including `eval_groot_full/videos/` and `eval_groot_test/videos/`
- All PyTorch tensor/checkpoint files (`*.pt`, `*.pth`, `*.ckpt`) -- this covers `feats_A.pt`, `feats_B.pt`, `images.pt`, `_checkpoint.pt`, and all model checkpoints
- `cknna/runs/` -- all timestamped run output directories
- `cknna/logs/` -- all timestamped log directories
- `cknna_data/` and `cknna_data_test/` -- the pre-computed data directories (contain `.pt` files)
- `eval_groot_full/` and `eval_groot_test/` -- evaluation output dirs

### What is committed
- All Python source files: `compute_cknna.py`, `extract_features.py`, `extract_features_groot.py`, `load_libero_data.py`
- All shell scripts: `run_all.sh`, `run_base_models.sh`, `run_groot_base.sh`, `run_groot_comparison.sh`
- Updated `.gitignore`
- All existing lerobot source code (the original huggingface/lerobot content)

### GitHub repo
- URL: https://github.com/yunfeixie233/lerobot
- Branch: main
- Remote: origin
