# Platonic Representation Alignment Pipeline Analysis
- Created: 2026-02-15
- Last updated: 2026-02-15
- Tags: platonic-rep, alignment, mutual-knn, CKA, representation-similarity, vision, language, kernel-methods

## Current state
- The platonic-rep codebase measures representational alignment between vision and language models
- Data: WIT subset of Facebook PMD, hosted as `minhuh/prh` on HuggingFace (1024 or 4096 image-text pairs)
- Feature extraction: All hidden layers extracted; LLMs use avg/last pooling over tokens, ViTs use CLS token
- Feature files: `{"feats": [N, L, D], ...}` saved as `.pt` files
- Alignment: Exhaustive search over (layer_i, layer_j) pairs to find best-aligning layers
- Primary metric: mutual_knn — fraction of shared k-nearest neighbors between two models' representations
- 8 total metrics: mutual_knn, cycle_knn, lcs_knn, cka, unbiased_cka, cknna, svcca, edit_distance_knn
- Preprocessing: outlier removal (clamp at 95th percentile), L2 normalization
- KNN computed via cosine similarity (dot product after L2 norm), self excluded, top-k selected
- Pairwise alignment score matrix saved as `.npy` for all model combinations
- Supports 12 LLMs and 17 LVMs in "val" set, 8 LLMs in "test" set
- API: `platonic.Alignment` class for programmatic use, `measure_alignment.py` CLI for batch evaluation

## Questions asked
- Q1: Full pipeline trace — data loading, embedding extraction, alignment computation, orchestration

## Q1: Full pipeline trace
### Answer (short)
The pipeline has 4 stages: (1) data creation from Facebook PMD/WIT, (2) feature extraction via HF/timm models saving all-layer representations, (3) alignment scoring via exhaustive layer search + metrics like mutual KNN, (4) batch orchestration saving pairwise score matrices.

### Details
- **Data**: `data.py` creates dataset from `facebook/pmd` wit subset, saves images + JSONL metadata, uploads to HuggingFace as `minhuh/prh`
- **Features**: `extract_features.py` + `models.py` extract all hidden states; LLM pooling (avg=masked mean, last=last token); LVM pooling (cls=CLS token at each ViT block)
- **Preprocessing**: `remove_outliers(feats, q=0.95)` clamps to 95th percentile absolute value; `F.normalize(x, p=2, dim=-1)` for unit-length vectors
- **KNN**: `compute_nearest_neighbors` = dot product similarity matrix, fill diagonal with -1e8, argsort descending, take top-k
- **Mutual KNN**: Build binary N×N masks for each model's KNN graph, element-wise AND, count shared neighbors / k, average over all points
- **CKA**: K = X@X^T, L = Y@Y^T, then HSIC(K,L)/sqrt(HSIC(K,K)*HSIC(L,L)); biased HSIC = trace(KHLH), unbiased = Song et al. 2012 Eq. 5
- **Layer search**: Nested loop over all (layer_i from model_x, layer_j from model_y), keep max score
- **Orchestration**: `measure_alignment.py` loads model lists from `tasks.py`, constructs file paths via `utils.py`, runs pairwise alignment, exploits symmetry, saves `.npy` with scores + best layer indices

### Evidence / links / notes
- All 13 Python files read in full
- `metrics.py` lines 55-84: mutual_knn implementation
- `metrics.py` lines 272-285: compute_nearest_neighbors
- `metrics.py` lines 96-119: CKA implementation
- `metrics.py` lines 230-249: unbiased HSIC (Song et al. 2012)
- `measure_alignment.py` lines 34-71: compute_score with layer search
- `extract_features.py` lines 20-107: LLM feature extraction
- `extract_features.py` lines 110-169: LVM feature extraction

### Open items
- Only `minhuh/prh` dataset supported; external datasets marked as TODO
- ConvNet support requires list-of-tensors path (different hidden dims per layer)
- `from_init` mode in `load_llm` allows random-init baselines
