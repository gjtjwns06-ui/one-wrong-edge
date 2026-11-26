# Source and related-work audit

Checked 2026-11-23 against primary sources. Numerical results from other papers are not measurements of this project.

| Source | Verified relevance | Consequence for this design |
|---|---|---|
| [Metamath database repository](https://github.com/metamath/set.mm), pinned commit `7d1aa054166171211aa5c3f5e497024031bdbd27` | Formal assertions, explicit proof references, CC0-1.0 database; licence and source hashes saved in `data/raw/manifest.json`. Upstream documents multiple proof verifiers. | Exact source-attribution labels are available. Preserve essential hypotheses and distinct-variable constraints. Our local checker is additional QA, not an independent certification claim. |
| [Metamath book, Appendix B](https://us.metamath.org/downloads/metamath.pdf) | Specifies compressed-proof reference numbering and saved steps. | Decode actual references and test the extractor; do not treat a label table as an executed proof. |
| [LeanDojo, NeurIPS 2023](https://arxiv.org/abs/2306.15626) | Established large-scale formal premise retrieval with carefully defined accessible premises and generalization splits. | Premise selection itself is established. Our question is signed intervention utility; constrain accessible premises and avoid self-proof retrieval. |
| [TheoremGraph, 2026](https://arxiv.org/abs/2606.25363) | Statement-level formal/informal dependency infrastructure, retrieval with graph expansion and attention to extraction precision. | A typed mathematical graph or expansion is not itself new. We select a related, auditable subject slice rather than claiming a new general mathematical graph. |
| [PathRAG, 2025](https://arxiv.org/abs/2502.14902) | Flow-based pruning of relational paths for graph retrieval. | Graph pruning is established; our baselines are transparent local algorithms, not a claimed faithful PathRAG reproduction. |
| [GraphRAG-FI, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1293/) | Evidence filtering and integration for GraphRAG. | Filtering motivation is prior work; any empirical comparison is limited to the implementations actually run here. |
| [CatRAG, 2026](https://arxiv.org/abs/2602.01965) | Query-adaptive traversal and edge weighting; emphasizes evidence-chain completeness. | Query conditioning is prior work. Report direct-premise coverage and complete citation recovery separately; neither certifies a generated proof. |

The previous draft's other citations are not automatically inherited as verified claims. No head-to-head superiority over the named systems is claimed in this experiment. An exhaustive novelty review and an informal-mathematics benchmark remain outside the completed scoped study.
