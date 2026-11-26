# One Wrong Edge after Learning Premise Retrieval

## A sequential study of candidate coverage, proof-use ranking, and additional edge-removal value

23 November 2026 · Empirical working draft0.3 · `sequential-v2`

Author metadata are inherited from the original draft and require confirmation before external submission.

## Abstract

We follow a failure analysis of a scoped mathematical dependency retriever with three sequential interventions: hybrid candidate generation, supervised proof-use ranking, and learned edge removal on the frozen improved retriever. The corpus and original block split remain fixed:3,843 core assertions,2,773 terminal boundary assertions, and567 test questions from25 source blocks. All validation/test proof-source edges remain hidden. Candidate generation combines lexical anchors, expandable seeds and historical observed proof-citation priors, with a256-candidate cap. A histogram gradient-boosting classifier learns actual stored-proof citation labels. On reused test data, clean Recall@20 changes from0.1949 to0.3802. Additional utility-based edge removal on the two unseen random-corruption seeds changes Recall@20 by+0.125pp, with a95% source-block bootstrap interval[-0.175,+0.447]pp. The test set had already informed the preceding failure diagnosis, so these are exploratory sequential-development results, not independent confirmation. We distinguish accurate source attribution from query retrieval utility and make no logical-necessity, answer-generation or proof-search-success claim.

## 1. Motivation and relation to the preceding experiment

The preceding frozen local retriever selected four TF-IDF seeds, traversed proof-use edges for two hops and reranked by the same TF-IDF score. Its clean Recall@20 was19.49%. Stage diagnosis attributed72.56% of the macro gold-citation mass to the candidate boundary and7.83% to avoidable within-pool ranking loss. Withheld proof sources and terminal boundary nodes contributed to dead-end seeds. Increasing traversal depth admitted more gold but reduced final recall under the fixed lexical score. This motivates changing both candidate acquisition and ranking before asking whether edge-removal learning adds value.

Formal premise selection and graph expansion are established tasks, including [LeanDojo](https://arxiv.org/abs/2306.15626) and [TheoremGraph](https://arxiv.org/abs/2606.25363). Query-aware graph weighting and pruning also have precedents in [CatRAG](https://arxiv.org/abs/2602.01965), [PathRAG](https://arxiv.org/abs/2502.14902) and [GraphRAG-FI](https://aclanthology.org/2025.emnlp-main.1293/). Our contribution is the controlled sequential diagnostic and remeasurement of signed intervention effects, not a claim that learned premise retrieval or query conditioning is new. We do not reproduce or claim superiority over those named systems.

## 2. Fixed data and evidence semantics

The Metamath source snapshot, subject allowlist, normalized statements, essential hypotheses, distinct-variable constraints and train/validation/test source blocks are unchanged. The slice spans elementary sets,relations,functions,finite sets,and graph walks/paths/Eulerian paths. Out-of-scope direct citation documents remain terminal boundary nodes; no recursive expansion or new source is introduced. Historical premise accessibility excludes the query assertion,all same-signature assertions,and later-source statements. Validation and test proof-source edges remain withheld in all searchable graphs.

Gold is the set of actual direct `|-` citations in a designated checked proof. A positive example means that proof uses the candidate; it does not mean every proof needs it. Alternative sufficient evidence routes are not exhaustively annotated. Formal statement retrieval, source citation recovery and successful mathematical reasoning are distinct evaluation levels. The source snapshot and licence remain pinned as in the previous draft.

## 3. Sequential protocol

All choices use the original2,313 training questions and804 validation questions. Candidate selection is frozen before ranking selection; ranking is frozen before edge-policy selection; all selections are hashed before the final test evaluation. The original test set was previously inspected during diagnosis, which limits the strength of subsequent claims even though we do not tune new hyperparameters on the new test results.

### 3.1 Candidate generation

We preserve64 lexical anchors and32 observed historical-citation anchors, select16 expandable statements from the lexical top128 plus the lexical top8 as graph seeds, and traverse directed proof-use references for two steps with weighted path support. Up to2,048 transitions are selected and the pool is capped at256 documents. Remaining slots are filled lexically. Citation priors count only observed edges whose sources precede the query; the current query proof and future proof references are never used.

Three predeclared variants are compared on validation candidate-oracle Recall@20, breaking ties by coverage and traversal cost. The selected variant has validation coverage62.24% and ceiling61.55%, versus38.29% coverage for256 global lexical candidates. This is a candidate-acquisition result, not final retrieval accuracy. Keeping the same TF-IDF reranker makes the expanded pool's top20 identical to global lexical retrieval, because those lexical anchors are retained.

### 3.2 Learned proof-use ranking

We label question-candidate pairs using training proof citations and fit standardized Logistic Regression and histogram gradient boosting. The42 observable features describe whole-statement and conclusion/hypothesis similarities, lengths and hypothesis/variable constraints, statement roles, historical graph degrees and path support. We exclude question and premise identity embeddings, gold membership, corruption flags, clean-graph comparisons and held-out proof edges.

Each training query contributes all in-pool positives,up to32 high-lexical negative examples and32 other sampled negatives. Query and positive/negative weights are balanced. The selected ranker has31 leaf nodes,180 iterations,learning rate0.08,minimum50 samples per leaf,L2 regularization1,and no early stopping. It is fitted to328,345 sampled pairs,including32,281 positives. Its validation Recall@20 is30.53%,versus20.25% for lexical ranking,13.43% for historical frequency and23.00% for a fixed lexical/frequency/support mixture. These comparisons motivate the final frozen learned ranker but do not establish logical condition checking.

### 3.3 New intervention supervision

For every evaluated edge,we rerun the hybrid candidate generator,recompute graph-dependent features and rank with the frozen proof-use model. We do not reuse the preceding retriever's intervention labels. Adaptive seeds and capped pools mean deletion can change candidate replacement,so the previous monotone candidate-set argument does not automatically apply.

Utility-model training uses three out-of-fold rankers on training source blocks. Each fold ranker excludes that fold's labeled questions and additionally masks outgoing proof references of the fold's question signatures in all its graphs. The selected ranker family/hyperparameters remain fixed. Each held-out training fold supplies its new signed Recall@20 effects. This avoids directly using in-sample ranker effects for the policy,though fold models have less training data than the final deployed ranker.

We sample up to24 observed traversal edges per query,half prioritized by predicted destination relevance and half from the remainder. We train Ridge and gradient-boosted signed-effect regressors,plus a separate false-attribution classifier. Corruption/source-support status is a training target only for that classifier. Policies choose abstention or at most one/three deletions,with thresholds selected by actual validation retrieval change. Batch effects are rerun rather than obtained by adding single-edge labels. The full selection grid and frozen choices are preserved.

After observing mostly zero effects in the first training fold, and before policy validation or the new test evaluation, we add a documented secondary classifier-based filter over all observed traversal edges. The same fitted classifier is used; thresholds and maximum24/64/128 deletions are selected on validation. A matched-count random filter controls its broader action budget. The original single-removal candidate oracle is not an upper bound for this broader filter. This amendment is preserved separately and does not replace the primary one-/three-edge comparisons.

## 4. Evaluation

Primary comparisons use paired direct-citation Recall@20. We also report candidate coverage/ceilings,nDCG,complete citation recovery,K=10/40,actual documents and formal whitespace tokens,transition counts,and harm/deletion/source-correctness rates. Random-corruption repetitions are averaged within question before source-block bootstrap intervals(2,000 resamples). These are descriptive intervals conditional on fixed training and selection,not multiplicity-adjusted or fully training-uncertainty-aware confirmatory tests.

Controls include the original local retriever,global lexical retrieval,expanded-pool lexical ranking,historical frequency,a fixed nonlearned mixture,no deletion,matched-count random deletion,low-relevance deletion,utility regression,wrongness classification,source verification,and a candidate-limited single-removal oracle. Source verification uses explicit background proof provenance and can preprocess all false attributions;it is not an action-budget-matched lightweight predictor.

## 5. Measured results

Values are direct-premise Recall@20 percentages on the reused test questions.

| Condition | Original local | Expanded+lexical | Learned proof-use | Utility ML≤3 | Wrongness filter | Source verification |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | 19.49 | 23.38 | 38.02 | 38.02 | 38.02 | 38.02 |
| Random20,3 seeds | 19.01 | 23.38 | 37.54 | 37.71 | 37.54 | 38.02 |
| Random20,unseen seeds | 19.08 | 23.38 | 37.47 | 37.60 | 37.47 | 38.02 |
| Lexical-hard20 | 18.31 | 23.38 | 37.32 | 37.45 | 37.32 | 38.02 |
| Missing20 | 19.50 | 23.38 | 38.07 | 38.02 | 38.07 | 38.07 |

The complete candidate-coverage table,paired ranking increments and additional-policy confidence intervals are generated in `RESULTS_V2.md` and machine-readable CSVs. In particular,the utility-three policy's unseen-seed difference is+0.125pp[-0.175,+0.447]. We report that value independently of the larger candidate/ranking-stage gains; improving the base retriever does not itself show that learned deletion adds value.

Wrongness classification scores and source correctness of actual removed relations are evaluated separately. A policy that improves recall by deleting a source-supported relation is a utility intervention,not a correction of that source relation. Conversely,high wrong-attribution discrimination does not imply a measurable retrieval gain. Abstention selected on validation is reported as an outcome rather than silently replaced by a more active test-tuned policy.

On the two unseen random-corruption seeds, the wrongness-three policy removes false attributions with about90.86% precision among its actions, but Recall@20 remains unchanged. The utility-three policy instead removes source-supported relations in most of its actions: only about12.10% are false attributions. It is a query-utility policy rather than a pure error corrector. The hard lexical-attribution condition has unique-edge classifier ROC-AUC about0.520, showing limited transfer to that corruption distribution. The secondary bulk filter selects abstention on validation and therefore makes no test deletions.

On clean queries, the learned ranker improves396 cases, harms51, and leaves120 unchanged relative to the original local retriever. The Euler example eulerpathpr changes from0/26 to8/26 retrieved citations; elimf still misses both gold citations despite both being in its candidate pool. These concrete regressions prevent interpreting the mean gain as uniform proof-necessity recognition.

## 6. Verification,failures and cost

Fourteen automated tests verify no-gold candidate and edge feature construction,past/same-signature boundaries,OOF hiding,materialized graph deletion equivalence,and exact prediction-cache reuse. We retain the70 original delivered artifacts unchanged. Independent replays confirm126 single-edge cases and210 production-policy outputs against materialized deleted graphs;all three OOF partitions pass nonoverlap/proof-hiding audits.

The inference entry point removes query gold before execution and returns statement/hypothesis documents with source pointers. All policy failures,removed edges and lost/gained gold citations are saved. Indexed-formal-query timing samples have median13.63ms for the ranker and22.51ms with utility-three deletion;they exclude loading and offline oracle/label construction. Candidate expansion and learned scoring use more computation than the original four-seed retriever. Equal output caps do not equate formal-token or compute budgets. No paid model API is used.

## 7. Limitations and conclusion

The corpus remains one formal source and a related subject slice. Query blocks can share mathematical material;the prior hidden-proof/boundary restrictions remain;synthetic false proof attributions do not cover assumption omissions or all logical errors. Our historical accessibility is not a chronological training split,and background TF-IDF indexing is transductive. The pointwise classifier learns recorded proof use,not a proof-theoretic necessity decision. Generalization to informal mathematical questions and independently annotated alternative evidence remains untested.

Most importantly,the same test set informed the preceding failure diagnosis. This run is a transparent development comparison with frozen validation choices;an independent fresh question/source evaluation is still required for strong generalization claims. Candidate/ranking improvements and additional deletion utility must be stated separately. The measured additional effect and its uncertainty,rather than a preselected success narrative,determine the edge-removal conclusion.
