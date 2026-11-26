# One Wrong Edge

## Signed Retrieval Effects in a Scoped Set-Theory and Graph-Theory Dependency Graph

Seojun Heo — author and affiliation metadata inherited from the background draft; confirmation required before external submission.

23 November 2026 · Empirical working draft 0.2 · Protocol `scoped-v1`

## Abstract

Mathematical relation correctness and query-specific retrieval utility need not coincide. We study this distinction in a deliberately bounded Metamath slice covering elementary sets, relations, functions, finite sets, and graph-theoretic walks and Eulerian paths. The slice contains 3,843 core assertions and 2,773 explicitly marked direct-reference boundary assertions. We construct 3,684 formal statement queries, hold out source blocks, hide validation and test proof references, and measure signed changes after deleting observed edges. On 567 test queries from 25 source blocks, a validation-selected policy deleting at most three edges changes direct-premise Recall@20 by +0.400 percentage points under 20% random attribution additions averaged over three seeds (95% block-bootstrap interval +0.016 to +0.802). The interval includes zero when restricted to the two unseen corruption seeds. Source verification is competitive, complete citation recovery does not improve at this budget, and gains do not transfer to K=40. Global lexical retrieval has higher recall at the same output cap but uses more output tokens; post-hoc count- and token-matched diagnostics do not establish an ordering in the random-corruption condition. We find 55 distinct edges that exhibit opposite deletion-effect signs across queries within at least one fixed graph condition. The results support a distinction between source attribution and retrieval utility, while providing limited evidence for the additional value of learned deletion. We make no answer-generation or proof-search improvement claim.

## 1. Research question and scope

The question is operational: for a fixed question, index, retriever, and output budget, does suppressing a particular observed proof-use relation improve or reduce recovery of the premises cited by a designated source proof? A correct proof citation may expose distracting documents; a falsely attributed citation may occasionally expose useful ones. We preserve positive, zero, and negative intervention outcomes.

Following a scope refinement before performance experiments, the entire mathematical library is not the evaluation domain. Core sections cover elementary set operations and pairs, binary relations and functions, finite sets, vertices and edges, undirected graphs, walks and paths, and Eulerian paths. All qualifying assertions in the specified sections are included. Out-of-scope direct citations are retained as terminal boundary documents, without recursive closure. The purpose is an auditable related-subject study, not coverage of formal mathematics as a whole.

The empirical contribution is diagnostic rather than a general GraphRAG improvement: a signed intervention dataset with explicit proof boundaries, evidence that the same source-supported edge can have opposite query effects, and a documented case where small learned gains coexist with stronger simple alternatives and important failures.

## 2. Related work

[LeanDojo](https://arxiv.org/abs/2306.15626) establishes formal premise retrieval and the importance of accessible premises and challenging splits. [TheoremGraph](https://arxiv.org/abs/2606.25363) supplies statement-level formal/informal dependency infrastructure and evaluates graph expansion. These precedents rule out novelty claims based merely on mathematical graphs or premise selection.

[PathRAG](https://arxiv.org/abs/2502.14902) studies relational-path pruning; [GraphRAG-FI](https://aclanthology.org/2025.emnlp-main.1293/) studies filtering and integration; [CatRAG](https://arxiv.org/abs/2602.01965) studies query-adaptive traversal and complete evidence recovery. Query conditioning and filtering are therefore established motivations. Our implementations are simple lexical and local-dependency retrievers, not faithful reproductions of those systems. No head-to-head superiority over them is claimed. A comprehensive novelty review remains necessary before submission.

## 3. Source-grounded data

We pin [Metamath set.mm](https://github.com/metamath/set.mm/tree/7d1aa054166171211aa5c3f5e497024031bdbd27) at commit `7d1aa054166171211aa5c3f5e497024031bdbd27` dated 22 November 2026, before the client-date cutoff. The database is distributed under CC0-1.0. The source manifest records URLs, byte lengths and SHA-256 digests. A local proof checker additionally checks 48,420 proofs in the complete source file, including substitutions, essential hypotheses, distinct-variable constraints, and final expressions; this is extraction QA, not a claim of independent certification. The upstream project documents multiple independently implemented verifiers.

We decode actually executed compressed-proof references, including mandatory-hypothesis numbering and saved subproof steps. Unused labels in a compression table do not create edges. Nodes in the retrieval graph are assertions of type `|-`; syntactic construction assertions are excluded from retrieval gold. Each statement retains its essential hypotheses and distinct-variable constraints. `proof_uses` means that this specified proof explicitly references an assertion, not that every proof requires it or that either statement alone logically implies the other. Hypothesis records are distinct from proof-use edges.

The slice has 45,213 proof-use edges before hiding validation/test proof sources and 27,946 searchable reference edges after hiding. There are 6,777 essential-hypothesis records across included nodes. Boundary premises account for 69.50% of all query gold citations. This substantial boundary fraction is part of the benchmark definition and limits interpretations about closed-domain reasoning.

A query consists of the formal conclusion, essential hypotheses, and distinct-variable constraints. Documents use the same representation; variable names are normalized by type and first occurrence. We exclude proof text and descriptive comments that may explicitly name answers. Gold is the set of distinct directly used `|-` assertions in one recorded proof, not a minimal sufficient set or all valid alternative proofs. Complete citation recovery is therefore a narrow retrieval measure, not proof validity.

We remove 81 alpha-signature duplicate questions and retain all 3,684 eligible questions with at least two direct premises. There are 2,313 training queries in 87 source blocks, 804 validation queries in 28 blocks, and 567 test queries in 25 blocks. The test set includes 200 graph queries and 367 set/relation/function queries. Eighty-seven test gold sets exceed K=20; these remain in recall evaluation and cannot achieve complete recovery at that budget.

## 4. Leakage controls

Consecutive blocks of 32 core assertions within a source section are assigned by a stable seeded hash to approximately 60/20/20 partitions. Identical normalized query signatures do not cross partitions. This controls exact query duplicates and local source grouping, but does not establish independence of neighboring blocks or novel-field generalization.

All outgoing proof-use edges of validation/test query signatures are withheld from every searchable graph, including graphs used to form training features. For each query, retrieval admits only earlier source-order assertions and excludes all assertions with the same normalized statement signature. Thus it cannot retrieve its own proof or future assertions. Source order is not publication time.

TF-IDF is fitted without labels on background statement documents excluding the query signatures. Its vocabulary statistics may include later-source text; the setup is transductive indexing, not strict historical vocabulary isolation. Feature construction receives only observed graph structure and statement text. It does not receive corruption flags, query gold citations, held-out outgoing references, or measured intervention outcomes.

## 5. Corruptions, retrieval and interventions

We evaluate a clean reference; random false-attribution additions at 5% and 20%; lexical-hard false-attribution additions at 20%; and removal of 20% of true references. The 20% random condition has seeds 17, 29 and 43. Rates are fractions of the 27,946 searchable reference edges, not estimates of real extraction noise. Realized additions are 1,397 or approximately 5,589 edges; one draw realizes 5,588 after valid-candidate rejection. Added targets come from the empirical target distribution, are earlier than their sources, match the original target's core/boundary role, and do not already occur in that source proof. Hard additions choose the most similar destination among sampled valid candidates. No special identifier, missing-provenance flag, or fabricated source sentence marks an addition.

These additions are demonstrably wrong **proof attributions**, not necessarily false mathematical implications. The missing-reference condition addresses a different failure: a deletion policy cannot restore absent relations.

Global lexical retrieval ranks eligible documents by frozen TF-IDF cosine similarity. Local graph retrieval chooses four lexical seeds, follows directed proof-use edges for two steps, and reranks reached documents by the same similarity. Ties use stable node IDs. Output caps are 10, 20 and 40; K=20 is primary. This deliberately simple local retriever does not represent all GraphRAG architectures.

For query q, observed graph G and edge e, define

`B_K(q,e;G) = Recall_K(q,G without e) - Recall_K(q,G)`.

No operation has effect zero. Candidates consist of up to 32 traversed edges with the highest query-target similarity, then up to 16 deterministic random remaining edges. Neighborhoods with at most 48 edges use all of them. The one-edge oracle maximizes measured B_20 over exactly this candidate set plus no operation; it is not a global repair oracle. Candidates and model features are selected without gold citations.

We train standardized Ridge regressors with alpha 1 and 10, and histogram gradient boosting with 160 iterations, at most 15 leaf nodes, learning rate 0.08, minimum 40 samples per leaf and L2 regularization 1. Features describe query-endpoint similarities, endpoint similarity, observed degrees and local route support, source-order availability, statement lengths and hypothesis counts. These are textual and structural signals, not a logical condition checker. An edge-feature ablation excludes explicit query-similarity and local-route features but still shares query-conditioned candidates and accessibility.

Training and validation use only clean and random20-seed17 graphs. Each query-context receives equal total training weight across its candidates. We retain 152,238 training edge effects; 95.92% are zero. Validation chooses the model and abstention threshold separately for at-most-one and at-most-three policies. Both select gradient boosting, with thresholds 0.002 and 0 respectively. The three-edge policy selects a batch using the predicted scores; final retrieval is rerun after the full batch, without summing individual effects.

Comparisons include no deletion, fixed and action-count-matched random deletion, a low-query-similarity deletion heuristic, the edge-feature ablation, global lexical retrieval, exact source-attribution verification, and the candidate one-step oracle. Source verification uses public background proof references, not the held-out query proof. It has stronger source access than the lightweight predictor and performs global preprocessing; its action budget is not a one-/three-edge per-query budget. It must not be presented as a matched-action learner comparison.

## 6. Evaluation and uncertainty

Primary results are paired changes in direct-premise Recall@20. We additionally record nDCG, complete citation recovery, harms and improvements, returned documents and formal tokens, neighborhood size, candidate coverage, and K sensitivity. There are 3,969 test graph-query cases and 154,791 policy/budget measurement rows, but only 567 distinct query identities and 25 source blocks. Noise repetitions do not multiply the question sample size.

We use 2,000 paired source-block bootstrap resamples. Repeated corruption seeds are averaged within question before pooled analysis. Intervals condition on this fitted model and fixed training set; they do not incorporate every source-selection or training uncertainty. The several condition and subgroup comparisons are descriptive and not multiplicity-adjusted confirmatory tests.

## 7. Results

### 7.1 Main output-cap comparison

Values are direct-premise Recall@20 percentages. The output cap is shared; realized document/token counts and computation are not equal.

| Condition | Local graph | Learned ≤1 | Learned ≤3 | Source verification | Global lexical |
| --- | --- | --- | --- | --- | --- |
| Clean | 19.49 | 19.67 | 19.88 | 19.49 | 23.38 |
| Random +5% | 19.23 | 19.41 | 19.69 | 19.49 | 23.38 |
| Random +20%, 3 seeds | 19.01 | 19.24 | 19.41 | 19.49 | 23.38 |
| Lexical-hard +20% | 18.31 | 18.80 | 18.75 | 19.49 | 23.38 |
| Missing 20% | 19.50 | 19.59 | 19.59 | 19.50 | 23.38 |

The learned three-edge policy changes recall by +0.385pp on clean graphs (95% interval −0.142 to +0.891), +0.400pp under the three-seed random20 mean (+0.016 to +0.802), +0.439pp under lexical-hard additions (−0.123 to +1.050), and +0.093pp under missing references (−0.234 to +0.447). Restricting random20 to previously unseen corruption seeds yields +0.297pp (−0.084 to +0.708). These results do not establish a general improvement.

For random20, learned-three minus source verification is −0.085pp (−0.502 to +0.332). Under lexical-hard additions the difference is −0.740pp (−1.379 to −0.175). Learned-one minus the edge-feature ablation in random20 is +0.105pp (−0.078 to +0.285), so the incremental learned value of explicit query features is not established.

Source verification restores the clean graph under additive corruptions, as mechanically checked. It can nevertheless reduce recall for individual queries that benefited from spurious shortcuts. Mathematical/source correctness does not make the clean retrieval condition a utility upper bound.

### 7.2 Realized-output diagnostic, added after the initial test

Global lexical retrieval returns 20 documents and approximately 847 formal tokens on average, while random20 local graph retrieval returns 13.78 documents and approximately 504 formal tokens. A same-cap comparison is consequently not a matched-output-cost comparison.

We preserve the original results and add an explicitly post-hoc diagnostic. Count-matched lexical retrieval returns as many documents as the unmodified local retriever for each query-context. Token-matched lexical retrieval greedily scans the global ranking, retaining whole documents that fit within the corresponding local formal-token cap, with the same maximum K. It changes neither models nor thresholds.

Under random20, count-matched lexical recall is 18.82%, versus local graph 19.01%; the lexical-minus-graph interval is −1.730 to +1.282pp. Token-matched lexical recall is 18.44%, with an interval also including zero for the difference. Thus the initial lexical advantage cannot be interpreted as superiority at equal output cost. These diagnostics still do not equate search or inference computation, and formal whitespace tokens are not model-tokenizer tokens.

### 7.3 Signed and query-dependent effects

We save 87,806 test candidate interventions. In random20-seed17, false-attribution candidates yield 36 positive, 9 negative and 1,915 zero deletion effects. True-attribution candidates also exhibit all three signs. These are candidate-conditional counts, not prevalence estimates for uniformly sampled graph edges.

Among 18,634 edge-condition pairs seen for at least two test queries, 101 have both a positive and negative deletion effect. Deduplicating across conditions gives 55 distinct edges. For example, deleting the clean source-supported relation `un00 → eqtrdi` changes recall by +20pp for `ima0` and approximately −4.35pp for `xpima`. This demonstrates query-dependent retrieval utility in this pipeline, not an absolute mathematical valuation of either relation.

### 7.4 Failures and budget dependence

The policy is frozen at K=20. Under random20, three-edge deletion changes recall by +0.508pp at K=10 (+0.013 to +1.172), +0.400pp at K=20, and −0.183pp at K=40 (−0.596 to +0.237). It therefore does not show budget-invariant benefit. Complete citation recovery at K=20 remains 1.35% for both baseline and learned-three retrieval.

On the clean `usgrfs` query, the policy removes supported relations, loses `ibi` from a two-citation gold set, and reduces recall by 50pp. On `pthontrlon`, deleting supported relations frees space for three of four gold citations, increasing recall by 75pp. Small gold denominators amplify these case-level changes; neither case should stand in for the aggregate result. Diverse saved failure and success records include the actual actions, lost/gained citations, and source locations.

The clean retriever has no traversable edge for 188 of 567 queries (33.16%). Terminal boundary nodes and withheld proof sources contribute to this sparsity. Deletion cannot repair an empty expansion. On nonempty neighborhoods, candidate coverage averages 83.08% for clean and 75.71% for random20-seed17. The sampled-candidate oracle is therefore limited even apart from the retriever's architecture.

Descriptively, random20 three-edge gains differ between graph queries (+0.800pp) and set/relation/function queries (+0.182pp). With 12 and 13 test blocks respectively, this is a limited within-slice analysis rather than evidence of general field transfer.

## 8. Separate finite-model audit of logical conditions

To distinguish wrong attribution from an invalid mathematical relation, we separately enumerate all 33,867 labelled simple undirected graphs on one through six vertices. A zero-edge trail is allowed. We compare a checked edge-by-edge Hierholzer witness with the non-isolated-connectivity and zero-or-two-odd-vertices criterion; there are no disagreements. An independent exhaustive trail search additionally checks all 1,099 graphs on at most five vertices.

Connectedness alone admits 14,209 counterexamples among 27,476 connected graphs. The odd-degree condition alone admits 185 counterexamples, including two disjoint triangles. Requiring connectivity of every vertex rather than the non-isolated support incorrectly rejects 3,703 valid instances under the chosen trail convention. These are exact finite-space counts, not estimates of real query frequencies, a new mathematical theorem, or a language-model reasoning benchmark. This audit is restricted to simple graphs; it does not validate every graph class in the formal corpus.

## 9. Verification, resources and reproducibility

Seven automated tests check proof-reference extraction, invalid conclusions, incomplete/out-of-scope proofs, distinct variables, exact deletion reachability, hidden-proof boundaries and Euler examples. For 250 saved test contexts, a separate BFS replays 5,591 single-edge effects and 750 selected learned-policy outputs; all match. The original protocol/data and execution-code hashes match their frozen manifests. Output caps, metric ranges, lexical invariance to graph noise, and additive source-verification restoration also pass recorded checks.

The main experiment takes 227.7 seconds on the available eight-logical-core CPU environment, with model fitting taking 3.47 seconds. We do not use a paid model API. This excludes data acquisition, source verification, implementation and follow-up analyses. Online query latency and peak RAM were not separately measured. Initial source inspection used approximately 16GB available system memory; no claim about a measured memory peak is made.

The repository preserves the pinned source manifest and licence, extraction code, explicit section allowlist, query/gold partitions, searchable graphs, models, validation selection grid, individual intervention effects, per-query rankings/actions, summary tables, figures, failure cases, and replay audit. `README.md` gives commands and locked dependency versions. The Korean results report supplies the complete condition tables and points to raw artifacts.

## 10. Limits and next research decision

This is one formal source and a related-topic slice. Exact duplicates are removed, but related neighboring source blocks, shared logical lemmas, and a large boundary-premise fraction limit independence and external validity. Formal statement queries and one-proof citation gold do not establish natural-language relevance or logical sufficiency of alternative evidence. The false-attribution stress model is controlled and not calibrated to real mathematical extraction errors. Essential hypotheses are preserved, but the learned model has no dedicated logical-condition verifier.

The source-verification baseline exploits unusually explicit formal provenance; an endpoint-only predictor has little reason to replace it when that information is available. Small partial-recall changes, unchanged complete recovery, many zero effects, and budget-dependent failures do not justify a claim of improved mathematical reasoning reliability. The post-hoc output-cost comparison is informative but cannot retroactively become a confirmatory design.

A reasonable next study would keep the same subject scope, independently annotate actual assumption-omission and relation-semantics errors, and predefine retrieval fallback or multi-relation repair with genuine evidence-sufficiency and answer/proof checks. Those experiments have not been run here. The current conclusion is narrower: source correctness and query retrieval utility differ in measurable ways, while the added practical value of learned deletion remains uncertain.
