# One Wrong Edge

## Learning Question-Conditioned Edge Removal for Mathematical Evidence Retrieval

Seojun Heo  
20 November 2026 · Working draft 0.1

**Affiliation:** Department of Mathematics, Sogang University.  
**Status:** Methods and evaluation-protocol draft. Empirical experiments have not been run. The author list and affiliation should be confirmed before external submission.

# Abstract

Mathematical knowledge graphs encode connections among statements, hypotheses, and proofs, but a questionable relation need not affect every question equally. Removing a relation can also discard useful evidence, even when the removal policy aims to reduce graph errors. We propose *One Wrong Edge*, a controlled study of question-conditioned edge removal for mathematical evidence retrieval. Under a fixed corpus, retriever, and retrieval budget, we define an edge’s signed removal benefit as the change in evidence-retrieval utility after deleting that edge from the observed graph. This target preserves both beneficial and harmful deletion outcomes. We propose learning the benefit from observable question, relation, provenance, and local-structure features, then selecting at most one edge to suppress for a query, with an option to leave the graph unchanged. The planned evaluation compares learned decisions with unfiltered retrieval, rule-based and question-aware filters, matched random removal, and a one-step intervention oracle. The protocol separates relation correctness from retrieval utility and groups related questions before splitting data. Two elementary analytical observations specify a locality condition and a conditional decision-error bound. This draft reports a formulation and an evaluation plan; it makes no empirical performance claim.

**Keywords:** mathematical knowledge graphs; evidence retrieval; graph retrieval-augmented generation; edge interventions; supervised learning; robustness.

# 1. Introduction

Graph-based retrieval can connect evidence that appears in different parts of a document collection. The term GraphRAG covers substantially different retrieval architectures: for example, the original graph-and-community approach targets global, query-focused summarization ([Edge et al. 2025](#ref-edge2025graphrag)), whereas later methods adapt graph traversal to individual queries ([Lau et al. 2026](#ref-lau2026catrag)). The usefulness of a connection therefore depends on the retrieval algorithm and the question, as well as on the graph itself.

Mathematical sources make this distinction particularly important. A definition, a sufficient condition, and a lemma cited in one proof express different relationships. Consider the false implication “a connected graph has an Euler trail.” For finite undirected graphs, connectedness alone is insufficient: the star with three leaves is connected and has four odd-degree vertices. A system answering a question about Euler trails could be affected by that relation, while a question about an unrelated counting identity might never encounter it. This is a mathematical illustration, not an observed retrieval result. The proposed corpus will use an open undergraduate discrete-mathematics text and its explicitly stated assumptions ([Levin n.d.](#ref-levinbook)).

Existing work already studies graph-evidence filtering, question-dependent edge weighting, graph poisoning, and the effects of corrupted triples ([Guo et al. 2025](#ref-guo2025fi); [Lau et al. 2026](#ref-lau2026catrag); [Xiao et al. 2026](#ref-xiao2026logicpoison); [Raksha and Jasmine 2026](#ref-raksha2026blast)). Consequently, neither “filtering a graph by question” nor “one wrong relation can cause damage” is the novelty claimed here. We investigate a narrower operational target: **can a lightweight predictor learn the measured consequence of a particular edge-removal action and use it to make better retrieval decisions?**

This framing also exposes a weakness in a simple error-removal objective. An incorrect edge can be irrelevant to the current question, and a correct edge can retrieve distracting material under a limited context budget. Deletion benefit and factual correctness are distinct variables. Moreover, a model trained only on the harm caused by inserting false edges has not directly learned the cost of deleting ordinary edges. We therefore construct supervision from the same action that the deployed policy will take: removing an existing edge from an observed graph.

The intended study has three components:

1.  A typed, source-linked mathematical graph and an auditable protocol for labeling question-specific removal effects.
2.  A simple predictor of signed removal benefit, evaluated through an actual deletion-or-abstention decision.
3.  A comparison protocol that controls retrieval budgets, groups dependent examples, and measures harmful deletions as well as improvements.

The formulation and analytical observations are specified in this draft. Dataset construction, model fitting, and empirical validation remain to be completed. A small initial study would establish feasibility and describe failure modes; it would not establish broad superiority over GraphRAG systems.

# 2. Related Work and Scope

**Graph construction and retrieval.** Edge et al. construct entity graphs and community summaries for corpus-level question-focused summarization ([Edge et al. 2025](#ref-edge2025graphrag)). TheoremGraph connects statement-level dependencies across informal mathematical texts and formal libraries ([Kurgan et al. 2026](#ref-kurgan2026theoremgraph)). Its discussion of retrieval also illustrates that graph expansion can have different effects across tasks. Our planned graph is a small, annotated case study. It is neither a reproduction of the community-summary architecture nor a replacement for large mathematical graph infrastructure.

**Filtering and adaptive traversal.** GraphRAG-FI uses two-stage evidence filtering and an integration mechanism that balances retrieved and intrinsic model knowledge ([Guo et al. 2025](#ref-guo2025fi)). CatRAG adapts traversal with query-aware edge weighting and evaluates recovery of complete evidence chains ([Lau et al. 2026](#ref-lau2026catrag)). INSES combines guided navigation with similarity-based expansion for noisy or sparse graphs ([Gao and Metaxas 2026](#ref-gao2026inses)). These approaches establish strong overlap with our motivation. The narrower distinction proposed here is the use of measured, signed removal effects as supervised targets.

**Graph corruption and downstream impact.** LogicPoison perturbs logical connections using type-preserving entity substitution, including question-specific reasoning connections ([Xiao et al. 2026](#ref-xiao2026logicpoison)). Raksha and Jasmine measure the spread of triple-level poisoning across questions and examine a structure-based trust score ([Raksha and Jasmine 2026](#ref-raksha2026blast)). Our primary intervention operates on a fixed graph index while keeping source passages unchanged. It is a diagnostic and retrieval-policy experiment, rather than an attack against a deployed service.

**Scope of the contribution.** The proposed study does not establish that no earlier paper has used a related intervention target. Its empirical contribution would depend on whether the annotation protocol, mathematical error analysis, and held-out evaluation add useful evidence beyond these approaches. Comparisons with simplified heuristic filters will support claims only against those filters. Claims against CatRAG, GraphRAG-FI, or INSES would require separately documented, faithful evaluations.

# 3. Mathematical Representation and Retrieval

## 3.1 A source-linked graph

Let $G=(V,E)$ be a directed, typed multigraph. A node represents a concept, a complete mathematical statement, a hypothesis, or a specified proof. A statement record retains its domain, quantifiers, hypotheses, conclusion, and source location. Relations include:

| Relation                        | Intended meaning                                                                  |
|:--------------------------------|:----------------------------------------------------------------------------------|
| $\mathrm{proof\_uses}(P,L)$     | The specified proof $P$ explicitly uses statement $L$.                            |
| $\mathrm{has\_hypothesis}(T,A)$ | Statement $T$ explicitly includes assumption $A$.                                 |
| $\mathrm{mentions}(T,C)$        | The source statement $T$ mentions concept $C$.                                    |
| $\mathrm{entails}(S,T)$         | Under the recorded domain and conditions, the complete statement $S$ entails $T$. |

A relation used in one proof does not mean that every proof needs it. A prerequisite for learning is not automatically a logical implication. If a conclusion requires $A\land B$, we preserve the conjunction in the statement; we do not encode $A\to C$ and $B\to C$ as separate sufficient conditions. Ambiguous relations will be marked unresolved and excluded from the primary verified reference graph.

Each edge stores a proposed supporting passage and its location. Factual adjudication labels are held separately from the information available to the predictor. A provenance link is a pointer to a source, not proof that the relation is supported by that source.

## 3.2 A fixed evidence-retrieval pipeline

Let $\mathcal D$ be a fixed set of passage chunks. Node-to-passage links are frozen. The primary retriever will:

1.  Select two seed nodes by cosine similarity between the question and fixed node descriptions.
2.  Traverse the declared directed relations up to $h=2$ hops, with explicit, fixed traversal conventions.
3.  Collect passages linked to reached nodes.
4.  Rank these passages by a fixed question–passage score, breaking ties by stable passage ID.
5.  Return at most $K=5$ complete passages.

The proposed initial encoder is the frozen **sentence-transformers/all-MiniLM-L6-v2** model ([Sentence Transformers n.d.](#ref-minilmcard)). Its exact revision, tokenizer, preprocessing, and chunk boundaries must be recorded in the experiment manifest. The encoder is not fine-tuned on evaluation questions. Parameter choices may be adjusted during the pilot, then locked before the held-out evaluation.

Seed selection, passage embeddings, and passage scoring remain fixed when an edge is removed. No global centrality is recomputed for ranking in the primary pipeline. Thus, an edge can change which passages are reachable, but it cannot change the raw source text or silently change the encoder. This deliberately limited retriever makes the intervention interpretable. It should be called **local graph retrieval**, not a faithful implementation of every method called GraphRAG.

The vector baseline ranks the same passage collection directly. All methods use the same output limit. An optional answer-generation evaluation will also impose a common input-token cap and use the same language model and prompt. Actual passage counts, tokens, and computation will be reported; equal output caps do not imply equal computational cost.

## 3.3 Evidence annotations and utility

For each question $q$, annotators identify acceptable evidence sets $\mathcal A(q)$. Each $A\in\mathcal A(q)$ is sufficient for the specified reference answer or named proof. Multiple sets allow genuinely different valid evidence routes; taking the union of all routes would incorrectly require every alternative proof.

If $R_K(q,G)$ denotes the retrieved passages, define:

$$
U(q,G)=
\max_{A\in\mathcal A(q)}
\frac{|R_K(q,G)\cap A|}{|A|}.
\tag{1}
$$

We also report complete-evidence retrieval:

$$
F(q,G)=
\mathbf 1\{\exists A\in\mathcal A(q): A\subseteq R_K(q,G)\}.
\tag{2}
$$

Primary questions will have at least one annotated sufficient set with $1\le |A|\le K$. A question whose required evidence exceeds the retrieval budget will be reported separately. Evidence annotations will be frozen before test evaluation.

These measures assess recovery of the annotated evidence. They do not establish that a generated proof is valid, that an unannotated answer is wrong, or that a learner understands the mathematics. Questions tied to a particular proof will state that restriction explicitly.

# 4. The Learning Target

## 4.1 Signed removal benefit

Let $G$ be the graph actually observed at query time and let $e\in E$ be an existing candidate edge. We define:

$$
B(q,e;G)=U(q,G\setminus\{e\})-U(q,G).
\tag{3}
$$

The no-operation action, denoted $\varnothing$, has $B(q,\varnothing;G)=0$.

| Value | Interpretation for this question and retriever                |
|:------|:--------------------------------------------------------------|
| $B>0$ | Removing the edge improves annotated-evidence retrieval.      |
| $B=0$ | Removal does not change this utility.                         |
| $B<0$ | Removal loses useful evidence or otherwise worsens retrieval. |

The target is computed for ordinary edges as well as injected errors. Correct edges are not automatically assigned zero. Conversely, an injected false edge is not automatically assigned positive benefit.

This is a controlled intervention effect within a specified computational pipeline. It does not identify a population-level causal effect on human learning. For a later stochastic generation metric, the corresponding target must use expected utility or paired repeated estimates; a single pair of sampled answers is not an exact effect.

## 4.2 Connection to the original insertion experiment

Let $G^\star$ be the verified reference graph and add one false edge $e\notin E^\star$ to obtain $G_e=G^\star\cup\{e\}$. Then:

$$
B(q,e;G_e)=U(q,G^\star)-U(q,G_e).
\tag{4}
$$

The original nonnegative damage label is therefore $D(q,e)=\max\{0,B(q,e;G_e)\}$ in this special case. Keeping only $D$ discards the sign and does not measure the effect of deleting other, already existing edges. Equation (3) directly matches the intended deployment action.

A clean graph is a **reference condition**, not an upper bound on retrieval quality. Under limited retrieval, a shortcut can occasionally increase evidence recall even if its relation is semantically wrong. Such cases must be recorded rather than relabeled to match an expected result.

## 4.3 One deletion and the oracle comparison

Let $\mathcal C(q,G)$ contain candidate relation edges in the query’s observed $h$-hop neighborhood. Candidates are selected without gold answers or clean-graph comparisons. The action set is:

$$
\mathcal S(q,G)=\{\varnothing\}\cup\mathcal C(q,G).
\tag{5}
$$

The one-step intervention oracle chooses the best action from exactly this set:

$$
U_{\mathrm{oracle}}(q,G)=
U(q,G)+\max_{a\in\mathcal S(q,G)} B(q,a;G).
\tag{6}
$$

It is computed only for offline evaluation and is never used as an inference feature. With deterministic retrieval and exact labels, it is an upper bound over the evaluated one-step actions, including leaving the graph unchanged. It is not an upper bound over arbitrary graph repairs, retrievers, or multi-edge policies.

Deleting an edge here means suppressing it in a query-specific working copy of the index. The underlying annotated graph and its source records remain available for later questions and auditing.

# 5. Learning and Decision Procedure

## 5.1 Constructing supervision

For each training question and observed graph, we first compute $u_0=U(q,G)$. We then remove each candidate edge separately, recompute retrieval, and store $b_e=U(q,G\setminus\{e\})-u_0$. Every label compares two runs with the same corpus, encoder, seeds, ranking rule, and budget.

We will sample both verified and corrupted edges; their measured benefits may be positive, zero, or negative. Semantic correctness and utility outcomes will be tabulated separately, and the observed category counts will be reported. The pilot may reveal too few beneficial deletions to support the proposed learning task; this will be reported rather than corrected by selecting only favorable errors. All variants of a question family remain in the same data partition.

The primary setup will use exact retrieval evaluation on a small candidate set. If candidate enumeration becomes expensive, the sampling policy and its coverage must be reported. Sampling only edges already known to be harmful would create an unrealistic decision task.

## 5.2 Observable features

The predictor receives $\phi(q,e,G)$, computed from the observed graph and source collection. It does not receive gold evidence, the reference graph, injection flags, human correctness labels, or removal outcomes.

| Feature group           | Proposed observable inputs                                                                                                    | Purpose                                                      |
|:------------------------|:------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------|
| Question and relation   | Question–relation and question–endpoint embedding similarities                                                                | Capture question relevance.                                  |
| Local structure         | Distances from fixed seeds, local endpoint degrees, available alternate routes                                                | Describe where the edge participates in retrieval.           |
| Source support          | Relation–proposed-source similarity and provenance availability                                                               | Estimate whether the source appears to support the relation. |
| Mathematical conditions | Relation type; overlap of explicitly recorded domain, quantifier, and hypothesis terms with the question and proposed support | Test whether condition-sensitive information adds value.     |

Condition features will use a fixed extraction procedure applied to both training and held-out source text. They are observable textual signals, not an oracle for logical validity. Ambiguous condition matches remain ambiguous. An ablation without these features is required before attributing any gain to mathematical structure.

Injected edges must not be recognizable simply because they lack provenance or have a special identifier format. The main corruption set will match provenance availability across correct and corrupted edges, using real but possibly non-supporting source passages. No false source sentence is added to the corpus in the primary experiment.

## 5.3 A small supervised model

The primary predictor will be regularized linear regression on the signed labels:

$$
(\widehat\beta_0,\widehat w)=
\arg\min_{\beta_0,w} \sum_i \omega_i\bigl(b_i-\beta_0-w^\top\phi_i\bigr)^2+
\lambda\|w\|_2^2.
\tag{7}
$$

An intercept is included and not penalized. Numeric features are standardized using training data only; categorical relation types are encoded with a fixed vocabulary. Predicted benefits are clipped to $[-1,1]$. Weights $\omega_i$ give each question comparable total influence despite different numbers of candidates or graph variants.

A logistic-regression comparison will predict $\mathbf 1\{B>0\}$. This tests whether recognizing possible improvement is sufficient, or whether preserving the size and sign of deletion outcomes improves the final decision. More complex models are optional follow-up experiments. Their addition is not necessary to establish whether this prediction problem is meaningful.

Regularization and the deletion threshold are selected on validation families by the resulting retrieval policy, not by test-set prediction accuracy. The planned regularization grid is $\{0.1,1,10\}$, with abstention thresholds selected from $\{0,0.05,0.10\}$ for benefit regression. Logistic thresholds are tuned on validation data separately. These are proposed pilot settings, not fitted values.

## 5.4 Deletion with abstention

At inference time, score candidates against the same observed graph. Choose:

$$
\widehat e\in\arg\max_{e\in\mathcal C(q,G)}
\widehat B(q,e;G).
\tag{8}
$$

If there are no candidates, or the largest predicted benefit is at most the frozen threshold $\tau\ge0$, leave the graph unchanged. Otherwise, remove $\widehat e$ once and rerun the fixed retriever. No gold utility is consulted by this policy.

**Algorithm 1 — Train and apply a one-step removal policy**

1.  Partition question and theorem families before creating graph variants.
2.  For every training $(q,G)$, enumerate candidates from the observed graph.
3.  Compute features before deletion, then measure each signed removal label.
4.  Fit the predictor on training families; tune it on validation families.
5.  For a held-out $(q,G)$, predict candidate benefits without gold information.
6.  Remove at most one edge above threshold, or abstain.
7.  Score the resulting evidence only after the action has been fixed.

Primary decisions are restricted to one deletion. With several deletions, the effect of the second action depends on the first; individual benefits need not add. Recomputing predictions after each deletion is a reasonable extension, but it does not inherit a global multi-step guarantee from the one-step analysis.

# 6. Two Analytical Observations

The following observations are elementary consequences of the specified model. They clarify its boundaries and are not presented as new general graph-theoretic theorems.

## 6.1 Locality of an irrelevant deletion

**Proposition 1.** Fix the seed set independently of graph edges. Let $V_h(q,G)$ be the nodes reachable in at most $h$ allowed traversal steps, and let $H=G[V_h(q,G)]$ be their induced subgraph. Suppose retrieval depends only on this local subgraph and graph-independent node attributes, with deterministic scoring and tie-breaking. If $e\notin E(H)$, removing $e$ leaves the retrieval result unchanged, so $B(q,e;G)=0$.

*Proof.* Deletion cannot create a reachable node. Every path of length at most $h$ from a seed lies entirely within $V_h(q,G)$, and all of its edges are in $H$. The deleted edge therefore lies on none of these paths. The reached nodes and their induced subgraph remain unchanged, as do their attributes, scoring, and ordering. The retrieved evidence and its utility are consequently identical. $\square$

This is a sufficient condition, not a characterization of all zero-effect edges. Global PageRank, graph-dependent seed selection, globally recomputed embeddings, or changed sampling rules would require a different argument. The condition concerns deletion from the observed graph: an added edge can make a previously unreachable region reachable.

## 6.2 Conditional one-step decision stability

**Proposition 2.** For a fixed question, graph, and candidate action set containing no operation, suppose:

$$
\max_{a\in\mathcal S(q,G)}
|\widehat B(q,a;G)-B(q,a;G)|\le\epsilon,
\tag{9}
$$

with $B(q,\varnothing;G)=\widehat B(q,\varnothing;G)=0$. If the policy maximizes predicted benefit over this action set, then its regret relative to the one-step oracle is at most $2\epsilon$.

*Proof.* Let $a^\star$ maximize true benefit and $\widehat a$ maximize predicted benefit. Then:

$$
\begin{aligned}
B(a^\star)-B(\widehat a)
&=[B(a^\star)-\widehat B(a^\star)]\\
&\quad+[\widehat B(a^\star)-\widehat B(\widehat a)]\\
&\quad+[\widehat B(\widehat a)-B(\widehat a)]\\
&\le \epsilon+0+\epsilon .
\end{aligned}
\tag{10}
$$

The middle term is nonpositive by the choice of $\widehat a$. $\square$

The proposition assumes a uniform prediction-error bound; test MAE or RMSE does not establish that bound. It applies directly to the zero-threshold policy with the no-operation action. If a valid bound $\epsilon$ were known, deleting only when $\widehat B>\epsilon$ would ensure a nonnegative true benefit for that deletion. We do not assume that validation supplies such a guarantee.

These observations neither prove generalization of the learned model nor guarantee that the proposed method outperforms a competing retriever.

# 7. Planned Experiments

## 7.1 Corpus and annotation

The first pilot will use 10–15 nodes and 10–15 questions drawn from an explicitly licensed discrete-mathematics source. A larger case study will target 30–50 nodes, 50–100 verified relations, and 50–80 questions. These are planning ranges, not completed dataset counts or a statistical power calculation.

Questions will cover definitions, named proof dependencies, and combinations of two or more evidence units. Retrieval hop count will not be equated with mathematical reasoning depth: one passage may contain an entire proof, and a multi-hop search can retrieve a simple fact.

For each question, the annotation record will contain the reference answer, acceptable evidence sets, statement or theorem family, assumptions, and source locations. A second reviewer should check the test evidence and mathematical relation labels. If only one reviewer is available, that limitation and the review procedure will be reported. Unresolved examples are excluded from the primary benchmark and counted in the dataset summary.

## 7.2 Corruption regimes

The primary experiment adds one semantically incorrect, type-compatible relation to a reference graph. The source corpus, node texts, node-to-passage links, and clean evidence annotations remain fixed. Corruptions may include an unsupported proof dependency, a reversed implication represented as an added relation, or an implication that ignores a required condition. Each example needs a documented reason why the added relation is incorrect under the chosen semantics.

All original correct edges remain in this first regime. Thus a one-edge insertion can be exactly reversed by removing the inserted edge. A literal direction replacement, missing correct edge, or modified theorem statement is a different corruption regime and will not be described as repaired by simple deletion.

Errors are proposed without inspecting held-out utility labels. Real LLM extraction mistakes, independently adjudicated, form a desirable secondary evaluation. Results on hand-constructed errors alone cannot establish robustness to natural extraction errors. Multi-error graphs will be an extension reported separately from the single-error case.

## 7.3 Partitions and dependent observations

The dataset will be partitioned by theorem or question family before paraphrasing, corruption generation, or labeling. All paraphrases, graph variants, edge rows, and repeated measurements of a family stay together. A proposed split is approximately 60% training, 20% validation, and 20% test families, subject to the actual family count.

The primary small-corpus evaluation is a **held-out-question-family study within a fixed source corpus**. Source passages may be indexed at inference time, but evaluation questions, reference evidence sets, and intervention labels cannot be used to fit or select the method. Shared generic definitions and graph entities will be reported. This design does not by itself establish transfer to a new graph.

A separate transfer experiment, if sufficient data are available, will hold out an entire chapter or independently constructed graph and exclude its supervised labels from training. If too few independent families remain, the evaluation will be reported as a pilot rather than forcing a fragile train/test claim. Thousands of edge rows derived from a few questions are not thousands of independent test questions.

## 7.4 Comparisons

| Method                           | Role in the evaluation                                                             |
|:---------------------------------|:-----------------------------------------------------------------------------------|
| Vector retrieval                 | Same source passages, without graph traversal.                                     |
| Unfiltered local graph retrieval | Observed graph, with no edge suppression.                                          |
| Provenance-rule filter           | Fixed source-support rule, tuned on validation data.                               |
| Question-aware relevance filter  | Suppresses an edge using question–relation relevance, without intervention labels. |
| Benefit regression               | Proposed signed-benefit predictor and abstention policy.                           |
| Binary-benefit classifier        | Learns whether removal helps, without modeling its magnitude.                      |
| Matched random removal           | Random candidate removal with the same deletion count as the evaluated policy.     |
| Clean-graph reference            | Uses the verified reference graph; not an oracle upper bound.                      |
| One-step intervention oracle     | Best evaluated removal or no operation, using gold utility offline.                |

The question-aware heuristic is important: comparing only with a question-independent rule would not isolate the value of the intervention target. It is not labeled a reproduction of CatRAG. Faithful comparisons with published full systems are a later requirement for broader performance claims.

Deletion policies use the same candidate set. A fixed-budget comparison requires each policy to remove exactly one edge when candidates exist. A separate abstention comparison allows zero or one removal and reports its rate. Matched random removal follows that rate on the same examples; it does not look at gold outcomes.

## 7.5 Metrics and uncertainty

The primary endpoint is the paired utility change on held-out questions:

$$
\Delta_{\pi}(q,G)=U(q,G_{\pi})-U(q,G).
\tag{11}
$$

We will report mean change, complete-evidence retrieval, harmful-deletion frequency, deletion rate, and regret against Equation (6). Prediction MAE and classification precision–recall are diagnostic metrics, not substitutes for the policy endpoint. If a classification stratum has no positive examples, its undefined statistics will be marked not applicable.

Semantic correctness and functional effect will be cross-tabulated. Removing a correct relation is not necessarily a retrieval loss, and a positive recall change is not evidence that an incorrect relation has become mathematically valid. Both outcomes must remain visible.

Every method will be evaluated on matched questions and graph realizations. Corruption variants will first be aggregated within the question; related questions will then be grouped by family. Family-level resampling may provide exploratory uncertainty intervals when enough families exist. With very few families, we will emphasize individual-family effects and descriptive uncertainty instead of presenting a precise population claim. Random seeds are recorded, but extra seeds do not create independent mathematical content.

## 7.6 Ablations and optional answer evaluation

The planned ablations remove question features, local-structure features, or mathematical-condition features while holding the prediction model and data fixed. Another comparison changes signed regression to the binary target. These tests ask whether each component supports a better decision, rather than merely increasing training fit.

Final-answer evaluation is optional and follows the retrieval study. All systems will use one frozen language model, a common prompt, and a common evidence-token cap. A reviewer will score mathematical correctness, retention of required assumptions, and support by the returned passages using a written rubric. If generation is stochastic, paired repetitions will be reported separately from deterministic retrieval effects.

# 8. Results Reporting Plan

**No empirical results are available in this draft.** Table 1 is an explicit reporting template; none of its entries represents a measurement. Observations from a pilot must be labeled as such and cannot silently become held-out test results.

| Method                           | Evidence utility | Change from unfiltered | Harmful-deletion rate |  Oracle regret  |
|:---------------------------------|:----------------:|:----------------------:|:---------------------:|:---------------:|
| Unfiltered local graph retrieval |     Not run      |       Reference        |    Not applicable     |     Not run     |
| Provenance-rule filter           |     Not run      |        Not run         |        Not run        |     Not run     |
| Question-aware relevance filter  |     Not run      |        Not run         |        Not run        |     Not run     |
| Benefit regression               |     Not run      |        Not run         |        Not run        |     Not run     |
| Binary-benefit classifier        |     Not run      |        Not run         |        Not run        |     Not run     |
| Matched random removal           |     Not run      |        Not run         |        Not run        |     Not run     |
| One-step intervention oracle     |     Not run      |        Not run         |        Not run        | 0 by definition |

*Table 1. Planned policy comparison. The oracle’s zero regret is a definition, not an experimental observation. Vector retrieval and the clean-graph reference will be reported in a separate context table because they use different indexing conditions.*

Four questions will guide interpretation:

1.  Does a given edge have different signed effects across questions?
2.  Do observable features predict those effects on held-out families?
3.  Does the resulting policy improve over both provenance and question-aware heuristics?
4.  Do mathematical-condition features provide evidence beyond generic relevance and structure?

If no candidate has positive measured benefit in most questions, the one-step oracle will reveal little opportunity for any selector. If the predictor fits labels but its deletion policy fails, the study should analyze calibration, thresholding, and negative outcomes. If rules match the learned method, model complexity is not justified by that evaluation. None of these possibilities should be hidden.

Planned figures are: an observed edge with contrasting question-specific outcomes; utility change and harmful-deletion rate by corruption regime; and benefit predictions against measured labels. All plots must use actual measurements. The mathematical illustration in the introduction is not a substitute for such evidence.

# 9. Limitations and Reproducibility

**Annotation and scope.** A small textbook graph reflects a specific mathematical presentation. A proof dependency may differ across valid proofs. Incomplete evidence annotation can penalize legitimate alternatives. Human review and explicit proof scope reduce, but do not eliminate, this problem.

**Synthetic error bias.** Type-compatible corruptions can still differ from natural extraction mistakes. Matching source availability avoids one obvious shortcut, but does not make synthetic errors representative. Natural-error evaluation must be reported separately.

**Pipeline dependence.** Benefits are defined for a frozen retriever and budget. A useful policy for two-hop candidate expansion may fail under PageRank or community summarization. Query-aware retrieval improvements are not universal statements about graph truth or formal proof.

**Single-step scope.** Other errors can mask an edge’s effect, and several deletions can interact. The one-step bound does not imply a bound for a multi-step repair policy. Missing relations and corrupted statement text require additional actions.

**Independent sample size.** The pilot will have far fewer independent mathematical questions than edge labels. Wider-domain claims require more source material, question families, and graph-level evaluations.

The reproducibility package should contain source and license records, graph schemas, reviewed questions, hidden-label separation, split manifests, all corruption seeds, encoder revisions, complete retrieval settings, candidate lists, predictions, action logs, and evaluation commands. The reference graph and answers are evaluation resources, not inference inputs. The paper should state exactly which annotations, code, and analyses were produced by the author and how AI assistance was used. No completed dataset, repository release, or experiment is claimed in this version.

# 10. Conclusion

This draft specifies a study of question-conditioned edge removal in mathematical evidence retrieval. Its central target is the signed utility change caused by removing an existing edge, which aligns supervision with the deployed action and exposes the cost of deleting useful relations. The proposed policy selects at most one removal and can abstain. Elementary locality and decision-stability observations clarify the limited setting in which the formulation can be analyzed.

The empirical question remains open: whether such labels can be predicted well enough to improve retrieval beyond simple, question-aware alternatives. Completing the annotated corpus, grouped evaluation, and failure analysis is necessary before claiming an effective method. A negative or mixed result would still clarify where graph filtering is useful and where its costs outweigh its benefits.

# Appendix A. Annotation and Run Records

Each statement record should contain an immutable ID, full text, domain, assumptions, conclusion, and source location. Each relation record should contain its direction and type, a proposed supporting passage, and a separately stored review status. A claim involving multiple assumptions must preserve all of them.

Each question record should contain its family ID, exact wording, reference answer, acceptable evidence sets, and proof scope. If adjudication changes a test annotation, record the reason, version the benchmark, and rerun all comparisons; do not change only the score of the proposed method.

An intervention log should include the question family, graph version, edge ID, unmodified retrieval IDs, removal retrieval IDs, both utilities, the signed difference, model features, predicted benefit, chosen action, and run seed. Gold labels and correctness flags must be stored outside the predictor’s inference schema.

A minimal integrity check should verify that a removal changes exactly one edge, leaves node texts and provenance passages unchanged, keeps seed IDs fixed, and preserves deterministic tie-breaking. Pipeline tests should also include a provably irrelevant deletion, a deletion of an indispensable evidence route, and a no-operation action.

# Appendix B. Multiple Deletions Are Not Additive

For a deletion set $S$, define:

$$
\Delta(S)=U(q,G\setminus S)-U(q,G).
\tag{12}
$$

In general, $\Delta(S)$ is not $\sum_{e\in S}B(q,e;G)$. Two routes may substitute for each other, so deleting either alone has zero effect while deleting both loses essential evidence. This prevents interpreting a ranking of one-edge benefits as an automatically valid multi-edge optimization rule.

If a later analysis assumes an interaction bound:

$$
\left|\Delta(S)-\sum_{e\in S}B(q,e;G)\right|
\le \rho_k
\quad\text{for all } |S|\le k,
\tag{13}
$$

then uniform prediction error $\epsilon$ and optimization of the predicted additive sum imply:

$$
\Delta(S^\star)-\Delta(\widehat S)
\le 2k\epsilon+2\rho_k.
\tag{14}
$$

Here $S^\star$ and $\widehat S$ range over the same candidate sets of size at most $k$. The bound follows by applying the interaction error and prediction error to each selected set, with the predicted-sum optimality inequality between them. Equation (13) is an additional assumption; it is not established for the proposed retriever. No multi-edge empirical claim follows from this appendix.

# References

<div id="refs" class="references csl-bib-body hanging-indent" entry-spacing="0">

<div id="ref-edge2025graphrag" class="csl-entry">

Edge, Darren, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2025. “From Local to Global: A Graph RAG Approach to Query-Focused Summarization.” arXiv preprint arXiv:2404.16130. <https://doi.org/10.48550/arXiv.2404.16130>.

</div>

<div id="ref-gao2026inses" class="csl-entry">

Gao, Hang, and Dimitris N. Metaxas. 2026. “Beyond Explicit Edges: Robust Reasoning over Noisy and Sparse Knowledge Graphs.” In *Proceedings of the 43rd International Conference on Machine Learning*, 306:33404–28. Proceedings of Machine Learning Research. PMLR. <https://proceedings.mlr.press/v306/gao26t.html>.

</div>

<div id="ref-guo2025fi" class="csl-entry">

Guo, Kai, Harry Shomer, Shenglai Zeng, Haoyu Han, Yu Wang, and Jiliang Tang. 2025. “Empowering GraphRAG with Knowledge Filtering and Integration.” In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, 25439–53. Suzhou, China: Association for Computational Linguistics. <https://doi.org/10.18653/v1/2025.emnlp-main.1293>.

</div>

<div id="ref-kurgan2026theoremgraph" class="csl-entry">

Kurgan, Simon, Evan Wang, Eric Leonen, Sophie Szeto, Luke Alexander, Artemii Remizov, Jarod Alper, Giovanni Inchiostro, and Vasily Ilin. 2026. “TheoremGraph: Bridging Formal and Informal Mathematics.” arXiv preprint arXiv:2606.25363. <https://doi.org/10.48550/arXiv.2606.25363>.

</div>

<div id="ref-lau2026catrag" class="csl-entry">

Lau, Kwun Hang, Fangyuan Zhang, Boyu Ruan, Yingli Zhou, Qintian Guo, Ruiyuan Zhang, and Xiaofang Zhou. 2026. “Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation.” arXiv preprint arXiv:2602.01965. <https://doi.org/10.48550/arXiv.2602.01965>.

</div>

<div id="ref-levinbook" class="csl-entry">

Levin, Oscar. n.d. “Discrete Mathematics: An Open Introduction.” Online textbook, 4th edition. Accessed November 20, 2026. <https://discrete.openmathbooks.org/dmoi4.html>.

</div>

<div id="ref-raksha2026blast" class="csl-entry">

Raksha, H. R. Shree, and K. S. Jasmine. 2026. “Characterizing the Blast Radius of Triple Level Poisoning Attacks in Knowledge Graph Retrieval Augmented Generation Systems.” *Discover Artificial Intelligence* 6 (November). <https://doi.org/10.1007/s44163-026-02733-8>.

</div>

<div id="ref-minilmcard" class="csl-entry">

Sentence Transformers. n.d. “<span class="nocase">all-MiniLM-L6-v2</span> Model Card.” Hugging Face model repository. Accessed November 20, 2026. <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>.

</div>

<div id="ref-xiao2026logicpoison" class="csl-entry">

Xiao, Yilin, Jin Chen, Qinggang Zhang, Yujing Zhang, Chuang Zhou, Longhao Yang, Lingfei Ren, Xin Yang, and Xiao Huang. 2026. “LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation.” In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 5575–91. San Diego, California, United States: Association for Computational Linguistics. <https://doi.org/10.18653/v1/2026.acl-long.252>.

</div>

</div>
