"""Build the Korean results report and revised English draft from saved tables."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    root = Path("runs/scoped_v1")
    stats = json.loads(Path("data/processed/scoped/dataset_stats.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(root / "summary.csv")
    matched = pd.read_csv(root / "matched_budget_summary.csv")
    finite = json.loads((root / "finite_graph_audit.json").read_text(encoding="utf-8"))
    switches = json.loads((root / "query_conditioning.json").read_text(encoding="utf-8"))
    training = json.loads((root / "training_effects.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "run_complete.json").read_text(encoding="utf-8"))
    replay = json.loads((root / "replay_audit.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))

    def get(condition, method, budget=20):
        return summary[(summary.condition == condition) & (summary.method == method) & (summary.budget == budget)].iloc[0]

    def pct(x):
        return f"{100*x:.2f}"

    def effect(row):
        return f"{100*row.delta_recall:+.3f} [{100*row.delta_ci_low:+.3f}, {100*row.delta_ci_high:+.3f}]"

    condition_names = [("clean", "정상 / Clean"), ("random05_s17", "무작위 +5%"),
                       ("random20_mean", "무작위 +20%, 3시드 평균"),
                       ("hard20_s17", "어휘 유사 오류 +20%"), ("missing20_s17", "관계 20% 누락")]
    main_table = table(["조건", "지역 그래프", "학습 ≤1", "학습 ≤3", "출처 확인", "전역 어휘"],
        [[name] + [pct(get(cond, method).recall) for method in ["no_op", "learned1", "learned3", "source_verified", "lexical"]]
         for cond, name in condition_names])
    effects_table = table(["조건", "학습 ≤1의 변화와 95% 구간 (pp)", "학습 ≤3의 변화와 95% 구간 (pp)"],
        [[name, effect(get(cond, "learned1")), effect(get(cond, "learned3"))] for cond, name in condition_names]
        + [["미관측 오류 시드 29·43 평균", effect(get("random20_unseen_mean", "learned1")), effect(get("random20_unseen_mean", "learned3"))]])
    matched_rows = []
    for cond, name in condition_names:
        count = matched[(matched.condition == cond) & (matched.method == "lexical_count_matched") & (matched.budget == 20)].iloc[0]
        token = matched[(matched.condition == cond) & (matched.method == "lexical_token_matched") & (matched.budget == 20)].iloc[0]
        matched_rows.append([name, pct(get(cond, "no_op").recall), pct(count.recall), pct(token.recall)])
    matched_table = table(["조건", "지역 그래프", "문서 수 일치 어휘", "형식 토큰 상한 일치 어휘"], matched_rows)
    budget_table = table(["K", "지역 그래프 Recall (%)", "학습 ≤3 Recall (%)", "변화와 95% 구간 (pp)", "전체 인용 회수율, 기본→학습 (%)"],
        [[k, pct(get("random20_mean", "no_op", k).recall), pct(get("random20_mean", "learned3", k).recall),
          effect(get("random20_mean", "learned3", k)),
          f"{pct(get('random20_mean', 'no_op', k).complete)} → {pct(get('random20_mean', 'learned3', k).complete)}"] for k in [10, 20, 40]])
    r20 = get("random20_mean", "learned3")
    zero_fraction = training["zero"] / training["candidate_effects"]
    report = f"""# One Wrong Edge — 범위를 제한한 실험 결과

실행일: 2026-11-23. 설정: `scoped-v1`. 이 보고서는 실제 저장된 실험 표에서 생성했다.

## 확인한 결론

같은 증명 사용 관계라도 질문에 따라 삭제의 이득과 손해가 달라지는 사례를 확인했다. 하지만 학습 삭제 정책의 전반적 우월성은 확인하지 못했다. 무작위 오류 20%의 세 시드 평균에서 최대 3개 삭제는 Recall@20을 **{100*r20.delta_recall:+.3f}pp** 바꾸었다. 미관측 오류 시드 29·43에 한정하면 **{effect(get('random20_unseen_mean', 'learned3'))}pp**로 구간이 0을 포함한다. 출처 확인보다 나았다는 근거도 없으며, 출력 예산을 어떻게 맞추는지에 따라 검색 방식 비교의 해석이 달라졌다.

## 범위와 실제 데이터

사용자의 요청에 따라 전체 수학 라이브러리 성능 실험은 하지 않았다. 집합의 기본 연산·순서쌍·이항관계·함수·유한집합 및 정점·간선·무향 그래프·보행·경로·오일러 경로를 대상으로 했다. 그래프 단원은 단순 그래프뿐 아니라 원문에 등장하는 다중그래프 등의 명제도 원래 가정 그대로 포함한다. 별도 유한모형 검사는 단순 무향 그래프에만 해당한다.

- 핵심 명제 **{stats['core_nodes']:,}개**, 범위 밖 직접 인용 경계 근거 **{stats['boundary_nodes']:,}개**. 경계 노드의 증명은 확장하지 않는다.
- 숨기기 전 핵심 명제의 직접 인용 관계 **{stats['core_reference_edges_before_hiding']:,}개**. 검증·시험 증명의 출발 관계를 숨긴 뒤 검색 관계 **{stats['searchable_reference_edges']:,}개**.
- 적격 형식 질의 **{stats['query_count']:,}개**: 학습 2,313개/87블록, 검증 804개/28블록, 시험 **567개/25블록**. 시험은 그래프 200개, 집합·관계·함수 367개이다.
- 정답 인용 중 경계 근거 비중 **{100*stats['gold_boundary_fraction']:.2f}%**. 이는 그래프 경계를 무시할 수 없음을 뜻한다. 시험 87개는 직접 인용 수가 K=20보다 많아 전체 인용 회수가 구조적으로 불가능하다.
- 변수 이름을 정규화한 동일 결론·가정·변수 제약의 중복 질의 81개를 제거했다. 블록 분할은 분야 전체 미관측 평가나 발표 시점 분할이 아니다.

원문은 Metamath `set.mm`의 커밋 `{stats['source_commit']}`이고 데이터 이용 조건은 CC0-1.0이다. [원문 및 이용 조건](https://github.com/metamath/set.mm/tree/{stats['source_commit']})을 보존했다. 전체 파일은 정확한 부분집합 추출과 사전 원문 검사에 사용했다. 로컬 검사기로 전체 파일의 48,420개 증명을 추가 검사했지만 이를 전체 라이브러리 성능 실험으로 세지 않는다. 사용하지 않은 다른 데이터베이스 다운로드는 실험에 포함하지 않았다.

## 질문·오류·평가의 정확한 뜻

질문은 정리의 형식적 결론과 필수 가정·변수 제약이며, 정답은 **그 원문 증명에 실제 쓰인 직접 인용 명제**이다. 자연어 이해, 새로운 증명 생성, 최소 근거 집합을 평가한 것이 아니다. 모든 가능한 올바른 대체 증명을 정답으로 주석화하지 않았다.

주입 오류는 '이 증명이 이 정리를 사용했다'는 **거짓 귀속 관계**이다. 대상 정리 자체가 수학적으로 거짓이라는 뜻이 아니다. 원문 순서를 지키고 동일 관계형·출처 포인터 형식·목적지 핵심/경계 유형을 유지했다. 무작위 목적지는 실제 목적지 분포에서 표집하고 어휘 유사 오류는 그 후보 중 명제 유사도가 큰 것을 골랐다. 현실 오류율을 추정한 비율이 아니다.

학습·검증은 정상과 무작위20%-시드17만 사용했다. 검증에서 histogram gradient boosting과 단일 삭제 임계값 {selection['learned1']['threshold']}, 최대 3개 삭제 임계값 {selection['learned3']['threshold']}를 선택하고 저장한 뒤 시험 개입 효과를 만들었다. Ridge와 질문 관련 특징을 제거한 모델도 비교했다. 특징 제거 모델도 질문별 후보 집합과 원문 순서 제한은 공유하므로 완전한 질문 독립 파이프라인은 아니다.

평가는 같은 질문의 삭제 전후 직접 인용 Recall 차이다. 95% 구간은 원문 블록을 재표집하는 paired cluster bootstrap 2,000회이다. 반복 오류 시드는 질문 안에서 평균하고 독립 표본 수를 늘리지 않았다. 구간은 고정된 학습 자료·선택 모델에 조건부이며, 25개 블록의 선택이나 모델 재학습 불확실성 전체를 담지 않는다. 여러 조건·하위군의 구간은 기술적 비교이며 다중 비교를 보정한 확정적 검정이 아니다.

## 실제 검색 결과

모든 값은 직접 인용 Recall@20의 **퍼센트**이다. 같은 출력 상한 K를 적용했지만 실제 반환 수와 계산량은 같지 않다.

{main_table}

지역 그래프를 기준으로 한 변화이며 단위는 **퍼센트포인트(pp)**이다.

{effects_table}

무작위20%에서 최대3개 학습 삭제와 출처 확인의 차이는 −0.085pp, 95% 구간 [−0.502, +0.332]pp였다. 어휘 유사 오류에서 최대3개 학습 삭제는 출처 확인보다 −0.740pp [−1.379, −0.175]pp 낮았다. 질문 관련 특징의 추가 가치 역시 명확하지 않다: 무작위20%에서 단일 삭제 학습과 특징 제거 모델의 차이는 +0.105pp [−0.078, +0.285]pp였다.

출처 확인은 공개된 배경 증명의 명시적 인용표로 거짓 귀속을 전역 제거하는 비교군이다. 시험 질문의 정답을 쓰지는 않지만 예측 모델보다 강한 출처 접근권을 사용한다. 이처럼 명시적인 출처가 있으면 복잡한 학습 없이 오류를 확인할 수 있다는 점이 이 실험의 중요한 한계이자 발견이다. 출처를 정확히 복구한 그래프가 검색 효용의 상한은 아니다.

## 실제 출력 예산에 대한 사후 추가 분석

이 분석은 최초 시험 결과를 본 뒤 추가했고 [별도 기록](FOLLOWUP_BUDGET_AUDIT.md)에 이유와 규칙을 남겼다. 모델·분할·최초 결과는 변경하지 않았다.

무작위20%에서 전역 어휘 검색은 평균 20개/약 847개 형식 토큰을 반환했고 지역 그래프는 평균 **13.78개/504개**를 반환했다. 따라서 상한만 같은 비교를 토큰 비용까지 같은 비교로 해석할 수 없다.

{matched_table}

무작위20%에서 문서 수 일치 어휘 검색과 지역 그래프의 차이는 −0.189pp [−1.730, +1.282]pp였다. 형식 토큰 상한 일치 비교도 차이의 구간이 0을 포함했다. **전역 어휘 검색이 같은 비용에서 항상 우월하다는 주장도 하지 않는다.** 토큰은 공백으로 나눈 형식 토큰이며 LLM 토크나이저 토큰이 아니다. 문서 수·토큰을 맞춰도 실제 탐색·추론 비용은 다르다.

## 예산 변화와 실패

K=20에서 선택한 모델·정책을 재학습 없이 다른 예산에 적용했다. 아래는 무작위20% 세 시드 평균이다.

{budget_table}

K=40에서는 평균 이득이 사라지고 음수가 되었다. K=20에서도 전체 직접 인용 회수율은 기본과 학습 모두 **{pct(r20.complete)}%**였다. 부분 회수의 개선을 완전한 근거 확보나 답변·증명 품질의 개선으로 해석할 수 없다.

정상 그래프 시험 567개 중 188개(33.16%)는 시드에서 탐색할 관계가 없었다. 경계 노드를 종단으로 두고 검증·시험 증명을 숨긴 설계가 이런 희소성에 영향을 준다. 삭제 정책은 이 문제를 해결하지 못한다. 비어 있지 않은 이웃에서 평균 후보 포함률은 정상 약 83.08%, 무작위20%-시드17 약 75.71%였다. 오라클은 최대48개 후보 안에서만 정의된다.

실제 실패 예로 `usgrfs` 질의의 정상 그래프에서 최대3개 정책은 정상 관계들을 삭제해 정답 인용 2개 중 `ibi`를 잃었다. Recall@20 변화는 −50pp였다. 반대로 `pthontrlon`에서는 정상 관계 삭제 후 인용 4개 중 3개를 추가 회수해 +75pp였다. 극단적인 변화에는 작은 정답 집합 분모가 작용한다. [서로 다른 질문별 실제 실패·성공 사례](runs/scoped_v1/failure_cases.md)와 전체 검색 결과를 저장했다.

## 정확성과 효용을 분리한 관찰

시험의 단일 삭제 **87,806개**를 저장했다. 이는 후보 선택 규칙 안에서 측정한 효과이며 전체 그래프의 균등 표본이 아니다. 무작위20%-시드17에서 거짓 귀속 후보 1,960개 중 삭제로 좋아진 것은 36개, 나빠진 것은 9개, 변하지 않은 것은 1,915개였다. 정상 귀속 후보에서도 개선·악화·무변화가 모두 나왔다.

같은 오류 조건 안에서 두 질문 이상에 나타난 간선-조건 조합 {switches['reused_edge_condition_pairs']:,}개 중 **{switches['both_positive_and_negative']}개 조합**은 양·음 효과를 모두 보였다. 조건을 합쳐 중복 제거하면 **55개 간선**이다. 예를 들어 정상 관계 `un00 → eqtrdi` 삭제는 `ima0` 질문에서 +20pp, `xpima`에서는 약 −4.35pp였다. [원시 사례](runs/scoped_v1/same_edge_opposite_effects.json)를 확인할 수 있다. 이는 이 검색기와 원문 인용 정답에 대한 관찰이며 간선의 절대적인 수학적 중요도는 아니다.

학습 효과 {training['candidate_effects']:,}개 중 **{100*zero_fraction:.2f}%가 0**이었다. 0 예측이 쉬운 불균형 문제이므로 평균 예측 오차만으로 성공을 주장하지 않았다.

## 오일러 조건의 별도 의미 검증

정점 1~6개의 표지 단순 무향 그래프 **{finite['total_graphs']:,}개**를 전수 검사했다. 간선이 없는 그래프에는 길이 0 트레일을 허용했다. 비고립 정점의 연결성과 홀수 차수 정점 수 0 또는 2라는 판정은 실제 간선을 한 번씩 쓰는 경로 검사와 모두 일치했다. n≤5의 1,099개에는 독립적인 완전 탐색도 대조했다.

- '연결이면 오일러 트레일이 있다'는 조건을 만족하는 27,476개 중 **14,209개가 반례**였다. 별 K1,3이 한 예다.
- '홀수 차수 정점이 0 또는 2개면 충분하다'는 주장에는 **185개 반례**가 있었다. 분리된 두 삼각형이 한 예다.
- 모든 정점의 연결성을 필요조건으로 요구하면 고립 정점 때문에 유효한 트레일 **3,703개를 배제**했다.

이 수치는 이 열거 공간의 정확한 개수이며 현실 질문의 오류 확률이 아니다. 유한모형 검사와 귀속 오류 검색 실험은 별개다. 루프·다중간선·무한 그래프나 생성형 답변 정확도로 확대하지 않는다. [검사 결과와 반례](runs/scoped_v1/finite_graph_audit.json)를 보존했다.

## 재현성과 완료 상태

주 실험은 약 **{runtime['seconds']:.1f}초**, 모델 적합은 약 {runtime['model_fit_seconds']:.2f}초였다. CPU 실행이며 별도 유료 모델 API를 쓰지 않았다. 이는 원문 확보·코드 작성·추가 분석을 모두 포함한 총 연구 시간이 아니다. 온라인 질의 지연과 최대 메모리는 별도 측정하지 않았다.

주 실험 측정행은 **{runtime['result_rows']:,}개**지만 서로 다른 질문은 567개이다. 7개 오류 조건·13개 정책·3개 예산의 측정행을 독립 실험 반복으로 세지 않는다. 자동 검사 7개가 통과했다. 저장된 250개 그래프-질문 사례에서 단일 삭제 **{replay['single_edge_interventions_replayed_with_BFS']:,}개**와 학습 정책 출력 **{replay['learned_policy_outputs_replayed_with_BFS']}개**를 별도 BFS로 재실행해 일치시켰다. 원래 프로토콜과 실행 코드 해시도 일치했다.

코드·원문 커밋/해시·질문과 정답·분할·모델·검증 선택표·개입별 결과·표·그림·실패 사례·개정 논문을 보존했다. 실행법은 [README](README.md), 사전 설계는 [연구 설계](RESEARCH_DESIGN.md), 선행연구와 기여 경계는 [문헌 검토](LITERATURE.md)에 있다.

이 실행에서 완료하지 않은 것은 자연어 질문의 독립 주석, 모든 대체 증명 근거의 주석, 생성 답변 평가, 새 증명 탐색 및 형식 검증 성공률 평가, 다른 데이터 출처로의 외적 검증이다. 다음 연구는 이 범위 안에서 가정 누락·논리적 관계 오류를 독립 검증 가능한 질문으로 주석화하고, 그래프가 부족한 경우의 검색 보완 정책을 새 프로토콜로 평가하는 것이 타당하다. 현재 결과로 해당 단계의 성공을 예측하지 않는다.
"""
    Path("RESULTS.md").write_text(report, encoding="utf-8")

    english_rows = [[name] + [pct(get(cond, method).recall) for method in ["no_op", "learned1", "learned3", "source_verified", "lexical"]]
                    for cond, name in [("clean", "Clean"), ("random05_s17", "Random +5%"),
                                       ("random20_mean", "Random +20%, 3 seeds"), ("hard20_s17", "Lexical-hard +20%"), ("missing20_s17", "Missing 20%")]]
    english_table = table(["Condition", "Local graph", "Learned ≤1", "Learned ≤3", "Source verification", "Global lexical"], english_rows)
    paper = f"""# One Wrong Edge

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

We pin [Metamath set.mm]({data_source_url(stats)}) at commit `{stats['source_commit']}` dated 22 November 2026, before the client-date cutoff. The database is distributed under CC0-1.0. The source manifest records URLs, byte lengths and SHA-256 digests. A local proof checker additionally checks 48,420 proofs in the complete source file, including substitutions, essential hypotheses, distinct-variable constraints, and final expressions; this is extraction QA, not a claim of independent certification. The upstream project documents multiple independently implemented verifiers.

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

{english_table}

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
"""
    Path("One_Wrong_Edge_Paper_Draft_Revised.md").write_text(paper, encoding="utf-8")
    print("Wrote RESULTS.md and One_Wrong_Edge_Paper_Draft_Revised.md from measured tables")


def data_source_url(stats):
    return "https://github.com/metamath/set.mm/tree/" + stats["source_commit"]


if __name__ == "__main__":
    main()
