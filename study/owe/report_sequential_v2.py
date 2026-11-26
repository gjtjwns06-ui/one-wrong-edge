"""Write outcome-led Korean and English reports from the completed sequential run."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .report import table
from .sequential_stage12 import ROOT


def main():
    summary=pd.read_csv(ROOT/'summary.csv')
    comparison=pd.read_csv(ROOT/'comparisons.csv')
    coverage=pd.read_csv(ROOT/'candidate_coverage_summary.csv')
    classification=pd.read_csv(ROOT/'wrongness_classification.csv')
    s1=json.loads((ROOT/'stage1_selection.json').read_text())
    s2=json.loads((ROOT/'stage2_selection.json').read_text())
    s3=json.loads((ROOT/'stage3_selection.json').read_text())
    train=json.loads((ROOT/'utility_training_summary.json').read_text())
    audit=json.loads((ROOT/'replay_audit.json').read_text())
    complete=json.loads((ROOT/'run_complete.json').read_text())
    timings=pd.read_csv(ROOT/'inference_timings.csv')
    change_counts=json.loads((ROOT/'base_ranker_change_counts.json').read_text())
    worked=json.loads((ROOT/'worked_examples.json').read_text())
    def get(c,m,k=20):return summary[(summary.condition==c)&(summary.method==m)&(summary.budget==k)].iloc[0]
    def pct(x):return f'{100*x:.2f}'
    def change(row):return f'{row.delta_vs_stage2*100:+.3f} [{row.ci_low*100:+.3f}, {row.ci_high*100:+.3f}]'
    conditions=[('clean','정상'),('random05_s17','무작위+5%'),('random20_mean','무작위+20%,3시드 평균'),
                ('random20_unseen_mean','무작위+20%,미관측2시드'),('hard20_s17','어휘유사 오류+20%'),('missing20_s17','관계20% 누락')]
    coverage_table=table(['조건','기존 후보 포함률','새 후보 포함률','새 후보 내 Top20 상한','전역 어휘256 후보'],
        [[name]+[pct(coverage[coverage.condition==c].iloc[0][col]) for col in ['old_pool_coverage','new_pool_coverage','new_pool_ceiling20','lexical256_coverage']] for c,name in conditions])
    result_table=table(['조건','기존 지역 검색','확장후 어휘 순위','증명 인용 순위','효용 ML≤3','오류 분류 일괄 필터','출처 확인'],
        [[name]+[pct(get(c,m).recall) for m in ['v1_local','stage1_text','stage2_proof','utility3','wrongness_filter','source_verified']] for c,name in conditions])
    extra_table=table(['조건','효용 ML≤1 변화(pp)','효용 ML≤3 변화(pp)','오류 분류 ML≤3 변화(pp)','오류 일괄 필터 변화(pp)'],
        [[name]+[change(get(c,m)) for m in ['utility1','utility3','wrongness3','wrongness_filter']] for c,name in conditions])
    comp_rows=[]
    for c,name in conditions:
        r=comparison[(comparison.condition==c)&(comparison.comparison=='stage2_proof minus stage1_text')].iloc[0]
        comp_rows.append([name,f'{r.difference*100:+.3f} [{r.ci_low*100:+.3f}, {r.ci_high*100:+.3f}]'])
    ranking_table=table(['조건','학습 순위 − 같은 후보 어휘 순위(pp)'],comp_rows)
    worked_table=table(['질문','정답 인용 수','기존 Recall','새 후보 포함률','새 순위 Recall'],
        [[r['qid'],r['gold_count'],pct(r['old_recall20']),pct(r['new_pool_coverage']),pct(r['proof_recall20'])] for r in worked])
    clean=get('clean','stage2_proof')
    unseen=get('random20_unseen_mean','utility3')
    if unseen.ci_low>0:
        verdict='미관측 오류 두 시드의 조건에서 효용 ML의 추가 평균 이득을 관측했다. 이 구간은 재사용 시험 자료와 고정 모델에 조건부이다.'
    elif unseen.ci_high<0:
        verdict='미관측 오류 두 시드의 조건에서 효용 ML이 추가로 회수율을 악화시켰다.'
    else:
        verdict='미관측 오류 두 시드의 조건에서는 효용 ML의 추가 이득이 명확하지 않았다.'
    filtered=get('random20_unseen_mean','wrongness_filter')
    if filtered.mean_deletions==0:verdict+=' 오류 분류 일괄 필터는 검증에서 보류가 선택되어 시험에서도 관계를 삭제하지 않았다.'
    elif filtered.ci_low>0:verdict+=' 오류 분류 일괄 필터에서는 추가 이득을 관측했다.'
    elif filtered.ci_high<0:verdict+=' 오류 분류 일괄 필터는 회수율을 악화시켰다.'
    else:verdict+=' 오류 분류 일괄 필터의 추가 이득도 명확하지 않았다.'
    deletion_rows=[]
    for c,name in [('clean','정상'),('random20_mean','무작위20%'),('random20_unseen_mean','미관측2시드'),('hard20_s17','어휘유사 오류')]:
        for m in ['utility3','wrongness3','wrongness_filter']:
            r=get(c,m)
            precision='—' if pd.isna(r.wrong_deletion_precision) else pct(r.wrong_deletion_precision)+'%'
            deletion_rows.append([name,m,f'{r.mean_deletions:.3f}',pct(r.action_rate)+'%',precision,pct(r.harmed)+'%'])
    deletion_table=table(['조건','정책','평균 삭제 수','삭제 실행 질문 비율','삭제 중 거짓 귀속 비율','회수율 악화 질문 비율'],deletion_rows)
    budget_table=table(['K','새 순위 검색 Recall','+효용 ML≤3 Recall','추가 변화(pp)','새 순위 전체 인용 회수율'],
        [[k,pct(get('random20_mean','stage2_proof',k).recall),pct(get('random20_mean','utility3',k).recall),change(get('random20_mean','utility3',k)),pct(get('random20_mean','stage2_proof',k).complete)] for k in [10,20,40]])
    policy_text='; '.join(f"{name}: {p['model']}, 임계값 {p['threshold']}, 검증 평균 변화 {p['validation_gain']*100:+.3f}pp" for name,p in s3['policies'].items())
    classification_table=table(['조건','서로 다른 관측 간선','그중 거짓 귀속','ROC-AUC','Average precision'],
        [[r.condition,r.unique_edges,r.wrong_edges,f'{r.ROC_AUC:.3f}',f'{r.average_precision:.3f}'] for r in classification.itertuples()])
    timing_base=timings.ranker_seconds.median()*1000
    timing_policy=timings.with_utility3_seconds.median()*1000
    report=f'''# 순서대로 실행한 검색 개선과 간선 제거 ML 재평가

2026-11-23 · `sequential-v2` · 실제 완료 실행에서 생성.

## 결론과 해석 범위

후보 확보 방식을 고정하고, 실제 학습 증명의 직접 인용을 정답으로 순위를 학습한 뒤, 그 검색기를 고정해 간선 제거 ML을 다시 실험했다. 정상 조건에서 새 순위 검색의 Recall@20은 **{pct(clean.recall)}%**였다. {verdict}

이 결과는 기존 시험 질문567개/25개 원문 블록을 **재사용한 후속 개발 평가**이다. 이미 실패 분석에 사용한 시험 집합이므로 새 독립 데이터에서 재현된 성능이라고 주장하지 않는다. 단계별 선택은 학습·검증에서 마친 뒤 시험에 적용했다. 최초 설정, 기존 결과·모델·보고서와 ZIP은 변경하지 않았다.

정답은 한 원문 증명의 직접 인용이다. 학습 모델은 이 인용 관계를 통계적으로 예측하며, 모든 가능한 증명에서의 필수 관계나 논리적 필요충분성을 증명하지 않는다. 자연어 답변·새 증명 생성 성공률은 측정하지 않았다.

## 1단계 — 출발점과 후보 확보

선택한 설정은 `{s1['variant']['name']}`이다. 기본 어휘 후보64개, 현재 출발 관계가 있는 시드16개와 어휘 시드8개, 과거 배경 증명의 관측 인용 수 상위32개를 조합한다. 두 단계의 가중 탐색 후 최대256개 후보를 확보한다. 전이는 최대2,048개다. 막힌 시드를 포함하더라도 기본 어휘·과거 인용 후보를 남겨 빈 확장을 보완한다.

과거 인용 수는 질문보다 앞선 **관측 배경 증명**에서 계산한다. 질문 자신의 증명, 미래 증명, 정답이나 주입 표시를 읽지 않는다. 검증·시험 증명 출발 간선과 범위 경계의 종단 처리를 그대로 유지했고 범위 밖 노드를 추가하지 않았다.

검증에서 후보 포함률은{s1['validation']['coverage']*100:.2f}%, 후보 내 Recall@20 상한은{s1['validation']['ceiling20']*100:.2f}%였다. 같은256개 한도의 전역 어휘 후보 포함률은{s1['validation']['lexical256_coverage']*100:.2f}%였다. 시험 결과는 아래와 같다. 모든 값은 질문별 평균 퍼센트다.

{coverage_table}

기존 후보보다 큰 집합을 쓰므로 기존 검색기와 계산 예산이 같지는 않다. 동일한256개 한도의 어휘 후보를 별도 비교했다. 후보만 확장하고 같은 어휘 점수로 정렬하면 상위20개는 전역 어휘 검색과 같아진다. 이를 그래프 추론 능력의 개선으로 부르지 않는다.

## 2단계 — 실제 증명 인용을 학습한 순위

학습 질문의 (질문,후보)에 원문 인용 여부를 붙였다. 질문·후보의 결론과 가정 간 유사도, 가정·변수 제약 수, 명제 역할·길이, 과거 인용 및 탐색 지지를42개 특징으로 사용했다. 개별 정리/질문 식별자와 정답·주입 표시·시험 증명은 특징에 포함하지 않았다. 모든 양성과 가까운 음성·표집 음성으로 질문 가중치를 균형화했다.

Logistic Regression과 histogram gradient boosting을 비교해 검증에서 `{s2['model']}`을 골랐다. 학습 후보쌍은{s2['training_candidate_pairs']:,}개, 그중 양성은{s2['training_positive_pairs']:,}개였다. 검증 Recall@20은{s2['validation']['recall20']*100:.2f}%였고 동일 후보의 어휘 순위20.25%, 고정 혼합 점수23.00%와 비교했다. 개별 명제 ID를 외우는 임베딩이나 논리 검증기는 사용하지 않았다.

아래는 최종 시험의 각 단계와 간선 제거 결과다. 값은 직접 인용 Recall@20의 퍼센트다.

{result_table}

후보 집합을 같게 두고 순위 모델만 바꾼 차이는 다음과 같다. 95% 구간은 원문 블록 bootstrap 2,000회이며 단위는퍼센트포인트(pp)다.

{ranking_table}

정상 조건에서 기존 지역 검색보다 좋아진 질문은{change_counts['improved']}개, 나빠진 질문은{change_counts['harmed']}개, 같은 질문은{change_counts['unchanged']}개였다. 평균 개선이 모든 질문의 개선을 뜻하지 않는다. 이전 진단 사례와 가장 큰 악화 사례를 함께 남겼다. 아래 값도퍼센트다.

{worked_table}

예를 들어 `eulerpathpr`는 원문 인용26개 중 기존0개에서 새 순위8개 회수로 바뀌었다. 반면 `elimf`는 정답2개가 모두 후보에 있었는데도 새 순위에서 둘 다 놓쳤다. 모델은 통계적으로 기록된 인용을 예측하며, 이처럼 정답 후보를 충분히 확보한 뒤에도 실패할 수 있다. [질문별 변화](runs/sequential_v2/base_ranker_query_changes.csv)와 [실제 근거 사례](runs/sequential_v2/worked_examples.json)를 보존했다.

## 3단계 — 새 검색기 위에서 간선 제거를 다시 학습

이전 검색기의 삭제 효과를 가져오지 않았다. 새 후보·특징·순위를 삭제 후 실제 재계산했다. 학습 원문 블록을3분할하고, 각 분할의 질문과 증명 출발 관계를 제외한 순위 모델로 그 질문들의 효과를 만들었다. 최종 순위 모델에 대한 학습 내 효과로 정책을 가르치는 과적합을 줄이려는 out-of-fold 절차다. OOF 모델과 전체 학습 모델의 학습 규모 차이는 남는다.

OOF 질문-조건 사례{train['OOF_query_context_cases']:,}개에서 후보 삭제 효과{train['candidate_effects']:,}개를 만들었다. 양의 효과{train['positive']:,}개, 음의 효과{train['negative']:,}개, 0효과{train['zero']:,}개였다. 관측 탐색 간선 중 최대24개를 평가하며, 정답·주입 표시로 후보를 고르지 않는다.

두 종류를 구분했다. **효용 ML**은 부호 있는 Recall@20 변화를 회귀한다. **오류 분류 ML**은 원문 증명에 없는 거짓 귀속을 분류한다. 오류 표시는 후자의 학습 정답으로만 쓰며 입력 특징에는 넣지 않는다. 최대1개/3개 삭제 또는 삭제 보류를 검증에서 선택했다. 3개 삭제 효과는 단일 효과를 더하지 않고 재검색했다.

선택 내역: {policy_text}.

단일 효과 대부분이0인 학습 분할을 확인한 뒤, 검증·시험 전에 [다수 관계 필터 비교](SEQUENTIAL_V2_FILTER_FOLLOWUP.md)를 추가했다. 같은 오류 분류기로 관측 탐색 간선 전체를 점수화하고 임계값과 최대24/64/128개 억제를 검증에서 고른다. 기존 최대1/3개 비교와 구분하며 실제 일괄 재검색 및 같은 수의 무작위 삭제와 비교한다. 이 정책은 최대24개 후보의 단일 효용 오라클보다 넓은 행동 공간과 계산량을 사용하므로 그 오라클이 필터의 상한은 아니다.

새 순위 검색을 기준으로 한 **추가 효과**는 다음과 같다.

{extra_table}

원문 출처 확인은 공개 배경 증명의 인용표로 잘못된 귀속을 전역 정리하는 더 강한 입력 접근 비교군이다. 질문 정답을 읽지는 않지만 최대1/3개 행동 예산의 ML과 동일 비용 비교는 아니다. 무작위 개수 일치 삭제·관련도 삭제·단일 효용 오라클도 [전체 표](runs/sequential_v2/summary.csv)에 포함했다. 구간은 고정된 학습·선택 모델에 조건부이고 다중 비교를 보정한 확정적 검정이 아니다.

## 실제로 무엇을 삭제했는가

효용이 좋아지는 관계 제거와 잘못된 관계 판별은 서로 다른 목표다. 효용 ML은 실제 관계를 삭제할 수도 있으므로 '거짓 관계만 고쳤다'고 부르지 않는다. 원문 귀속 기준으로 삭제한 관계와 손해를 따로 기록했다. 무작위 반복 조건의 삭제 수·정밀도는 질문 내 평균과 합계로 계산했다. 실행·개선·악화 비율은 개별 시드의 사건 여부를 먼저 계산한 뒤 질문 안에서 평균했다. 반복을 독립 질문으로 세지 않는다.

{deletion_table}

오류 분류기의 점수 자체는 아래와 같다. 동일 간선이 여러 질문에 나타나면 점수를 간선별 평균하고, 서로 다른 간선으로 기술적 AUC를 계산했다. 독립적인 간선 모집단에 대한 유의성 검정은 아니다. 정상 조건은 거짓 간선이 없어 AUC를 계산하지 않는다.

{classification_table}

무작위20%의 미관측 두 시드에서 오류 분류≤3 정책은 삭제 중 거짓 귀속 비중 약90.86%였지만 Recall@20 변화는0이었다. 같은 조건의 효용≤3 정책에서는 삭제 중 거짓 귀속 비중이 약12.10%였다. 효용 정책의 대부분 삭제는 실제 관계에 대한 질문별 억제이므로 오류 정정과 동일하게 해석하지 않는다. 어휘상 유사한 거짓 귀속에서 분류 AUC는약0.520으로, 무작위 오류보다 판별이 어려웠다.

## 예산·완전 근거·비용

K=20에서 선택한 모델을 재학습 없이 적용했다. 다음 표는 무작위20% 세 시드 평균이다. Recall과 전체 인용 회수율은퍼센트다.

{budget_table}

질문별 실제 반환 수·공백 기준 형식 토큰 수와 탐색량은 원시 표에 보존했다. 원시 표의 candidate/transition 항목은 삭제 전 기본 후보 구성의 값이며, 간선 정책의 추가 재검색까지 합친 총 탐색량은 아니다. 정책의 실제 추가 추론 비용은 별도 지연 표본으로 비교한다. 같은 출력 상한이 같은 토큰·계산 비용을 뜻하지 않는다. 토큰은 LLM 토크나이저 토큰이 아니다. 새 순위 모델과 간선 정책 모두 CPU에서 실행했고 유료 모델 API를 사용하지 않았다.

별도 추론 실행{len(timings)}개에서 순위 검색의 중앙값은{timing_base:.2f}ms, 효용≤3 정책 포함 중앙값은{timing_policy:.2f}ms였다. 이는 이미 색인된 형식 질문과 적재된 모델을 쓰며 모델/인덱스 로딩과 모든 사후 효과·오라클 계산을 제외한다. 자유로운 자연어 입력을 포함한 서비스 지연으로 해석하지 않는다. 시험 평가의 총 계산시간{complete['test_seconds']:.1f}초에는 많은 개입 정답·비교군 재검색도 들어 있으므로 추론 비용과 구분한다.

## 검증과 재현

자동 검사14개가 통과했다. 기존 배포 파일{audit['original_files_unchanged']}개가 그대로임을 확인했다. OOF{audit['OOF_fold_audits_passed']}개 분할의 질문/블록 비중첩과 증명 간선 가림을 확인했고, 실제 간선을 지운 그래프로 단일 삭제{audit['single_deletion_materialized_replays']}개와 추론/정책 출력{audit['production_inference_and_materialized_policy_replays']}개를 재실행해 일치시켰다. 예측 경로는 질문 정답을 제거한 입력으로 실행한다.

설정과 설계는 [프로토콜](SEQUENTIAL_V2_PROTOCOL.md), [실행 기록](SEQUENTIAL_V2_LOG.md), [재현 안내](README_V2.md)에 있다. 기존 시험 자료 재사용과 단일 형식 출처, 원문 블록 간 관련성, 오류 모형의 인위성, 일반 논리 보조정리까지 포함한 한 증명 인용 정답이라는 한계는 유지된다. 독립적인 신규 질문과 실제 논리적 가정 누락·답변/증명 검증에서의 재현은 아직 하지 않았다.

![세 단계와 추가 개입 효과](runs/sequential_v2/figures/sequential_stages.png)
'''
    Path('RESULTS_V2.md').write_text(report,encoding='utf-8')
    rows_en=[[name]+[pct(get(c,m).recall) for m in ['v1_local','stage1_text','stage2_proof','utility3','wrongness_filter','source_verified']] for c,name in [
        ('clean','Clean'),('random20_mean','Random20,3 seeds'),('random20_unseen_mean','Random20,unseen seeds'),('hard20_s17','Lexical-hard20'),('missing20_s17','Missing20')]]
    en_table=table(['Condition','Original local','Expanded+lexical','Learned proof-use','Utility ML≤3','Wrongness filter','Source verification'],rows_en)
    paper=f'''# One Wrong Edge after Learning Premise Retrieval

## A sequential study of candidate coverage, proof-use ranking, and additional edge-removal value

23 November 2026 · Empirical working draft0.3 · `sequential-v2`

Author metadata are inherited from the original draft and require confirmation before external submission.

## Abstract

We follow a failure analysis of a scoped mathematical dependency retriever with three sequential interventions: hybrid candidate generation, supervised proof-use ranking, and learned edge removal on the frozen improved retriever. The corpus and original block split remain fixed:3,843 core assertions,2,773 terminal boundary assertions, and567 test questions from25 source blocks. All validation/test proof-source edges remain hidden. Candidate generation combines lexical anchors, expandable seeds and historical observed proof-citation priors, with a256-candidate cap. A histogram gradient-boosting classifier learns actual stored-proof citation labels. On reused test data, clean Recall@20 changes from0.1949 to{clean.recall:.4f}. Additional utility-based edge removal on the two unseen random-corruption seeds changes Recall@20 by{unseen.delta_vs_stage2*100:+.3f}pp, with a95% source-block bootstrap interval[{unseen.ci_low*100:+.3f},{unseen.ci_high*100:+.3f}]pp. The test set had already informed the preceding failure diagnosis, so these are exploratory sequential-development results, not independent confirmation. We distinguish accurate source attribution from query retrieval utility and make no logical-necessity, answer-generation or proof-search-success claim.

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

{en_table}

The complete candidate-coverage table,paired ranking increments and additional-policy confidence intervals are generated in `RESULTS_V2.md` and machine-readable CSVs. In particular,the utility-three policy's unseen-seed difference is{unseen.delta_vs_stage2*100:+.3f}pp[{unseen.ci_low*100:+.3f},{unseen.ci_high*100:+.3f}]. We report that value independently of the larger candidate/ranking-stage gains; improving the base retriever does not itself show that learned deletion adds value.

Wrongness classification scores and source correctness of actual removed relations are evaluated separately. A policy that improves recall by deleting a source-supported relation is a utility intervention,not a correction of that source relation. Conversely,high wrong-attribution discrimination does not imply a measurable retrieval gain. Abstention selected on validation is reported as an outcome rather than silently replaced by a more active test-tuned policy.

On the two unseen random-corruption seeds, the wrongness-three policy removes false attributions with about90.86% precision among its actions, but Recall@20 remains unchanged. The utility-three policy instead removes source-supported relations in most of its actions: only about12.10% are false attributions. It is a query-utility policy rather than a pure error corrector. The hard lexical-attribution condition has unique-edge classifier ROC-AUC about0.520, showing limited transfer to that corruption distribution. The secondary bulk filter selects abstention on validation and therefore makes no test deletions.

On clean queries, the learned ranker improves{change_counts['improved']} cases, harms{change_counts['harmed']}, and leaves{change_counts['unchanged']} unchanged relative to the original local retriever. The Euler example eulerpathpr changes from0/26 to8/26 retrieved citations; elimf still misses both gold citations despite both being in its candidate pool. These concrete regressions prevent interpreting the mean gain as uniform proof-necessity recognition.

## 6. Verification,failures and cost

Fourteen automated tests verify no-gold candidate and edge feature construction,past/same-signature boundaries,OOF hiding,materialized graph deletion equivalence,and exact prediction-cache reuse. We retain the70 original delivered artifacts unchanged. Independent replays confirm{audit['single_deletion_materialized_replays']} single-edge cases and{audit['production_inference_and_materialized_policy_replays']} production-policy outputs against materialized deleted graphs;all three OOF partitions pass nonoverlap/proof-hiding audits.

The inference entry point removes query gold before execution and returns statement/hypothesis documents with source pointers. All policy failures,removed edges and lost/gained gold citations are saved. Indexed-formal-query timing samples have median{timing_base:.2f}ms for the ranker and{timing_policy:.2f}ms with utility-three deletion;they exclude loading and offline oracle/label construction. Candidate expansion and learned scoring use more computation than the original four-seed retriever. Equal output caps do not equate formal-token or compute budgets. No paid model API is used.

## 7. Limitations and conclusion

The corpus remains one formal source and a related subject slice. Query blocks can share mathematical material;the prior hidden-proof/boundary restrictions remain;synthetic false proof attributions do not cover assumption omissions or all logical errors. Our historical accessibility is not a chronological training split,and background TF-IDF indexing is transductive. The pointwise classifier learns recorded proof use,not a proof-theoretic necessity decision. Generalization to informal mathematical questions and independently annotated alternative evidence remains untested.

Most importantly,the same test set informed the preceding failure diagnosis. This run is a transparent development comparison with frozen validation choices;an independent fresh question/source evaluation is still required for strong generalization claims. Candidate/ranking improvements and additional deletion utility must be stated separately. The measured additional effect and its uncertainty,rather than a preselected success narrative,determine the edge-removal conclusion.
'''
    Path('One_Wrong_Edge_Paper_Draft_Sequential_v2.md').write_text(paper,encoding='utf-8')
    print('Wrote RESULTS_V2.md and One_Wrong_Edge_Paper_Draft_Sequential_v2.md')


if __name__=='__main__':main()
