# Sequential-v2 실행·재현 안내

기존 검색기의 후보 확보와 실제 증명 인용 순위를 개선한 뒤, 새 검색기에 간선 제거 ML이 추가로 도움이 되는지 순서대로 평가한다. 범위는 기존 집합·관계·함수·유한집합 및 그래프 경로 단원이다. 기존 `scoped-v1` 실행은 보존한다.

먼저 [고정 프로토콜](SEQUENTIAL_V2_PROTOCOL.md), [실행 기록](SEQUENTIAL_V2_LOG.md), [완료 결과](RESULTS_V2.md)를 읽는다. 기존 시험 자료를 재사용한 개발 평가이며 독립적인 신규 검증을 완료했다는 뜻이 아니다.

학습 첫 분할에서 개별 삭제 효과 대부분이0임을 확인하고, 검증·시험 전에 [다수 관계 필터 비교](SEQUENTIAL_V2_FILTER_FOLLOWUP.md)를 추가했다. 기존 최대1/3개 정책을 유지하고 같은 오류 분류기로 더 많은 관측 관계를 억제하는 보조 비교다.

## 재현

기존 `README.md`의 데이터와 `runs/scoped_v1` 산출물이 필요하다. 새 작업 사본에서 아래를 실행한다. 이미 선택·완료 파일이 있는 폴더는 덮어쓰지 않도록 보호된다.

```powershell
python -m pip install -r requirements_v2.txt
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

python -m owe.sequential_stage12 stage1
python -m owe.sequential_stage12 stage2
python -m owe.sequential_stage3 oof
python -m owe.sequential_stage3 policy
python -m owe.sequential_stage3 test
python -m owe.analyze_sequential_v2
python -m owe.audit_sequential_v2
python -m pytest -q
python -m owe.report_sequential_v2
```

OOF 학습과 삭제 효과 생성은 시간이 걸린다. 완료된 OOF 분할과 학습 모델은 재사용한다. 시험도 완료한 오류 조건을 재사용한다. 정확히 동일한 후보 특징의 점수만 재사용하며, 작은 예측 배치는 OpenMP1개 스레드로 실행한다. 모델 적합의 스레드 설정과 구분한다.

서로 독립적인 질문별 개입 효과 계산은 CPU 프로세스3개로 실행한다. 직렬 계산과 완전히 일치함을 검사하며, 프로세스 수를 실험 반복이나 독립 표본 수로 세지 않는다.

## 실제 검색 실행

이미 확보한 형식 질문 ID에 대해 정답 없이 실행할 수 있다. 자연어 입력을 받는 범용 서비스는 아니다. 원문 순서 제한을 위해 해당 질문의 위치를 함께 사용한다.

```powershell
python -m owe.sequential_infer --qid eulerpathpr --condition clean --policy none
python -m owe.sequential_infer --qid eulerpathpr --condition random20_s29 --policy utility3
```

반환값은 명제·가정 문서, 원문 링크, 실제 억제한 관계이다. 추론 함수는 질문의 `gold`를 제거하고 실행한다. 원문 정답이나 주입 표시를 예측 특징으로 전달하지 않는다.

## 주요 산출물

- `configs/sequential_v2.json`: 모든 선택 후보와 수치 설정.
- `runs/sequential_v2/stage1_selection.json`, `stage2_selection.json`, `stage3_selection.json`: 순서대로 저장한 검증 선택.
- `ranker.pkl`, `utility_models.pkl`: 실제 순위 모델과 추가 간선 ML.
- `oof*_audit.json`, `oof*_contexts.pkl`, `oof*_intervention_cases.pkl`: OOF 비중첩·간선 가림과 새로운 효과 정답.
- `results.csv`, `summary.csv`, `comparisons.csv`: 전체 정책·예산 결과와 paired 구간.
- `candidate_coverage_summary.csv`: 후보 확보의 변화와 같은256개 한도의 어휘 비교.
- `test_*_effects.csv.gz`, `test_*_rankings.jsonl.gz`: 개입별 실제 효과, 행동, 순위.
- `wrongness_classification.csv`, `failure_cases.json`: 거짓 귀속 분류와 검색 효용의 구분 및 실패.
- `test_lock.json`, `replay_audit.json`, `analysis_checks.json`: 코드·선택 고정과 독립 재실행 검사.
- `inference_timings.csv`: 사후 효과 계산을 제외한 형식 질문 추론 시간.
- `figures/`: 연구용 PNG·SVG 그림.

출처 확인은 공개 배경 증명의 명시적 인용표에 접근하는 비교군이며, 경량 ML과 같은 입력·행동 예산으로 해석하지 않는다. 회수율 개선을 생성 답변이나 새로운 증명의 정확성 개선으로 확대하지 않는다.

## 배포 사본

`One_Wrong_Edge_Sequential_v2.zip`에 코드·선택된 모델·분할·결과·재실행 자료를 함께 담았다. [배포 검증](DELIVERY_CHECKS_V2.json)과 [파일별 해시](ARTIFACTS_V2.json)를 제공한다. 원문 대용량 파일과 전체 라이브러리 파싱 캐시는 기존 배포와 같이 제외하며, 고정된 `data/raw/manifest.json`으로 다시 확보한다. 추가로 순위 모델 검증 후보의 파생 캐시 `ranker_validation_cases.pkl`만 제외했다. 최종 모델의 실행·결과 재분석·독립 개입 재실행 검사에는 이 캐시가 필요하지 않다.

전체 학습을 새로 반복하려면 배포 사본의 `runs/sequential_v2`를 보존한 채 새 작업 사본에서 별도로 실행 폴더를 마련한다. 완료된 선택 파일은 덮어쓰지 않는다. 결과 재분석은 기존 폴더에서 `python -m owe.analyze_sequential_v2`와 `python -m owe.report_sequential_v2`로 수행한다. 추가 간선 제거의 일반적인 이득이 확인되지 않았으므로 실제 검색 CLI의 기본값은 개선된 순위 검색만 사용하는 `--policy none`이다.
