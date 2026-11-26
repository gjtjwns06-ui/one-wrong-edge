# One Wrong Edge

집합·관계·함수·유한집합과 그래프의 보행·경로·오일러 경로에 범위를 한정한, 원문 근거 기반 간선 개입 실험이다. 전체 수학 라이브러리의 성능 평가가 아니다.

이번 실행은 실제로 완료했다. 질문마다 간선 삭제 효과가 달라지는 사례를 확인했으나 학습 삭제의 일반적인 우월성이나 답변·증명 품질 개선은 확인하지 않았다. 모든 수치는 저장된 실행 결과에 근거한다.

## 먼저 읽을 파일

- [결과 및 한계](RESULTS.md)
- [시험 실행 전 고정한 설계](RESEARCH_DESIGN.md)
- [개정 영문 논문](One_Wrong_Edge_Paper_Draft_Revised.md)
- [선행연구·출처 확인](LITERATURE.md)
- [실제 출력 예산에 관한 사후 추가 분석 기록](FOLLOWUP_BUDGET_AUDIT.md)
- [진행 및 판단 기록](RUN_LOG.md)

기존 `One_Wrong_Edge_Paper_Draft_REDOWNLOAD (2).md`는 배경 문서로 보존했다. 개정 논문은 별도 파일이며 기존 문서를 덮어쓰지 않았다.

## 실행 환경

실제 실행은 Windows, Python 3.13.5, CPU 8개 논리 코어, 시스템 RAM 약 16GB에서 했다. 모델 학습과 검색은 CPU에서 수행했으며 외부 유료 모델 API를 쓰지 않았다. Python 라이브러리 버전은 `requirements.txt`에 고정했다.

원문은 [Metamath set.mm](https://github.com/metamath/set.mm)의 커밋 `7d1aa054166171211aa5c3f5e497024031bdbd27`이다. 데이터는 CC0-1.0이고 원문 라이선스를 `data/raw/LICENSE`에 보존했다. 원문 URL·커밋·SHA-256은 `data/raw/manifest.json`에 있다. `iset.mm`, `hol.mm`는 초기 접근성 조사 때 함께 확보했으며 이번 실험에 사용하지 않았다.

## 새 작업 사본에서 재현하기

이 저장소의 코드·설정·설계 문서를 새 작업 폴더에 두고 다음을 실행한다. `data/raw`는 복사하거나 첫 명령으로 재확보할 수 있다. 기존 완료 실행은 덮어쓰지 않도록 보호되므로, 재실행에는 새 작업 사본을 사용한다. 처음 설치할 때만 가상환경 안에서 의존성을 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

python -m owe.acquire
python -m owe.metamath --corpus set --verify
python -m owe.prepare
python -m pytest -q
python -m owe.experiment
python -m owe.finite_graphs
python -m owe.analyze
python -m owe.budget_audit
python -m owe.audit_run
python -m owe.report
```

`owe.acquire`는 이미 있는 원문 manifest의 커밋을 재사용한다. 원문 manifest가 없으면 고정된 날짜 상한 이전 커밋을 조회한다. 엄밀한 동일 스냅샷 재현에는 배포된 manifest도 복사한다. `owe.prepare`는 설정의 커밋과 실제 확보 커밋이 일치하지 않으면 중단한다.

`owe.experiment`는 프로토콜·입력 해시를 확인하고 학습/검증을 수행한다. 모델과 임계값을 `selection.json`으로 저장한 뒤 시험 개입 효과를 만든다. 이미 `run_complete.json`이 존재하면 결과를 덮어쓰지 않는다. 재현 비교에서 실행 시간은 장비에 따라 달라지며, 수치 비교는 고정 라이브러리와 부동소수점 오차를 고려한다.

## 코드와 주요 데이터

| 경로 | 내용 |
|---|---|
| `configs/scoped_v1.json` | 정확한 단원 허용 목록, 범위, 분할·오류·검색·학습·통계 설정 |
| `owe/metamath.py` | 필수 가정과 변수 제약을 보존하는 원문 파서·압축 증명 디코더·로컬 증명 검사기 |
| `owe/prepare.py` | 핵심/경계 분리, 질문 중복 제거, 원문 블록 분할, 숨길 증명 관계 구성 |
| `owe/retrieval.py` | 지역 검색과 정확한 2단계 단일 삭제 효과, 다중 삭제 재검색 |
| `owe/experiment.py` | 오류 그래프, 학습·검증 선택·시험 정책 실행 |
| `owe/finite_graphs.py` | 유한 단순 그래프의 오일러 조건·반례 전수 검사 |
| `owe/analyze.py` | 블록 bootstrap, 표·그림·질문별 실패 사례 |
| `owe/budget_audit.py` | 사후 문서 수·형식 토큰 상한 일치 비교 |
| `owe/audit_run.py` | 저장된 개입과 정책 출력을 별도 BFS로 재실행 |
| `owe/report.py` | 저장된 집계표에서 결과 보고서·개정 논문 생성 |
| `data/processed/scoped/nodes.jsonl` | 핵심/경계 표시, 명제·가정 텍스트, 원문 위치 |
| `data/processed/scoped/queries_and_gold.jsonl` | 질문, 분할, 원문 블록, 정답 인용. 예측 특징과 분리해서 취급 |
| `data/processed/scoped/reference_edges.csv` | 공개 배경 증명에서 읽은 검색용 실제 귀속 관계 |
| `data/processed/scoped/design_lock.json` | 평가 전 고정한 설계·입력 해시 |

`proof_uses`와 `has_essential_hypothesis`는 의미가 다르다. 탐색은 전자만 따르고 가정은 명제 문서에 보존한다. 실제 관계를 가정 없는 함의로 바꾸지 않는다. 원문 필수 프레임과 `$d` 제약은 전체 파싱 결과에도 보존된다.

## 완료 실행 산출물

기본 실행 위치는 `runs/scoped_v1/`이다.

| 파일 | 내용 |
|---|---|
| `manifest.json`, `graph_audit.json`, `run_complete.json` | 버전·코드 해시·실제 오류 수·완료 영수증 |
| `validation_grid.json`, `selection.json`, `models.pkl` | 검증 선택과 실제 학습 모델 |
| `results.csv` | 154,791개 정책×질문×조건×예산 측정행. 독립 표본 수가 아님 |
| `single_interventions.csv.gz` | 시험 단일 삭제 87,806개의 실제 부호 있는 효과와 별도 출처 정답 |
| `retrievals.jsonl.gz` | 실제 삭제 행동과 반환 명제 순위 |
| `summary.csv`, `comparisons.csv`, `topic_summary.csv` | 질문 내 시드 평균과 블록 불확실성 분석 |
| `matched_budgets.csv`, `matched_budget_summary.csv` | 사후 예산 일치 비교 |
| `failure_cases.md`, `failure_and_success_cases.json` | 서로 다른 질문의 실제 실패·성공과 잃거나 얻은 정답 인용 |
| `same_edge_opposite_effects.json` | 같은 간선이 질문에 따라 반대 부호 효과를 보인 사례 |
| `finite_graph_audit.json` | 조건 누락의 반례와 33,867개 유한모형 검사 결과 |
| `analysis_checks.json`, `replay_audit.json` | 지표·예산·출처·재검색 일치 확인 |
| `figures/` | PNG와 SVG 형식의 연구 그림 |

예측 특징에는 주입 표시·정답 인용·삭제 효과·시험 증명 간선을 넣지 않는다. 출처 확인 비교군은 공개된 배경 증명의 인용표를 쓰며, 질문 정답을 사용하는 효용 오라클과 구분한다.

## 주장 범위

평가 대상은 형식적 질문의 한 원문 증명에 대한 직접 인용 회수다. 자연어 응답 정확도, 모든 가능한 대체 증명의 근거 충분성, 새로운 증명 탐색 성공률은 측정하지 않았다. 원문 순서 제한은 발표 시점 일반화가 아니며 단일 출처의 인접 블록 간 유사성은 남을 수 있다. 이 범위 안에서 확인한 발견과 실패를 보고하며 수학 전체나 모든 GraphRAG 시스템으로 확대하지 않는다.
