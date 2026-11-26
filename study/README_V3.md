# 새로 실행한 restart-v3

2026-11-25. [실험 계획](RESTART_V3_PROTOCOL.md)과 [문헌 재검토](LITERATURE_V3.md)에 따라 원문부터 다시 처리한다. 기존 v1/v2 파일은 보존한다. 주 실험 범위는 관련된 집합·관계·함수와 그래프 주제이며, 전체 Metamath 라이브러리는 출처 파싱·증명 검증 용도로만 읽는다.

이번에는 기존 색인의 어떤 문서와도 canonical signature가 겹치지 않는 신규 주제582개 질문을 시험한다. 기존 주제는 순위 학습2,306개, 정책 학습735개, 검증639개로 나눈다. 신규 시험은 문자열·관계 폐포·동치관계·순서·관계/함수 후속 절이다. 정확한 서명 중복은 없지만 같은 라이브러리의 유사한 정리와 의미적 독립성까지 보장하지 않는다.

## 새 실행

데이터·선택 파일을 보존하므로 전체 반복은 별도 작업 사본에서 새 `data/processed/restart_v3`, `runs/restart_v3` 폴더로 실행한다. 이전 실행 폴더를 삭제하는 명령은 제공하지 않는다. 고정된 원문은 `data/raw/manifest.json`, 기존 범위·이전 색인 서명은 `configs/scoped_v1.json`과 `data/processed/scoped/dataset.pkl`이 필요하다.

```powershell
python -m pip install -r requirements_v2.txt
python -m owe.prepare_v3
python -m owe.restart_v3 index
python -m owe.restart_v3 candidates
python -m owe.restart_v3 rank
python -m owe.restart_v3 policy
python -m owe.restart_v3 test
python -m owe.analyze_restart_v3
python -m owe.audit_restart_v3
python -m pytest -q
python -m owe.analyze_restart_v3 --reports-only
```

검증 선택을 잠근 뒤 시험하며 완료된 선택/시험 파일을 덮어쓰지 않는다. 정책 개입 정답은 고정된 최종 순위 모델과 별도 정책 학습 질문으로 새로 만든다. 개별 질문·조건의 정답 계산은 CPU 프로세스3개에서 독립적으로 수행한다. 반복 조건과 프로세스 수를 독립 표본 수로 세지 않는다.

## 형식 질문 검색

자유로운 자연어 서비스는 아니다. 준비된 형식 정리 ID에 대해 저장된 증명 정답을 입력에서 제거하고 실행한다. 모델·문서 색인은 먼저 적재한다.

```powershell
python -m owe.restart_v3 infer --qid trrelssd --condition clean
python -m owe.restart_v3 infer --qid trrelssd --condition random20_s401 --policy utility
```

각 명제의 결론·가정·서로 달라야 하는 변수 조건과 원문 링크를 반환한다. 간선 억제의 효과를 미리 알거나 거짓 귀속 표시를 읽어서 행동하지 않는다. 시험 개입 효과/오라클 계산은 실제 추론과 분리한다.

## 파일과 해석

- `data/processed/restart_v3/`: 원문 재파싱, 정확한 분할, 원문 근거, 공개 증명에서만 정한 경계, 색인 밖 정답 분모, 해시.
- `runs/restart_v3/`: 새 TF-IDF/오류 그래프/모델, 단계별 검증 선택, 개입 효과와 질문별 순위·행동, 통계 분석, 실패 사례, 독립 감사.
- `RESULTS_V3.md`: 실제 결과와 한계.
- `One_Wrong_Edge_Paper_Draft_Restart_v3.md`: 실제 결과를 반영한 영문 논문 초안.
- `RESTART_V3_LOG.md`: 완료 작업과 판단 기록.

기존 실험과 데이터 분할·색인·가림·시험 주제가 모두 달라졌다. v2의38.02%와 v3 수치를 같은 시험의 전후 차이로 비교하지 않는다. 인용 회수, 잘못된 증명 사용 귀속 판별, 논리적 필수성, 새 증명 생성 성공을 구분한다.

## 배포 자료

[전체 실험 ZIP](One_Wrong_Edge_Restart_v3.zip), [파일별 해시](ARTIFACTS_V3.json), [배포 검사 기록](DELIVERY_CHECKS_V3.json)을 제공한다. ZIP은 이전 배포의186개 파일과 새 코드·설정·모델·개입 사례·원시 측정·분석·감사를 함께 담는다. 고정된 `set.mm` 원문과 새 `parsed.pkl`도 포함하므로 v3 원문 재처리와 전체 감사에 원문 재다운로드가 필요하지 않다. 이전 ZIP을 새 ZIP 안에 중첩하지 않는다.

완료 폴더에서 결과 분석만 다시 만들려면 `python -m owe.analyze_restart_v3`를 실행한다. 모델·정책·시험 설정은 바뀌지 않는다. 최종 감사 내용을 문서에 반영할 때는 `--reports-only`를 사용한다.
