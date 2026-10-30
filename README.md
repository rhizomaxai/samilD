# 스팸 풍선효과 분석 — Streamlit 웹앱 (밤티팀)

## 폴더 구조
```
streamlit_app/
├── app.py              # 웹앱 (6개 탭: 문제 정의 ~ 결론, 풍선효과 검증·예측 체험 포함)
├── spam_features.py    # 공통 전처리 + '최근 값' 모델 클래스
├── train_models.py     # 모델 학습 → models/*.joblib 저장 (노트북 결과 재현)
├── requirements.txt
├── data/spam_raw.csv   # KOSIS 광고유형별 휴대전화 스팸 수신량 추이 (cp949)
└── models/
    ├── baseline.joblib  last.joblib  linear.joblib  poly2.joblib
    └── catalog.json    # 유형별 점수·2026 예측
```
각 `.joblib` 파일은 `{광고유형: 모델}` 딕셔너리이고, 입력은 반기 순서(0 = 2020 상반기)입니다.
앱에서 모델과 유형을 고르면 해당 파일을 불러와 바꿔 씁니다.

## 실행 / 배포
```bash
pip install -r requirements.txt
python train_models.py      # (선택) 모델 다시 만들기
streamlit run app.py
```
GitHub에 올린 뒤 share.streamlit.io에서 `app.py`를 지정하면 배포됩니다.
`spam_features.py`는 반드시 app.py와 같은 폴더에 있어야 joblib 모델을 불러올 수 있습니다.
