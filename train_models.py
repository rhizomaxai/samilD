"""
모델 학습 & joblib 저장 스크립트 (밤티팀 노트북 재현 + 확장)
실행:  python train_models.py
결과:  models/<모델키>.joblib — {광고유형: 모델} 딕셔너리, 입력 X = 반기 순서(0 = 2020 상반기)
       models/catalog.json    — 유형별 점수·2026 예측
평가:  ① 팀 방식: 12개 반기 전체로 학습하고 같은 데이터로 채점
       ② 시간 순서: 2020 상반기~2023 하반기(8개)로 학습 → 2024 상반기~2025 하반기(4개) 예측
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from spam_features import TYPES, LastValue, load_spam

ROOT = Path(__file__).parent
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

cnt, share = load_spam(ROOT / "data" / "spam_raw.csv")
X = np.arange(len(share)).reshape(-1, 1)        # 반기 순서 0~11 (노트북과 동일)
SPLIT = 8                                        # 앞 8개 반기 학습 / 뒤 4개 평가
X_future = np.array([[12], [13]])                # 2026 상반기, 하반기

SPECS = [
    ("baseline", "기준선 (평균값)", "지금까지의 평균 비율을 그대로 예측", lambda: DummyRegressor(strategy="mean"), False),
    ("last", "최근 값 그대로", "'다음 반기도 지금과 같겠지' — 가장 최근 반기 비율을 그대로 사용", LastValue, False),
    ("linear", "첫 모델: 선형 회귀", "반기 순서로 직선 추세를 그음 (노트북 첫 모델)", LinearRegression, True),
    ("poly2", "최종 모델: 2차 다항 회귀", "반기 순서로 2차 곡선을 그음 (노트북 최종 모델)",
     lambda: make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()), True),
]

catalog = []
for key, name, desc, make, in_nb in SPECS:
    bundle, info = {}, {"key": key, "name": name, "desc": desc, "in_notebook": in_nb, "by_type": {}}
    for t in TYPES:
        y = share[t].to_numpy()
        full = make().fit(X, y)
        fit = full.predict(X)
        part = make().fit(X[:SPLIT], y[:SPLIT])
        test = part.predict(X[SPLIT:])
        fut = full.predict(X_future)
        bundle[t] = full
        info["by_type"][t] = {
            "r2_in": round(float(r2_score(y, fit)), 3), "mae_in": round(float(mean_absolute_error(y, fit)), 2),
            "mae_time": round(float(mean_absolute_error(y[SPLIT:], test)), 2),
            "test_pred": [round(float(v), 2) for v in test],
            "pred_2026": [round(float(v), 1) for v in fut],
        }
    joblib.dump(bundle, OUT / f"{key}.joblib", compress=3)
    catalog.append(info)
    g = info["by_type"]["금융"]
    print(f"{name:<20} [금융] 팀방식 R² {g['r2_in']:.3f} MAE {g['mae_in']:5.2f} | 시간순 MAE {g['mae_time']:5.2f}"
          f" | 2026 상/하 {g['pred_2026']}")

(OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
