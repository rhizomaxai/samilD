"""
공통 모듈: 스팸 수신량 데이터 전처리 + '가장 최근 값' 모델
- train_models.py(모델 학습)와 app.py(웹앱)가 함께 import 합니다.
- joblib 모델 안에 이 파일의 클래스가 들어 있으므로 app.py와 같은 폴더에 두어야 합니다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

TYPES = ["도박", "불법대출", "금융", "통신가입", "성인", "불법의약품", "기타"]


def load_spam(csv_path: Path):
    """KOSIS 가로형 CSV(cp949) → (통수 표, 비율 표)
    두 표 모두 한 행 = 한 반기(2020 상반기 ~ 2025 하반기), 열 = 광고유형.
    원본 열 이름 '2020.1/2'는 2020년 상반기(2개 반기 중 1번째), '2020.2/2'는 하반기다."""
    raw = pd.read_csv(csv_path, encoding="cp949", header=None)
    periods = raw.iloc[0, 1::2].tolist()
    labels = [p.replace(".1/2", " 상반기").replace(".2/2", " 하반기") for p in periods]
    body = raw.iloc[2:].set_index(0)
    vals = body.astype(float).to_numpy()
    cnt = pd.DataFrame(vals[:, 0::2].T, index=labels, columns=body.index)[TYPES]
    share = pd.DataFrame(vals[:, 1::2].T, index=labels, columns=body.index)[TYPES]
    cnt.index.name = share.index.name = "반기"
    return cnt, share


def period_label(i: int) -> str:
    """반기 순서(0 = 2020 상반기) → '2026 상반기' 같은 이름"""
    return f"{2020 + i // 2} {'상' if i % 2 == 0 else '하'}반기"


class LastValue(BaseEstimator, RegressorMixin):
    """'다음 반기도 지금과 같겠지' — 가장 최근 값을 그대로 내놓는 단순 예측"""

    def fit(self, X, y):
        self.last_ = float(np.asarray(y)[-1])
        return self

    def predict(self, X):
        return np.full(len(X), self.last_)
