"""
스팸 단속의 '풍선효과'는 있을까? — 삼일고 2학년 1반 밤티팀 광고유형별 휴대전화 스팸 수신량 분석
실행:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import spam_features  # noqa: F401  (joblib 모델 안의 클래스를 불러오는 데 필요)
from spam_features import TYPES, load_spam, period_label

ROOT = Path(__file__).parent
COLORS = {"도박": "#7c3aed", "불법대출": "#dc2626", "금융": "#2563eb", "통신가입": "#0891b2",
          "성인": "#db2777", "불법의약품": "#65a30d", "기타": "#94a3b8"}
MODEL_COLORS = {"baseline": "#94a3b8", "last": "#f59e0b", "linear": "#60a5fa", "poly2": "#dc2626"}

st.set_page_config(page_title="밤티팀 · 스팸 풍선효과 분석", page_icon="📵", layout="wide")


@st.cache_data
def get_data():
    return load_spam(ROOT / "data" / "spam_raw.csv")


@st.cache_data
def get_catalog():
    return json.loads((ROOT / "models" / "catalog.json").read_text(encoding="utf-8"))


@st.cache_resource          # 모델별로 한 번만 joblib 로드
def load_model(key):
    return joblib.load(ROOT / "models" / f"{key}.joblib")


cnt, share = get_data()
catalog = get_catalog()
cat = {c["key"]: c for c in catalog}
names = {c["key"]: c["name"] for c in catalog}
periods = list(share.index)
short = [p.replace(" 상반기", "-상").replace(" 하반기", "-하") for p in periods]

# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.markdown("### 📵 밤티팀")
    st.caption("삼일고등학교 · 2학년 1반")
    st.markdown("""
| 이름 | 역할 |
|---|---|
| 강민지 | 팀장·발표, 데이터 수집·전처리 |
| 송현지 | 모델링·평가, 보고서·시각화 |
""")
    st.divider()
    st.caption("데이터: KOSIS 국가통계포털 「광고유형별 휴대전화 스팸(문자+음성) 수신량 추이」 2020~2025 반기별 · "
               "출처 표시 후 자유 이용")
    st.caption("제출일: 2026. 10. 30.")

# ───────────────────────── 헤더 ─────────────────────────
st.title("📵 스팸을 막으면 다른 스팸이 늘어날까?")
st.markdown("**광고유형별 휴대전화 스팸 수신량으로 '풍선효과' 검증하고 다음 반기 비율 예측하기**")

top_now = share.iloc[-1].idxmax()
c1, c2, c3, c4 = st.columns(4)
c1.metric("데이터", f"{len(share)}개 반기", help=f"2020 상반기 ~ 2025 하반기, 광고유형 {len(TYPES)}개")
c2.metric("2025 하반기 1위 유형", f"{top_now} {share.iloc[-1].max():.1f}%")
c3.metric("금융 스팸 비율 변화", f"{share['금융'].iloc[-1]:.1f}%",
          f"{share['금융'].iloc[-1] - share['금융'].max():+.1f}%p (최고점 대비)", delta_color="inverse")
c4.metric("최종 모델 R² (금융, 학습 데이터)", f"{cat['poly2']['by_type']['금융']['r2_in']:.3f}",
          f"첫 모델 {cat['linear']['by_type']['금융']['r2_in']:.3f}", delta_color="off", delta_arrow="off")

tabs = st.tabs(["① 문제 정의", "② 데이터", "③ 풍선효과 검증", "④ 모델 평가", "⑤ 2026 예측 체험", "⑥ 결론·한계"])

# ───────────────────────── ① 문제 정의 ─────────────────────────
with tabs[0]:
    st.subheader("배경과 목적")
    st.info("주식방, 불법 도박, 대출 사기 같은 스팸 문자가 끊이지 않아 많은 사람이 불편을 겪는다. 스팸의 종류가 계속 바뀌는 만큼, "
            "광고 종류별로 스팸이 시간에 따라 어떻게 변하는지 데이터로 확인하고, 어떤 스팸이 늘어나고 있는지 알아보고자 했다.")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("분석 질문")
        st.markdown("""
| 단계 | 내용 |
|---|---|
| 궁금증 | 특정 스팸(예: 금융)을 강하게 단속해서 줄어들면, 발송자들이 수법을 바꿔 다른 스팸(도박·기타)을 더 많이 보낼까? |
| 질문 | 한 광고유형의 수신 비율이 뚝 떨어질 때 다른 유형의 비율이 동시에 올라가는가? |
| **분석 질문** | **광고유형별 데이터의 상관관계로, 특정 스팸 단속이 다른 유형의 증가로 이어지는 '풍선효과'가 실제로 있는지 검증할 수 있는가?** |
""")
    with right:
        st.subheader("머신러닝 문제로 바꾸기")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 정답 y | 특정 스팸 유형의 반기별 수신 비율 (%) |
| 특성 X | 반기 순서 (0 = 2020 상반기) |
| 문제 종류 | ✅ **회귀** (정답이 숫자) |
| 사용자 | 방송통신위원회·통신사 등 스팸 대응 담당자 |
""")

# ───────────────────────── ② 데이터 ─────────────────────────
with tabs[1]:
    view = st.radio("보기", ["비율 (%) — 팀이 분석에 쓴 값", "통수 (통) — 실제 수신량"], horizontal=True)
    data = share if view.startswith("비율") else cnt
    pick = st.multiselect("광고유형", TYPES, default=TYPES)
    fig = go.Figure()
    for t in pick:
        fig.add_scatter(x=short, y=data[t], mode="lines+markers", name=t,
                        line=dict(color=COLORS[t], width=3 if t in ("금융", "기타") else 1.8))
    if not view.startswith("비율"):
        fig.add_scatter(x=short, y=cnt.sum(axis=1), mode="lines+markers", name="전체 합계",
                        line=dict(color="black", width=3, dash="dot"))
    fig.update_layout(height=430, yaxis_title="비율 (%)" if view.startswith("비율") else "수신량 (통)",
                      margin=dict(t=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("데이터 출처")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 데이터 | 광고유형별 휴대전화 스팸(문자+음성) 수신량 추이 |
| 출처 | KOSIS 국가통계포털 |
| 기간 | 2020 상반기 ~ 2025 하반기 (12개 반기) |
| 구조 | 7개 유형 × (통수, 비율) — 가로형 |
""")
    with right:
        st.subheader("전처리 기록")
        st.markdown("""
| 문제 | 처리 | 이유 |
|---|---|---|
| 통수·비율 열이 번갈아 섞임 | 비율 열만 골라 사용 | 유형별 비중 비교 |
| 첫 행이 단위 설명 | 삭제 (`drop(0)`) | 데이터가 아님 |
| 비율이 글자(object) | 숫자(float)로 변환 | 계산·그래프 |
| 표가 가로로 넓음 | `melt`로 세로형 변환 | 그래프·모델 학습 |
| 결측치·중복 | 확인 결과 없음 | — |
""")
    st.caption("※ 통수는 소수 둘째 자리(0.01통)까지만 공개되어 있어 값이 아주 작은 유형(성인·불법의약품)은 0.00으로 나온다. "
               "최근 반기의 비율은 이 작은 통수로 계산되어 4.35%, 17.39%처럼 거친 값이 반복된다.")
    with st.expander("전처리된 비율 표 보기"):
        st.dataframe(share, use_container_width=True)

# ───────────────────────── ③ 풍선효과 검증 ─────────────────────────
with tabs[2]:
    st.subheader("분석 질문: 한 스팸이 줄면 다른 스팸이 늘어날까?")
    left, right = st.columns(2)
    with left:
        st.markdown("**① 팀 분석 — 비율(%) 상관관계**")
        corr = share.corr().round(2)
        fig = go.Figure(go.Heatmap(z=corr.values, x=TYPES, y=TYPES, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                                   text=corr.values, texttemplate="%{text}"))
        fig.update_layout(height=400, margin=dict(t=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**② 추가 분석 — 실제 수신량(통)의 반기별 변화량 상관관계**")
        dc = cnt.drop(columns=["불법의약품"]).diff().dropna()
        corr2 = dc.corr().round(2)
        fig = go.Figure(go.Heatmap(z=corr2.values, x=list(dc.columns), y=list(dc.columns), colorscale="RdBu",
                                   zmid=0, zmin=-1, zmax=1, text=corr2.values, texttemplate="%{text}"))
        fig.update_layout(height=400, margin=dict(t=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("불법의약품은 통수가 모두 0.00이라 계산에서 제외했다.")

    st.warning("**비율로만 보면 생기는 함정** — 7개 유형의 비율은 항상 합이 100%다. 그래서 금융 스팸이 실제로 줄지 않아도 "
               "다른 스팸이 늘면 금융 비율은 저절로 떨어진다. 비율끼리 음(−)의 상관이 나오는 것은 '풍선효과'가 아니어도 "
               "생길 수 있다. 풍선효과를 확인하려면 실제 수신량(통수)이 반대로 움직이는지 봐야 한다.")

    st.markdown("**두 유형을 골라 반기별 변화 비교하기**")
    a, b, c = st.columns([1, 1, 2])
    t1 = a.selectbox("줄어든 유형 (단속 대상)", TYPES, index=TYPES.index("금융"))
    t2 = b.selectbox("늘어났을지 모를 유형", [t for t in TYPES if t != t1], index=5 if t1 == "금융" else 0)
    unit = c.radio("기준", ["통수 변화 (통)", "비율 변화 (%p)"], horizontal=True)
    src = cnt if unit.startswith("통수") else share
    d = src[[t1, t2]].diff().dropna()
    fig = go.Figure(go.Scatter(x=d[t1], y=d[t2], mode="markers+text", text=[s for s in short[1:]],
                               textposition="top center", textfont=dict(size=9),
                               marker=dict(size=11, color=COLORS[t1])))
    fig.add_hline(y=0, line_color="#cbd5e1")
    fig.add_vline(x=0, line_color="#cbd5e1")
    fig.add_annotation(x=d[t1].min(), y=d[t2].max(), text="풍선효과라면 여기(왼쪽 위)에 점이 몰림",
                       showarrow=False, xanchor="left", font=dict(color="#dc2626"))
    fig.update_layout(height=400, xaxis_title=f"{t1} 변화", yaxis_title=f"{t2} 변화", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    r = d[t1].corr(d[t2])
    st.caption(f"상관계수 r = {r:.2f} (음수이고 −1에 가까울수록 풍선효과에 가까움). 자료가 11개 변화량뿐이라 우연의 영향이 크다.")

    st.success(f"**검증 결과** — 비율 기준으로는 금융과 기타(r = {share['금융'].corr(share['기타']):.2f}), "
               f"금융과 통신가입(r = {share['금융'].corr(share['통신가입']):.2f})이 반대로 움직였지만, 실제 수신량 변화로 보면 "
               f"금융과 기타는 오히려 같은 방향(r = {corr2.loc['금융', '기타']:.2f})이었다. "
               "따라서 이 데이터만으로는 '금융 스팸 단속 → 다른 스팸 증가'라는 풍선효과를 확인할 수 없다. "
               "팀 보고서의 결론('뚜렷한 연관성은 확인되지 않음')과 같다.")

# ───────────────────────── ④ 모델 평가 ─────────────────────────
with tabs[3]:
    st.subheader("반기 순서로 스팸 비율 예측하기")
    t = st.selectbox("광고유형", TYPES, index=TYPES.index("금융"), key="eval_type")
    rows = []
    for c in catalog:
        s = c["by_type"][t]
        rows.append({"모델": c["name"], "R² (학습 데이터)": s["r2_in"], "MAE (학습 데이터)": s["mae_in"],
                     "MAE (시간순 평가)": s["mae_time"], "출처": "노트북" if c["in_notebook"] else "웹앱 추가"})
    score = pd.DataFrame(rows)
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(score, hide_index=True, use_container_width=True)
        st.caption("**학습 데이터** = 팀 방식(12개 반기로 학습하고 같은 데이터로 채점). "
                   "**시간순 평가** = 2020\\~2023년 8개 반기로 학습 → 2024\\~2025년 4개 반기를 처음 보는 문제로 예측.")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=short, y=share[t], mode="lines+markers", name="실제", line=dict(color="black", width=3))
        for c in catalog:
            fig.add_scatter(x=short[8:], y=c["by_type"][t]["test_pred"], mode="lines+markers", name=c["name"],
                            line=dict(color=MODEL_COLORS[c["key"]], dash="dot"))
        fig.add_vrect(x0=7.5, x1=11.5, fillcolor="#e2e8f0", opacity=0.4, line_width=0,
                      annotation_text="시간순 평가 구간", annotation_position="top left")
        fig.update_layout(height=380, yaxis_title=f"{t} 비율 (%)", margin=dict(t=20), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**7개 유형 전체: 시간순 평가 MAE (%p, 낮을수록 좋음)**")
    allm = pd.DataFrame({c["name"]: {ty: c["by_type"][ty]["mae_time"] for ty in TYPES} for c in catalog})
    st.dataframe(allm.style.highlight_min(axis=1, color="#bbf7d0").format("{:.1f}"), use_container_width=True)
    st.warning("**학습 점수가 높다고 미래도 잘 맞힐까?** 금융 스팸의 2차 다항 회귀는 R²가 0.116 → 0.757로 크게 올랐지만, "
               "이는 정답을 이미 본 데이터로 채점한 점수다. 2차 곡선은 끝부분이 계속 꺾여 내려가 2026년 하반기에는 "
               "비율이 −18%라는 불가능한 값을 예측한다. 유형마다 가장 좋은 모델도 다르다.")

# ───────────────────────── ⑤ 2026 예측 체험 ─────────────────────────
with tabs[4]:
    st.subheader("🔮 다음 반기 스팸 비율 예측해 보기")
    ctrl, out = st.columns([1, 3])
    with ctrl:
        ptype = st.selectbox("광고유형", TYPES, index=TYPES.index("금융"))
        mkey = st.selectbox("모델 선택 (joblib 파일 교체)", list(names), index=3, format_func=lambda k: names[k])
        horizon = st.slider("몇 반기 앞까지?", 1, 6, 2)
        st.caption(f"📂 `models/{mkey}.joblib` → `['{ptype}']`")
        st.markdown(f"**모델 설명** — {cat[mkey]['desc']}")
    model = load_model(mkey)[ptype]
    idx = np.arange(12 + horizon)
    pred = model.predict(idx.reshape(-1, 1))
    mae = cat[mkey]["by_type"][ptype]["mae_time"]
    with out:
        cols = st.columns(min(horizon, 3))
        for i, col in enumerate(cols):
            v = pred[12 + i]
            shown = min(max(v, 0), 100)
            col.metric(period_label(12 + i), f"{shown:.1f}%",
                       "⚠️ 0% 미만 → 0%로 표시" if v < 0 else (f"범위 {max(v - mae, 0):.1f} ~ {min(v + mae, 100):.1f}%"),
                       delta_color="off", delta_arrow="off")
        labels = [period_label(i).replace(" 상반기", "-상").replace(" 하반기", "-하") for i in idx]
        fig = go.Figure()
        fig.add_scatter(x=labels[:12], y=share[ptype], mode="lines+markers", name="실제",
                        line=dict(color="black", width=3))
        fig.add_scatter(x=labels, y=pred, mode="lines", name=names[mkey], line=dict(color=MODEL_COLORS[mkey], width=2.5))
        fig.add_scatter(x=labels[12:], y=pred[12:], mode="markers", marker=dict(size=12, symbol="star",
                        color=MODEL_COLORS[mkey]), showlegend=False)
        fig.add_hrect(y0=min(pred.min(), 0) - 5, y1=0, fillcolor="#fecaca", opacity=0.4, line_width=0,
                      annotation_text="불가능한 영역 (0% 미만)", annotation_position="bottom left")
        fig.add_vrect(x0=11.5, x1=11.5 + horizon, fillcolor="#e2e8f0", opacity=0.4, line_width=0,
                      annotation_text="예측", annotation_position="top left")
        fig.update_layout(height=420, yaxis_title=f"{ptype} 비율 (%)", margin=dict(t=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"범위 = 예측값 ± 시간순 평가 평균 오차({mae:.1f}%p). 비율은 0~100% 사이여야 한다.")

    st.markdown("**2026 상반기, 모델별 예측 비교 (%)**")
    comp = pd.DataFrame({c["name"]: {ty: c["by_type"][ty]["pred_2026"][0] for ty in TYPES} for c in catalog}).T
    comp["합계"] = comp.sum(axis=1)
    st.dataframe(comp.style.format("{:.1f}").map(lambda v: "color:#dc2626;font-weight:600" if v < 0 else ""),
                 use_container_width=True)
    st.caption("유형마다 따로 예측하면 7개 비율의 합이 100%가 되지 않는다. 비율 자료를 예측할 때 주의할 점이다.")

# ───────────────────────── ⑥ 결론·한계 ─────────────────────────
with tabs[5]:
    st.subheader("결론")
    st.success("**풍선효과**: 비율끼리는 반대로 움직이는 유형이 있었지만, 실제 수신량(통수)으로 보면 금융 스팸이 줄 때 다른 스팸이 "
               "늘어나는 뚜렷한 패턴은 없었다. 이 데이터만으로는 풍선효과를 확인할 수 없다.\n\n"
               "**예측**: 금융 스팸 비율은 2차 다항 회귀가 학습 데이터에서 R² 0.757로 가장 잘 맞았지만, 미래 구간에서는 곡선이 "
               "0% 아래로 내려가 그대로 쓸 수 없다. 12개 반기라는 짧은 자료로는 다음 반기 비율을 '신뢰성 있게' 예측하기 어렵고, "
               "최근 값이나 평균처럼 단순한 예측과 함께 비교해서 봐야 한다.")
    st.subheader("한계와 보완 방법")
    st.markdown("""
| 구분 | 한계 | 보완 방법 |
|---|---|---|
| 데이터 | 12개 반기뿐이고, 통수가 0.01 단위로 반올림되어 최근 비율이 거칠다 | 월별·분기별 자료, 더 정밀한 수신량 자료 확보 |
| 분석 | 비율은 합이 100%라 서로 음의 상관이 저절로 생김 | 실제 수신량(통수)과 단속 시점 자료로 검증 |
| 모델 | 반기 순서만 사용 → 단속 정책·사회 이슈 반영 못 함, 2차 곡선은 미래에서 폭주 | 정책 시행 시기 등 변수 추가, 0~100% 범위를 지키는 모델 |
| 평가 | 학습한 데이터로 채점해 점수가 부풀려짐 | 시간 순서로 나눠 평가 (이 앱의 '시간순 평가') |
""")
    st.subheader("윤리 점검")
    st.markdown("✅ 개인 식별 정보 없음 · ✅ 출처·이용 조건 준수 · ✅ 결과를 과장하지 않고 한계 제시 · "
                "⚠️ 상관관계 ≠ 인과관계: 비율이 반대로 움직여도 단속 때문이라고 단정할 수 없음")
    st.caption("참고: 국가통계포털(KOSIS), 「광고유형별 휴대전화 스팸 수신량 추이」, 2020~2025, 2026.10.29 내려받음.")
