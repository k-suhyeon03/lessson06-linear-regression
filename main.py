
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고 "
    "선형회귀를 이용해 선택한 연도의 기온을 추정합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
기준연도 = 2025
최소관측일 = 300

# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "평균기온"]
    df = df[필요한열].copy()

    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지의 데이터만 사용
    df = df[df["연도"] <= 기준연도]

    # 연도별 평균기온과 유효한 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 해만 사용
    yearly = yearly[
        yearly["관측일수"] >= 최소관측일
    ].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()
except Exception as e:
    st.error(
        "기온 데이터를 불러오지 못했습니다. "
        "인터넷 연결과 데이터 주소를 확인해 주세요."
    )
    st.exception(e)
    st.stop()

if len(yearly) < 2:
    st.error(
        "회귀 직선을 계산하기 위한 데이터가 부족합니다."
    )
    st.stop()

# -----------------------------
# 선형회귀 및 상관계수
# -----------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

기울기, 절편 = np.polyfit(x, y, 1)
상관계수 = np.corrcoef(x, y)[0, 1]

시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())
사용연도수 = len(yearly)

# -----------------------------
# 연도 선택 및 예상 기온 표시
# -----------------------------
st.subheader("예상 기온 확인")

선택연도 = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

예상기온 = 기울기 * 선택연도 + 절편

st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{예상기온:.2f} °C"
)

st.caption(
    "과거 데이터에 맞춘 선형회귀 결과입니다. "
    "실제 미래 기온을 보장하지 않습니다."
)

# -----------------------------
# 분석 정보
# -----------------------------
st.subheader("회귀 분석 정보")

c1, c2, c3, c4 = st.columns(4)

c1.metric("회귀에 사용한 연도 수", f"{사용연도수}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")
c4.metric("상관계수", f"{상관계수:.3f}")

st.write(
    f"**회귀식:** 연평균기온 = "
    f"{기울기:.5f} × 연도 + ({절편:.3f})"
)

if 상관계수 > 0:
    st.write("연도와 연평균기온 사이에 양의 상관관계가 있습니다.")
elif 상관계수 < 0:
    st.write("연도와 연평균기온 사이에 음의 상관관계가 있습니다.")
else:
    st.write("연도와 연평균기온 사이의 선형 상관관계가 없습니다.")

# -----------------------------
# Plotly 산점도 + 회귀 직선
# -----------------------------
st.subheader("연도별 연평균기온과 회귀 직선")

# 그래프는 관측 자료가 있는 연도부터 2100년까지 표시
line_x = np.linspace(시작연도, 2100, 400)
line_y = 기울기 * line_x + 절편

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 슬라이더로 선택한 연도의 추정값 표시
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers",
        name="선택한 연도의 예상 기온",
        marker=dict(size=13, symbol="diamond"),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(l=20, r=20, t=50, b=20)
)

fig.update_xaxes(range=[시작연도, 2100])
fig.update_layout(
    height=550,
    template="plotly_white"
)

st.plotly_chart(fig, width="stretch")

# -----------------------------
# 사용한 데이터 확인
# -----------------------------
with st.expander("회귀 분석에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(
            columns={
                "연도": "연도",
                "연평균기온": "연평균기온 (°C)",
                "관측일수": "관측일수 (일)"
            }
        ),
        hide_index=True,
        width="stretch"
    )

st.caption(
    "데이터 출처: 서울 기온 데이터 | "
    "분석 기준: 2025년까지, 연간 유효 관측일 300일 이상"
)
