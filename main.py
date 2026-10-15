
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================
# 1. 기본 설정
# =====================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선형회귀를 이용해 선택한 연도의 기온을 추정합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
기준연도 = 2025
최소관측일 = 300

# =====================================
# 2. 데이터 불러오기 및 전처리
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df = df[["날짜", "평균기온"]].copy()

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

    # 연도별 연평균기온 및 유효 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 유효 관측일이 300일 미만인 연도 제외
    yearly = yearly[
        yearly["관측일수"] >= 최소관측일
    ].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()
except Exception as e:
    st.error(
        "기온 데이터를 불러오지 못했습니다. "
        "인터넷 연결 또는 데이터 주소를 확인해 주세요."
    )
    st.exception(e)
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 위한 데이터가 부족합니다.")
    st.stop()

# =====================================
# 3. 전체 기간 회귀 분석
# =====================================
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

기울기, 절편 = np.polyfit(x, y, 1)
상관계수 = np.corrcoef(x, y)[0, 1]

시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())
사용연도수 = len(yearly)

# 연간 기울기를 100년당 변화량으로 환산
전체기간_100년변화 = 기울기 * 100

# =====================================
# 4. 최근 20년 회귀 분석
# =====================================
# 마지막 관측 연도를 기준으로 20년 범위 설정
최근20년_시작 = 끝연도 - 19

recent = yearly[
    (yearly["연도"] >= 최근20년_시작)
    & (yearly["연도"] <= 끝연도)
].copy()

if len(recent) >= 2 and recent["연도"].nunique() >= 2:
    최근기울기, 최근절편 = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1
    )

    최근20년_100년변화 = 최근기울기 * 100

    최근20년_상관계수 = np.corrcoef(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float)
    )[0, 1]
else:
    최근기울기 = None
    최근절편 = None
    최근20년_100년변화 = None
    최근20년_상관계수 = None

# =====================================
# 5. 100년당 기온 변화량 비교
# =====================================
st.header("📈 100년당 기온 변화량 비교")

st.caption(
    "회귀 직선의 기울기를 100배 하여 계산했습니다. "
    "이는 과거 관측 자료의 추세를 나타내며 "
    "미래의 실제 기온 변화를 보장하지 않습니다."
)

왼쪽, 오른쪽 = st.columns(2)

with 왼쪽:
    st.metric(
        label=f"전체 기간 ({시작연도}~{끝연도})",
        value=f"{전체기간_100년변화:+.2f} °C"
    )
    st.caption(f"사용한 연도 수: {사용연도수}년")
    st.caption(f"상관계수: {상관계수:.3f}")

with 오른쪽:
    if 최근20년_100년변화 is not None:
        st.metric(
            label=f"최근 20년 ({최근20년_시작}~{끝연도})",
            value=f"{최근20년_100년변화:+.2f} °C",
            delta=(
                f"{최근20년_100년변화 - 전체기간_100년변화:+.2f} °C "
                "vs 전체 기간"
            )
        )
        st.caption(f"사용한 연도 수: {len(recent)}년")
        st.caption(f"상관계수: {최근20년_상관계수:.3f}")
    else:
        st.warning(
            "최근 20년 구간에 회귀 계산을 위한 데이터가 부족합니다."
        )

st.divider()

# =====================================
# 6. 연도별 기온 예측
# =====================================
st.header("🌡️ 연도별 예상 기온")

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
    "전체 기간의 회귀 직선을 선택한 연도까지 연장한 추정값입니다. "
    "실제 기온 예보가 아닙니다."
)

# =====================================
# 7. 회귀 분석 정보
# =====================================
st.header("📊 회귀 분석 정보")

c1, c2, c3, c4 = st.columns(4)

c1.metric("사용한 연도 수", f"{사용연도수}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")
c4.metric("상관계수", f"{상관계수:.3f}")

st.write(
    f"**전체 기간 회귀식:** "
    f"연평균기온 = {기울기:.5f} × 연도 + ({절편:.3f})"
)

if 최근기울기 is not None:
    st.write(
        f"**최근 20년 회귀식:** "
        f"연평균기온 = {최근기울기:.5f} × 연도 + ({최근절편:.3f})"
    )

# =====================================
# 8. Plotly 산점도 및 회귀 직선
# =====================================
st.header("연도별 연평균기온 산점도와 회귀 직선")

# 실제 관측 연도부터 2100년까지 회귀 직선 표시
line_x = np.linspace(시작연도, 2100, 400)
line_y = 기울기 * line_x + 절편

fig = go.Figure()

# 실제 연평균기온 산점도
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

# 전체 기간 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers",
        name="선택 연도 예상 기온",
        marker=dict(size=13, symbol="diamond"),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    template="plotly_white",
    height=550,
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

st.plotly_chart(fig, width="stretch")

# =====================================
# 9. 연도별 데이터 표
# =====================================
with st.expander("회귀 분석에 사용한 연도별 데이터 보기"):
    표 = yearly.rename(
        columns={
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }
    )

    st.dataframe(
        표,
        hide_index=True,
        width="stretch"
    )

st.caption(
    "데이터 출처: 서울 기온 데이터 | "
    "분석 기준: 2025년까지, 유효 관측일 300일 이상"
)
