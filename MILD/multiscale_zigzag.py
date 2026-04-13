"""
Multi-Scale ZigZag Pattern Analyzer for MILD
Usage:
    python multiscale_zigzag.py [ticker] [period_years]
    python multiscale_zigzag.py               # default: 005930.KS (Samsung), 3yr
    python multiscale_zigzag.py TSLA 5
    python multiscale_zigzag.py NVDA 2
"""

import sys
import io

# Force UTF-8 output — 직접 실행 시에만 적용 (import 시 중복 래핑 방지)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# 1. 파라미터 설정
# ─────────────────────────────────────────────
TICKER = sys.argv[1] if len(sys.argv) > 1 else "005930.KS"   # 삼성전자
PERIOD_YEARS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# 변동성 계산에 사용할 기간
VOLATILITY_LOOKBACK_DAYS = 252          # 최근 1년 (영업일 기준)
# Small ZigZag: 상위 30% 변동률
SMALL_PERCENTILE = 70                   # np.percentile 기준 (100-30)
# Large ZigZag: 상위 5% 변동률 × 가중치
LARGE_PERCENTILE = 95                   # np.percentile 기준 (100-5)
LARGE_MULTIPLIER = 4.0                  # 3~5 사이 가중치

# 비대칭 임계값: 상승일/하락일 변동률 분포를 각각 별도로 계산
# UP_PERCENTILE   : 상승일 변동률 분포에서 사용할 분위수
# DOWN_PERCENTILE : 하락일 변동률 분포에서 사용할 분위수
# 같은 분위수를 써도 상승/하락 분포 자체가 다르므로 종목별 실제 특성이 반영됨
UP_PERCENTILE   = 70   # 상승일 변동률의 P70 → 상승 반전 임계값 (Small 기준)
DOWN_PERCENTILE = 70   # 하락일 변동률의 P70 → 하락 반전 임계값 (Small 기준)



# ─────────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────────
def load_data(ticker: str, years: int) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    print(f"\n[데이터 로드] {ticker} | {start_date.date()} ~ {end_date.date()}")

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"데이터를 불러올 수 없습니다: {ticker}")

    # MultiIndex 컬럼 처리 (yfinance 0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    print(f"  → {len(df)}개 캔들 로드 완료")
    return df


# ─────────────────────────────────────────────
# 3. 변동성 기반 임계값 계산
# ─────────────────────────────────────────────
def calc_thresholds(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Rolling Percentile 방식으로 임계값 계산.
    전달된 df 범위(매매일 T 기준 직전 CALC_YEARS년) 데이터만 사용.
    상승일/하락일 변동률 분포를 각각 별도로 계산(비대칭).

    Returns:
        (small_up, small_down, large_up, large_down)
    """
    returns = df["Close"].pct_change().dropna() * 100

    up_returns   = returns[returns > 0].values
    down_returns = returns[returns < 0].abs().values

    small_up   = float(np.percentile(up_returns,   UP_PERCENTILE))
    small_down = float(np.percentile(down_returns, DOWN_PERCENTILE))
    large_up   = float(np.percentile(up_returns,   LARGE_PERCENTILE)) * LARGE_MULTIPLIER
    large_down = float(np.percentile(down_returns, LARGE_PERCENTILE)) * LARGE_MULTIPLIER

    print(f"\n[임계값 계산]  (Rolling Percentile, {len(returns)}일)")
    print(f"  사용 데이터               : {len(returns)}일 (상승 {len(up_returns)}일 / 하락 {len(down_returns)}일)")
    print(f"  Small ZigZag  상승 임계값 : {small_up:.2f}%  (상승일 P{UP_PERCENTILE})")
    print(f"  Small ZigZag  하락 임계값 : {small_down:.2f}%  (하락일 P{DOWN_PERCENTILE})")
    print(f"  Large ZigZag  상승 임계값 : {large_up:.2f}%  (상승일 P{LARGE_PERCENTILE} × {LARGE_MULTIPLIER})")
    print(f"  Large ZigZag  하락 임계값 : {large_down:.2f}%  (하락일 P{LARGE_PERCENTILE} × {LARGE_MULTIPLIER})")
    return small_up, small_down, large_up, large_down


# ─────────────────────────────────────────────
# 4. ZigZag 알고리즘
# ─────────────────────────────────────────────
def calc_zigzag(prices: pd.Series, up_pct: float, down_pct: float) -> pd.DataFrame:
    """
    비대칭 임계값 ZigZag.
      - 상승 반전 감지: 저점 대비 up_pct%   이상 상승 → 저점 확정 후 상승 방향 전환
      - 하락 반전 감지: 고점 대비 down_pct% 이상 하락 → 고점 확정 후 하락 방향 전환

    up_pct > down_pct  → 하락 반전을 더 민감하게 감지 (하락 이탈 조기 포착)
    up_pct < down_pct  → 상승 반전을 더 민감하게 감지

    Returns:
        DataFrame with columns [date, price, type]
        type: 'peak' | 'trough'
    """
    prices = prices.reset_index()
    prices.columns = ["date", "price"]
    n = len(prices)

    pivots = []
    direction = None
    extreme_idx = 0
    extreme_price = prices["price"].iloc[0]

    thresh_up   = up_pct   / 100.0
    thresh_down = down_pct / 100.0

    for i in range(1, n):
        p = prices["price"].iloc[i]

        if direction is None:
            # 방향 미결정: 먼저 임계값에 도달한 방향으로 시작
            change = (p - extreme_price) / extreme_price
            if change >= thresh_up:
                pivots.append((prices["date"].iloc[extreme_idx], extreme_price, "trough"))
                direction = "up"
                extreme_idx = i
                extreme_price = p
            elif change <= -thresh_down:
                pivots.append((prices["date"].iloc[extreme_idx], extreme_price, "peak"))
                direction = "down"
                extreme_idx = i
                extreme_price = p
        elif direction == "up":
            if p >= extreme_price:
                extreme_idx = i
                extreme_price = p
            elif (extreme_price - p) / extreme_price >= thresh_down:
                # 고점 대비 down_pct% 이상 하락 → 고점 확정
                pivots.append((prices["date"].iloc[extreme_idx], extreme_price, "peak"))
                direction = "down"
                extreme_idx = i
                extreme_price = p
        elif direction == "down":
            if p <= extreme_price:
                extreme_idx = i
                extreme_price = p
            elif (p - extreme_price) / extreme_price >= thresh_up:
                # 저점 대비 up_pct% 이상 상승 → 저점 확정
                pivots.append((prices["date"].iloc[extreme_idx], extreme_price, "trough"))
                direction = "up"
                extreme_idx = i
                extreme_price = p

    # 마지막 미확정 극값 추가
    if direction == "up":
        pivots.append((prices["date"].iloc[extreme_idx], extreme_price, "peak"))
    elif direction == "down":
        pivots.append((prices["date"].iloc[extreme_idx], extreme_price, "trough"))

    result = pd.DataFrame(pivots, columns=["date", "price", "type"])
    return result


# ─────────────────────────────────────────────
# 5. 분석 결과 출력
# ─────────────────────────────────────────────
def print_stats(label: str, pivots: pd.DataFrame):
    peaks = pivots[pivots["type"] == "peak"]
    troughs = pivots[pivots["type"] == "trough"]

    print(f"\n[{label} 분석 결과]")
    print(f"  전체 피벗 수   : {len(pivots)}")
    print(f"  고점(Peak) 수  : {len(peaks)}")
    print(f"  저점(Trough) 수: {len(troughs)}")

    if len(troughs) >= 2:
        trough_dates = pd.to_datetime(troughs["date"])
        diffs = trough_dates.diff().dropna().dt.days
        print(f"  평균 주기 (저점→저점) : {diffs.mean():.1f}일 (±{diffs.std():.1f}일)")
        print(f"  최소/최대             : {diffs.min():.0f}일 / {diffs.max():.0f}일")

    if len(peaks) > 0:
        avg_gain = []
        for _, row in peaks.iterrows():
            # 직전 trough 찾기
            prev = pivots[(pivots["type"] == "trough") & (pivots["date"] < row["date"])]
            if not prev.empty:
                trough_price = prev.iloc[-1]["price"]
                gain = (row["price"] - trough_price) / trough_price * 100
                avg_gain.append(gain)
        if avg_gain:
            arr = np.array(avg_gain)
            print(f"  평균 상승폭     : {arr.mean():.1f}% ± {arr.std():.1f}% (저점→고점)")


# ─────────────────────────────────────────────
# 6. 시각화
# ─────────────────────────────────────────────
def plot_chart(df: pd.DataFrame, small_pivots: pd.DataFrame, large_pivots: pd.DataFrame,
               small_up: float, small_down: float,
               large_up: float, large_down: float, ticker: str):

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.82, 0.18],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker} — Multi-Scale ZigZag  |  Small ↑{small_up:.1f}% ↓{small_down:.1f}%  |  Large ↑{large_up:.1f}% ↓{large_down:.1f}%",
            "거래량"
        ]
    )

    # ── 캔들스틱 ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="캔들",
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",
        increasing_fillcolor="#26A69A",
        decreasing_fillcolor="#EF5350",
        line_width=1,
        showlegend=False,
    ), row=1, col=1)

    # ── Large ZigZag (굵은 파란색) ──
    fig.add_trace(go.Scatter(
        x=large_pivots["date"],
        y=large_pivots["price"],
        mode="lines+markers",
        name=f"Large ZigZag (↑{large_up:.1f}% ↓{large_down:.1f}%)",
        line=dict(color="#1565C0", width=3),
        marker=dict(
            size=[14 if t == "peak" else 10 for t in large_pivots["type"]],
            color=["#1565C0" if t == "peak" else "#42A5F5" for t in large_pivots["type"]],
            symbol=["triangle-up" if t == "peak" else "triangle-down" for t in large_pivots["type"]],
            line=dict(color="white", width=1.5),
        ),
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "가격: %{y:,.0f}<br>"
            "<extra>Large ZigZag</extra>"
        ),
    ), row=1, col=1)

    # ── Small ZigZag (얇은 오렌지색) ──
    fig.add_trace(go.Scatter(
        x=small_pivots["date"],
        y=small_pivots["price"],
        mode="lines+markers",
        name=f"Small ZigZag (↑{small_up:.1f}% ↓{small_down:.1f}%)",
        line=dict(color="#E65100", width=1.5, dash="dot"),
        marker=dict(
            size=[9 if t == "peak" else 7 for t in small_pivots["type"]],
            color=["#E65100" if t == "peak" else "#FFA726" for t in small_pivots["type"]],
            symbol=["triangle-up" if t == "peak" else "triangle-down" for t in small_pivots["type"]],
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "가격: %{y:,.0f}<br>"
            "<extra>Small ZigZag</extra>"
        ),
    ), row=1, col=1)

    # ── 거래량 바 차트 ──
    if "Volume" in df.columns:
        colors = ["#26A69A" if c >= o else "#EF5350"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["Volume"],
            name="거래량",
            marker_color=colors,
            opacity=0.6,
            showlegend=False,
        ), row=2, col=1)

    # ── 레이아웃 ──
    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Malgun Gothic, Arial", size=12, color="#e0e0e0"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#444",
            borderwidth=1,
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=80, b=40),
    )

    fig.update_xaxes(
        gridcolor="#2a2a4a",
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#2a2a4a",
        showgrid=True,
        zeroline=False,
    )
    # 가격 축만 로그 스케일 (거래량 row=2 는 선형 유지)
    fig.update_yaxes(type="log", row=1, col=1)

    # 고점/저점 주석 (Large ZigZag 만 표시 — 가독성)
    # yref="y" 는 로그 스케일에서 지수 변환이 일어나 축이 폭발하므로 사용 금지
    # → paper 좌표 대신 log10 변환 후 yref="y" 로 직접 지정
    import math
    annotations = []
    for _, row in large_pivots.iterrows():
        is_peak = row["type"] == "peak"
        log_price = math.log10(float(row["price"]))
        annotations.append(dict(
            x=row["date"],
            y=log_price,
            xref="x", yref="y",
            text=f"{row['price']:,.0f}",
            showarrow=False,
            font=dict(size=9, color="#90CAF9" if is_peak else "#80DEEA"),
            yshift=12 if is_peak else -14,
            xanchor="center",
        ))
    fig.update_layout(annotations=annotations)

    # 로그 스케일 y축 범위를 가격 데이터 기준으로 고정 (여백 5%)
    log_min = math.log10(float(df["Low"].min())) * 0.98
    log_max = math.log10(float(df["High"].max())) * 1.02
    fig.update_yaxes(range=[log_min, log_max], row=1, col=1)

    output_path = f"zigzag_{ticker.replace('.', '_')}.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"\n[차트 저장] {output_path}")

    fig.show()
    return output_path


# ─────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────
def main():
    df = load_data(TICKER, PERIOD_YEARS)

    # 거래량 별도 보관 (시각화용)
    vol_data = None
    try:
        raw = yf.download(TICKER,
                          start=df.index[0],
                          end=df.index[-1] + timedelta(days=1),
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if "Volume" in raw.columns:
            df["Volume"] = raw["Volume"]
    except Exception:
        pass

    small_up, small_down, large_up, large_down = calc_thresholds(df)

    print("\n[ZigZag 계산 중...]")
    small_pivots = calc_zigzag(df["Close"], small_up, small_down)
    large_pivots = calc_zigzag(df["Close"], large_up, large_down)

    print_stats("Small ZigZag", small_pivots)
    print_stats("Large ZigZag", large_pivots)

    plot_chart(df, small_pivots, large_pivots,
               small_up, small_down, large_up, large_down, TICKER)


if __name__ == "__main__":
    main()
