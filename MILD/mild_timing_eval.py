"""
MILD — 매매 타이밍 평가 엔진
매매일 T 기준으로 T 이전 데이터만 사용해 임계값/ZigZag를 계산하고,
각 매매가 Small/Large ZigZag의 어느 위치에 있었는지 판별한다.

Usage:
    python mild_timing_eval.py [ticker]
    python mild_timing_eval.py NVDA
    python mild_timing_eval.py 005930.KS
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# multiscale_zigzag.py의 핵심 함수 재사용
from multiscale_zigzag import (
    calc_zigzag,
    SMALL_PERCENTILE,
    LARGE_PERCENTILE,
    LARGE_MULTIPLIER,
    UP_PERCENTILE,
    DOWN_PERCENTILE,
)

CALC_YEARS = 5   # 임계값/ZigZag 계산에 사용할 최대 과거 기간


# ─────────────────────────────────────────────
# 1. 데이터 로드 (전체 기간, 캐시 역할)
# ─────────────────────────────────────────────
def load_price_data(ticker: str, years: int = CALC_YEARS) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    df = yf.download(ticker, start=start_date, end=end_date,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"데이터 없음: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ─────────────────────────────────────────────
# 2. 매매일 T 기준 Rolling 임계값 계산
# ─────────────────────────────────────────────
def calc_thresholds_at(df: pd.DataFrame, as_of: pd.Timestamp) -> tuple[float, float, float, float]:
    """
    as_of(매매일 T) 이전 데이터만 사용해 Rolling Percentile 임계값 계산.
    T 이후 데이터는 절대 사용하지 않아 hindsight bias를 방지한다.

    Returns:
        (small_up, small_down, large_up, large_down)
    """
    past = df[df.index <= as_of]
    if len(past) < 60:
        raise ValueError(f"{as_of.date()} 기준 과거 데이터 부족 ({len(past)}일)")

    returns = past["Close"].pct_change().dropna() * 100
    up_r   = returns[returns > 0].values
    down_r = returns[returns < 0].abs().values

    small_up   = float(np.percentile(up_r,   UP_PERCENTILE))
    small_down = float(np.percentile(down_r, DOWN_PERCENTILE))
    large_up   = float(np.percentile(up_r,   LARGE_PERCENTILE)) * LARGE_MULTIPLIER
    large_down = float(np.percentile(down_r, LARGE_PERCENTILE)) * LARGE_MULTIPLIER

    return small_up, small_down, large_up, large_down


# ─────────────────────────────────────────────
# 3. 위치 판별
# ─────────────────────────────────────────────
PROGRESS_LABELS = [
    (0.30, "early"),   # 0~30%: 초입 (무릎)
    (0.70, "mid"),     # 30~70%: 중반
    (1.00, "late"),    # 70~100%: 말단 (어깨)
]

def _progress_label(progress_pct: float) -> str:
    for threshold, label in PROGRESS_LABELS:
        if progress_pct / 100.0 <= threshold:
            return label
    return "late"


def locate_on_zigzag(trade_date: pd.Timestamp, trade_price: float,
                     pivots: pd.DataFrame) -> dict:
    """
    매매일/가격이 ZigZag 피벗 구조의 어느 위치에 있는지 판별.

    매매일 기준 직전 확정 피벗까지만 알 수 있으므로,
    직전 피벗 → 매매일 구간의 진행률을 계산한다.

    Returns:
        {
          phase,            # 'rising' | 'falling'
          prev_pivot,       # 'trough' | 'peak'
          prev_pivot_date,
          prev_pivot_price,
          position_pct,     # 직전 피벗 대비 변화율 (%)
          leg_total_pct,    # 직전 피벗 ~ 직후 피벗 전체 진폭 (알 수 있는 경우)
          progress_pct,     # 구간 내 진행률 (%) — leg_total 미확정 시 None
          progress_label,   # 'early' | 'mid' | 'late' — 미확정 시 None
          days_from_pivot,  # 직전 피벗으로부터 경과일
        }
    """
    past_pivots = pivots[pivots["date"] <= trade_date]
    if past_pivots.empty:
        return {"phase": None, "prev_pivot": None, "error": "피벗 없음"}

    prev = past_pivots.iloc[-1]
    prev_type  = prev["type"]
    prev_price = float(prev["price"])
    prev_date  = pd.Timestamp(prev["date"])

    # 직전 피벗이 trough → 현재 상승 구간
    # 직전 피벗이 peak  → 현재 하락 구간
    phase = "rising" if prev_type == "trough" else "falling"
    position_pct = (trade_price - prev_price) / prev_price * 100

    # 직후 피벗 (T 이후이므로 미래 → leg_total은 참고용으로만 포함)
    future_pivots = pivots[pivots["date"] > trade_date]
    leg_total_pct = None
    progress_pct  = None
    progress_label = None

    if not future_pivots.empty:
        next_price = float(future_pivots.iloc[0]["price"])
        leg_total_pct = abs((next_price - prev_price) / prev_price * 100)
        if leg_total_pct > 0:
            progress_pct  = abs(position_pct) / leg_total_pct * 100
            progress_label = _progress_label(progress_pct)

    days_from_pivot = (trade_date - prev_date).days

    return {
        "phase":            phase,
        "prev_pivot":       prev_type,
        "prev_pivot_date":  prev_date.strftime("%Y-%m-%d"),
        "prev_pivot_price": round(prev_price, 4),
        "position_pct":     round(position_pct, 2),
        "leg_total_pct":    round(leg_total_pct, 2) if leg_total_pct is not None else None,
        "progress_pct":     round(progress_pct, 1)  if progress_pct  is not None else None,
        "progress_label":   progress_label,
        "days_from_pivot":  days_from_pivot,
    }


# ─────────────────────────────────────────────
# 4. 단일 매매 평가
# ─────────────────────────────────────────────
def evaluate_trade(df: pd.DataFrame,
                   trade_date: pd.Timestamp,
                   trade_type: str,
                   trade_price: float,
                   trade_qty: float) -> dict:
    """
    매매일 T 기준으로:
      1. T 이전 데이터로 EWMA 임계값 계산
      2. Small / Large ZigZag 계산
      3. Small / Large 각각에서 위치 판별

    Returns:
        {
          date, type, price, qty,
          thresholds: {small_up, small_down, large_up, large_down},
          small_zigzag: {...locate 결과...},
          large_zigzag: {...locate 결과...},
        }
    """
    # 1. Rolling 임계값 (T 이전만)
    small_up, small_down, large_up, large_down = calc_thresholds_at(df, trade_date)

    # 2. ZigZag 계산 (T 이전만)
    past_prices = df[df.index <= trade_date]["Close"]
    small_pivots = calc_zigzag(past_prices, small_up, small_down)
    large_pivots = calc_zigzag(past_prices, large_up, large_down)

    # 3. 위치 판별
    # leg_total_pct(직후 피벗)는 전체 데이터로 계산해야 의미 있으므로
    # 위치 판별만 전체 ZigZag 피벗 기준으로 별도 수행
    full_small = calc_zigzag(df["Close"], small_up, small_down)
    full_large = calc_zigzag(df["Close"], large_up, large_down)

    small_loc = locate_on_zigzag(trade_date, trade_price, full_small)
    large_loc = locate_on_zigzag(trade_date, trade_price, full_large)

    return {
        "date":  trade_date.strftime("%Y-%m-%d"),
        "type":  trade_type,
        "price": round(trade_price, 4),
        "qty":   trade_qty,
        "thresholds": {
            "small_up":   round(small_up,   2),
            "small_down": round(small_down, 2),
            "large_up":   round(large_up,   2),
            "large_down": round(large_down, 2),
        },
        "small_zigzag": small_loc,
        "large_zigzag": large_loc,
    }


# ─────────────────────────────────────────────
# 5. 매수/매도 쌍 요약
# ─────────────────────────────────────────────
def summarize_pairs(evaluated: list[dict]) -> list[dict]:
    """
    BUY → SELL 순으로 페어링하여 수익률/보유일/포지션 조합 요약.
    단순 FIFO 방식으로 매칭.
    """
    buys  = [t for t in evaluated if t["type"].upper() == "BUY"]
    sells = [t for t in evaluated if t["type"].upper() == "SELL"]

    pairs = []
    for buy, sell in zip(buys, sells):
        ret = (sell["price"] - buy["price"]) / buy["price"] * 100
        hold_days = (
            datetime.strptime(sell["date"], "%Y-%m-%d") -
            datetime.strptime(buy["date"],  "%Y-%m-%d")
        ).days

        def label(loc, t_type):
            if loc.get("phase") is None:
                return "unknown"
            phase = loc["phase"]
            pl    = loc.get("progress_label") or "unknown"
            return f"{phase}_{pl}"

        pairs.append({
            "buy_date":  buy["date"],
            "sell_date": sell["date"],
            "buy_price": buy["price"],
            "sell_price": sell["price"],
            "return_pct":  round(ret, 2),
            "hold_days":   hold_days,
            "small_buy_label":  label(buy["small_zigzag"],  "buy"),
            "small_sell_label": label(sell["small_zigzag"], "sell"),
            "large_buy_label":  label(buy["large_zigzag"],  "buy"),
            "large_sell_label": label(sell["large_zigzag"], "sell"),
        })
    return pairs


# ─────────────────────────────────────────────
# 6. 종목 성격 프로파일 계산
# ─────────────────────────────────────────────
def calc_asset_profile(df: pd.DataFrame, ticker: str) -> dict:
    """
    종목 고유의 파동 성격 지표 계산.
    전체 기간 데이터 기준 (시각화/성격 카드용).

    Returns:
        {
          small_cycle_days, small_gain_pct, small_gain_std,
          small_volatility_tag,
          large_cycle_days, large_gain_pct, large_gain_std,
          large_cycle_tag,
          asymmetry,         # 상승/하락 임계값 비율
          tags,              # 종목 성격 태그 목록
        }
    """
    returns = df["Close"].pct_change().dropna() * 100
    up_r   = returns[returns > 0].values
    down_r = returns[returns < 0].abs().values

    small_up   = float(np.percentile(up_r,   UP_PERCENTILE))
    small_down = float(np.percentile(down_r, DOWN_PERCENTILE))
    large_up   = float(np.percentile(up_r,   LARGE_PERCENTILE)) * LARGE_MULTIPLIER
    large_down = float(np.percentile(down_r, LARGE_PERCENTILE)) * LARGE_MULTIPLIER

    small_pivots = calc_zigzag(df["Close"], small_up, small_down)
    large_pivots = calc_zigzag(df["Close"], large_up, large_down)

    def cycle_and_gain(pivots):
        troughs = pivots[pivots["type"] == "trough"]
        peaks   = pivots[pivots["type"] == "peak"]
        # 저점→저점 주기
        cycle_days, cycle_std = None, None
        if len(troughs) >= 2:
            diffs = pd.to_datetime(troughs["date"]).diff().dropna().dt.days
            cycle_days = round(float(diffs.mean()), 1)
            cycle_std  = round(float(diffs.std()),  1)
        # 저점→고점 상승폭
        gains = []
        for _, pk in peaks.iterrows():
            prev = pivots[(pivots["type"] == "trough") & (pivots["date"] < pk["date"])]
            if not prev.empty:
                g = (pk["price"] - prev.iloc[-1]["price"]) / prev.iloc[-1]["price"] * 100
                gains.append(g)
        gain_avg = round(float(np.mean(gains)), 1) if gains else None
        gain_std = round(float(np.std(gains)),  1) if gains else None
        return cycle_days, cycle_std, gain_avg, gain_std

    s_cycle, s_cycle_std, s_gain, s_gain_std = cycle_and_gain(small_pivots)
    l_cycle, l_cycle_std, l_gain, l_gain_std = cycle_and_gain(large_pivots)

    # 비대칭성: 상승 임계값 / 하락 임계값 (>1 이면 하락이 더 격렬)
    asymmetry = round(small_down / small_up, 2)

    # ── 성격 태그 ──
    tags = []

    # 변동성 태그 (Small 상승폭 기준)
    if s_gain is not None:
        if s_gain >= 10:   tags.append("고변동")
        elif s_gain >= 5:  tags.append("중변동")
        else:              tags.append("저변동")

    # 단기 사이클 태그
    if s_cycle is not None:
        if s_cycle <= 14:  tags.append("단기 사이클 빠름")
        elif s_cycle <= 30: tags.append("단기 사이클 보통")
        else:               tags.append("단기 사이클 느림")

    # 장기 흐름 태그
    if l_cycle is not None:
        irregularity = (l_cycle_std / l_cycle) if l_cycle else 0
        if irregularity > 0.5:  tags.append("장기 흐름 불규칙")
        else:                   tags.append("장기 흐름 규칙적")

    # 큰 흐름 폭발성
    if l_gain is not None:
        if l_gain >= 80:   tags.append("큰 흐름 폭발적")
        elif l_gain >= 40: tags.append("큰 흐름 강함")
        else:              tags.append("큰 흐름 완만")

    # 방향성 편향
    if asymmetry > 1.1:   tags.append("하락 속도 빠름")
    elif asymmetry < 0.9: tags.append("상승 속도 빠름")

    return {
        "small_cycle_days":  s_cycle,
        "small_cycle_std":   s_cycle_std,
        "small_gain_pct":    s_gain,
        "small_gain_std":    s_gain_std,
        "small_thresholds":  {"up": round(small_up, 2), "down": round(small_down, 2)},
        "large_cycle_days":  l_cycle,
        "large_cycle_std":   l_cycle_std,
        "large_gain_pct":    l_gain,
        "large_gain_std":    l_gain_std,
        "large_thresholds":  {"up": round(large_up, 2), "down": round(large_down, 2)},
        "asymmetry":         asymmetry,
        "tags":              tags,
    }


# ─────────────────────────────────────────────
# 7. 전체 종목 평가 → JSON 생성
# ─────────────────────────────────────────────
def evaluate_ticker(ticker: str, asset_name: str,
                    transactions: list[dict]) -> dict:
    """
    단일 종목의 전체 거래를 평가하고 AI 입력용 JSON 반환.

    transactions: [{"date": "YYYY-MM-DD", "type": "BUY"/"SELL",
                    "price": float, "qty": float}, ...]
    """
    print(f"\n[{ticker}] 데이터 로드 중...")
    df = load_price_data(ticker)
    print(f"  → {len(df)}행 로드 완료 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 종목 성격 프로파일
    profile = calc_asset_profile(df, ticker)
    print(f"  → 성격 태그: {', '.join(profile['tags'])}")

    evaluated = []
    for tx in transactions:
        trade_date = pd.Timestamp(tx["date"])
        if trade_date < df.index[0]:
            print(f"  ! {trade_date.date()} — 데이터 범위 밖, 건너뜀")
            continue
        print(f"  평가 중: {tx['type']:4s} {trade_date.date()} @ {tx['price']}")
        result = evaluate_trade(
            df,
            trade_date  = trade_date,
            trade_type  = tx["type"],
            trade_price = float(tx["price"]),
            trade_qty   = float(tx.get("qty", 1)),
        )
        evaluated.append(result)

    pairs = summarize_pairs(evaluated)

    return {
        "ticker":        ticker,
        "asset_name":    asset_name,
        "asset_profile": profile,
        "transactions":  evaluated,
        "pair_summary":  pairs,
    }


# ─────────────────────────────────────────────
# 8. 집계 통계 (종합 리포트용)
# ─────────────────────────────────────────────
def aggregate_stats(all_results: list[dict]) -> dict:
    """
    전체 종목 결과를 집계하여 Claude 종합 프롬프트용 통계 생성.
    토큰 효율을 위해 개별 거래 원문 대신 분포 통계만 포함.
    """
    all_pairs = []
    for r in all_results:
        all_pairs.extend(r["pair_summary"])

    def dist(key):
        counts: dict[str, int] = {}
        for p in all_pairs:
            v = p.get(key, "unknown")
            counts[v] = counts.get(v, 0) + 1
        return counts

    returns   = [p["return_pct"]  for p in all_pairs]
    hold_days = [p["hold_days"]   for p in all_pairs]

    return {
        "total_pairs":   len(all_pairs),
        "tickers":       [r["ticker"] for r in all_results],
        "avg_return_pct":  round(float(np.mean(returns)),   2) if returns   else None,
        "std_return_pct":  round(float(np.std(returns)),    2) if returns   else None,
        "avg_hold_days":   round(float(np.mean(hold_days)), 1) if hold_days else None,
        "small_buy_distribution":  dist("small_buy_label"),
        "small_sell_distribution": dist("small_sell_label"),
        "large_buy_distribution":  dist("large_buy_label"),
        "large_sell_distribution": dist("large_sell_label"),
    }


# ─────────────────────────────────────────────
# 9. 샘플 실행 (테스트용)
# ─────────────────────────────────────────────
SAMPLE_TRANSACTIONS = {
    # 가격은 yfinance auto_adjust=True 기준 조정 종가 (주식분할 소급 반영)
    # NVDA: 2024-06-10 10:1 분할 → 분할 전 거래도 1/10 가격으로 표기됨
    "NVDA": [
        {"date": "2022-10-14", "type": "BUY",  "price": 11.21,  "qty": 100},
        {"date": "2023-02-10", "type": "SELL", "price": 21.24,  "qty": 100},
        {"date": "2023-08-18", "type": "BUY",  "price": 43.27,  "qty": 50},
        {"date": "2023-11-20", "type": "SELL", "price": 50.37,  "qty": 50},
        {"date": "2024-04-19", "type": "BUY",  "price": 76.16,  "qty": 30},
        {"date": "2024-06-20", "type": "SELL", "price": 130.72, "qty": 30},
    ],
    # 삼성전자: 분할 없음, 조정 종가 사용
    "005930.KS": [
        {"date": "2022-09-30", "type": "BUY",  "price": 49528, "qty": 20},
        {"date": "2023-06-15", "type": "SELL", "price": 67496, "qty": 20},
        {"date": "2024-02-05", "type": "BUY",  "price": 71193, "qty": 15},
        {"date": "2024-07-10", "type": "SELL", "price": 84888, "qty": 15},
    ],
}

SAMPLE_NAMES = {
    "NVDA":       "엔비디아",
    "005930.KS":  "삼성전자",
}


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    tickers = [ticker] if ticker else list(SAMPLE_TRANSACTIONS.keys())

    all_results = []
    for t in tickers:
        if t not in SAMPLE_TRANSACTIONS:
            print(f"샘플 거래 데이터 없음: {t}")
            continue
        result = evaluate_ticker(
            ticker       = t,
            asset_name   = SAMPLE_NAMES.get(t, t),
            transactions = SAMPLE_TRANSACTIONS[t],
        )
        all_results.append(result)

    # 종목별 JSON 출력
    print("\n" + "="*60)
    print("[종목별 평가 결과 JSON]")
    print("="*60)
    for r in all_results:
        print(f"\n--- {r['ticker']} ({r['asset_name']}) ---")
        print(json.dumps(r, ensure_ascii=False, indent=2))

    # 종합 집계
    stats = aggregate_stats(all_results)
    print("\n" + "="*60)
    print("[종합 집계 (Claude 종합 프롬프트용)]")
    print("="*60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 파일 저장
    output = {"per_ticker": all_results, "aggregate": stats}
    out_path = "mild_eval_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
