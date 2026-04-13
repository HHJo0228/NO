# MILD — Multi-Scale ZigZag 분석 설계 문서

## 1. 개요

사용자의 과거 매매 타이밍이 **단기(Small) ZigZag**와 **장기(Large) ZigZag** 각각의 어느 위치에 있었는지를 분석하여, 매매 습관에 대한 자연어 평가 리포트를 생성하는 분석 엔진.

> "단기적으로는 오른쪽 무릎에서 사서 오른쪽 어깨에서 파는 훌륭한 매매를 하셨습니다.
> 하지만 장기적으로는 너무 일찍 팔아버리는 경향이 있습니다."

---

## 2. 분석 파이프라인

```
t_transactions + m_asset
        ↓
  ticker별 매매일 목록 추출
        ↓
  매매일 T마다:
    yfinance (T-5년 ~ T) 로드       ← T 이후 데이터 미사용 (hindsight bias 방지)
        ↓
    Rolling 임계값 계산 (T 기준 직전 5년)
        ↓
    Small / Large ZigZag 계산
        ↓
    매매 타이밍 위치 판별
        ↓
  위치 정보 JSON 생성
        ↓
  Gemini API → 자연어 평가 텍스트 생성
        ↓
  종목별 리포트 + 종합 리포트 출력
```

---

## 3. ZigZag 알고리즘

### 3-1. 임계값 계산 — Rolling Percentile

매매일 T 기준 직전 5년 데이터만 사용하여 단순 백분위수(Percentile)로 임계값을 계산한다.

- T 이전 데이터만 사용 → hindsight bias 방지
- 5년 고정 윈도우 → 통계적 안정성 확보
- 종목별 실제 상승/하락 분포를 분리하여 비대칭 적용

```
매매일 T의 임계값 = percentile( T-5년 ~ T 데이터 )
```

> **현재 코드 상태**: `mild_timing_eval.py`는 아직 EWMA 방식으로 구현되어 있음. Rolling Percentile로의 전환이 다음 단계 작업 항목임.

### 3-2. 비대칭 임계값

상승일과 하락일의 변동률 분포를 **각각 별도**로 계산한다.

```python
up_returns   = returns[returns > 0]        # 상승일 변동률
down_returns = returns[returns < 0].abs()  # 하락일 변동률

small_up   = percentile(up_returns,   P70)
small_down = percentile(down_returns, P70)
large_up   = percentile(up_returns,   P95) × LARGE_MULTIPLIER
large_down = percentile(down_returns, P95) × LARGE_MULTIPLIER
```

동일 분위수를 써도 **상승/하락 분포 자체가 다르므로** 종목 고유의 비대칭 특성이 임계값에 자연스럽게 반영된다.

| 종목 | 상승 P70 | 하락 P70 | 해석 |
|------|---------|---------|------|
| NVDA | 2.59% | 2.79% | 하락이 더 격렬 |
| 삼성전자 | 2.68% | 1.96% | 상승이 더 격렬 |

### 3-3. ZigZag 방향 전환 로직

```
상승 중: 고점 대비 down_thresh% 이상 하락 → 고점 확정, 하락 전환
하락 중: 저점 대비 up_thresh%   이상 상승 → 저점 확정, 상승 전환
```

### 3-4. 파라미터 목록

| 파라미터 | 기본값 | 역할 |
|---------|-------|------|
| `PERIOD_YEARS` | 5 | 데이터 로드 기간 |
| `VOLATILITY_LOOKBACK_DAYS` | 252 | (미사용, Rolling Percentile로 대체) |
| `SMALL_PERCENTILE` | 70 | Small ZigZag base 분위수 |
| `LARGE_PERCENTILE` | 95 | Large ZigZag base 분위수 |
| `LARGE_MULTIPLIER` | 4.0 | Large 임계값 확대 배율 (권장 3~5) |
| `UP_PERCENTILE` | 70 | 상승일 분포 분위수 |
| `DOWN_PERCENTILE` | 70 | 하락일 분포 분위수 |
| `CALC_YEARS` | 5 | Rolling 임계값 계산 윈도우 (고정) |

---

## 4. Rolling 임계값 (매매 타이밍 평가용)

과거 매매를 평가할 때는 **매매일 T 시점의 기준**으로 ZigZag를 계산한다.  
T 이후 데이터를 사용하면 후견지명(hindsight bias)이 발생하므로 절대 사용하지 않는다.

```
매매일 T의 임계값 = percentile( T-5년 ~ T 데이터만 사용 )
```

### 기간 구분

| 용도 | 기간 | 결정 방식 |
|------|------|---------|
| 임계값 계산 (`CALC_YEARS`) | 5년 고정 | 통계적 안정성 확보를 위해 고정값 사용 |
| 차트 표시 범위 | 가변 | 사용자의 첫 거래일 ~ 마지막 거래일을 자동으로 포함 |

두 기간은 독립적으로 동작한다. 사용자의 거래 기간이 5년을 초과하더라도 임계값은 항상 T 기준 직전 5년 데이터로 계산한다.

---

## 5. 위치 판별

매매일이 ZigZag 구간의 어느 지점에 있는지를 다음 수치로 표현한다.

| 필드 | 설명 | 예시 |
|------|------|------|
| `phase` | 상승/하락 구간 | `rising` / `falling` |
| `position_pct` | 직전 피벗 대비 변화율 | `+18.3%` |
| `leg_total_pct` | 해당 구간 전체 진폭 | `52.1%` |
| `progress_pct` | 구간 내 진행률 | `35.1%` (= 18.3/52.1) |
| `days_from_pivot` | 직전 피벗으로부터 경과일 | `12일` |
| `prev_pivot` | 직전 피벗 유형 | `trough` / `peak` |

### 구간 내 위치 레이블 (progress_pct 기준)

| 범위 | 레이블 | 의미 |
|------|--------|------|
| 0 ~ 30% | `early` | 초입 (무릎) |
| 30 ~ 70% | `mid` | 중반 |
| 70 ~ 100% | `late` | 말단 (어깨) |

---

## 6. AI 입력 데이터 구조 (JSON)

### 6-1. 종목별 분석 데이터

```json
{
  "ticker": "NVDA",
  "asset_name": "엔비디아",
  "transactions": [
    {
      "type": "BUY",
      "date": "2023-10-15",
      "price": 435.20,
      "qty": 10,
      "small_zigzag": {
        "phase": "rising",
        "position_pct": 18.3,
        "leg_total_pct": 52.1,
        "progress_pct": 35.1,
        "progress_label": "early",
        "days_from_pivot": 12,
        "prev_pivot": "trough"
      },
      "large_zigzag": {
        "phase": "rising",
        "position_pct": 23.1,
        "leg_total_pct": 210.4,
        "progress_pct": 11.0,
        "progress_label": "early",
        "days_from_pivot": 45,
        "prev_pivot": "trough"
      }
    },
    {
      "type": "SELL",
      "date": "2023-12-28",
      "price": 495.00,
      "qty": 10,
      "small_zigzag": { "..." },
      "large_zigzag": { "..." }
    }
  ],
  "pair_summary": {
    "return_pct": 13.7,
    "hold_days": 74,
    "small_buy_label": "rising_early",
    "small_sell_label": "rising_late",
    "large_buy_label": "rising_early",
    "large_sell_label": "rising_mid"
  }
}
```

### 6-2. 종합 리포트용 집계 데이터

토큰 효율을 위해 개별 종목 원문 대신 **패턴 집계 통계**만 전달한다.

```json
{
  "total_trades": 23,
  "tickers": ["NVDA", "TSLA", "005930.KS"],
  "avg_hold_days": 38.2,
  "avg_return_pct": 11.4,
  "small_buy_distribution": {
    "rising_early": 12,
    "rising_mid": 6,
    "rising_late": 3,
    "falling": 2
  },
  "large_buy_distribution": { "...": "..." },
  "small_sell_distribution": { "...": "..." },
  "large_sell_distribution": { "...": "..." }
}
```

---

## 7. 리포트 구조

```
분석 결과
  ├── 종목별 리포트 (per ticker)
  │     ├── 개별 매매 타이밍 평가 (매수/매도 각각)
  │     └── 해당 종목 매매 패턴 요약
  │
  └── 종합 리포트
        └── 전체 종목 통합 매매 습관 분석
```

종목별 리포트와 종합 리포트는 **별도의 Gemini API 호출**로 생성한다.

---

## 8. DB 스키마 (참고)

### 거래 내역

```sql
user.t_transactions (
  transaction_id  uuid PK,
  user_id         uuid FK → m_user_profile,
  date            timestamptz,
  account_id      uuid FK → m_account,
  asset_id        uuid FK → m_asset,
  transaction_type varchar(100),   -- 매수/매도 구분
  price           numeric(18,5),
  qty             numeric(18,5),
  fee             numeric(18,5),
  currency        varchar(100)
)
```

### 종목 마스터

```sql
asset.m_asset (
  asset_id      uuid PK,
  ticker        varchar(20),    -- yfinance 조회용 심볼
  market        varchar(20),
  currency      varchar(10),
  asset_name_en varchar(100),
  asset_name_ko varchar(100)
)
```

### 가격 캐시 (배치용, 미구현)

```sql
asset.t_price_cache (
  ticker      varchar(20),
  date        date,
  open        numeric(18,5),
  high        numeric(18,5),
  low         numeric(18,5),
  close       numeric(18,5),
  volume      bigint,
  fetched_at  timestamptz,
  PRIMARY KEY (ticker, date)
)
```

---

## 9. 배치 처리 설계 (매 주말)

```
[토요일 새벽] 가격 백필
  1. unique ticker 목록 추출 (active 사용자 보유 종목)
  2. yfinance 병렬 다운로드 (신규/누락 날짜만 증분)
  3. t_price_cache 저장

[일요일 새벽] ZigZag 분석
  1. 캐시된 가격 데이터 로드 (yfinance 재호출 없음)
  2. 사용자별 매매 타이밍 평가 (rolling 임계값)
  3. 분석 결과 저장
```

| 단계 | 예상 소요 시간 |
|------|-------------|
| unique ticker 추출 | < 1초 |
| 가격 백필 (증분) | 2~10분 |
| ZigZag 분석 전체 | 1~2분 |
| 결과 저장 | < 1분 |

---

## 10. 파일 구성

```
MILD/
├── multiscale_zigzag.py      # ZigZag 계산 엔진 + 시각화 (Plotly)
├── mild_timing_eval.py       # Rolling 임계값 기반 매매 타이밍 평가 엔진
├── mild_eval_result.json     # 평가 엔진 출력 샘플 (NVDA, 삼성전자)
├── mild_eval_viewer.html     # 평가 결과 뷰어 (종목 내러티브 + 매수/매도 스코어)
├── mild_pattern_chart.html   # 무차원 ZigZag 패턴 차트 (적응형 템플릿)
├── MILD_zigzag_design.md     # 본 설계 문서
└── zigzag_*.html             # 종목별 ZigZag 시각화 결과 (NVDA, 삼성전자, AAPL 등)
```

---

## 11. 구현 현황 및 다음 단계

### 구현 완료

- [x] yfinance 데이터 로드 (auto_adjust=True, 주식분할 소급 반영)
- [x] 상승/하락 비대칭 임계값 (up/down 분포 분리)
- [x] Rolling Percentile 임계값 계산 (T 기준 직전 5년, 비대칭 up/down 분포)
- [x] Small / Large ZigZag 계산
- [x] 시각화 (캔들 + ZigZag, 로그 스케일) — `multiscale_zigzag.py`
- [x] 분석 통계 출력 (평균 주기 저점→저점, 상승폭 ± 표준편차)
- [x] Rolling 임계값 계산 (매매일 T 기준, hindsight bias 방지)
- [x] 위치 판별 로직 (phase, progress_pct, progress_label)
- [x] 종목 성격 프로파일 계산 (`calc_asset_profile`) — 변동성 태그, 사이클 주기, 비대칭 등
- [x] 샘플 거래 데이터로 JSON 출력 검증 — `mild_eval_result.json`
- [x] 평가 결과 뷰어 — `mild_eval_viewer.html`
  - 종목 성격 한 줄 내러티브 (변동성 · 장기 파동 · 비대칭)
  - 매수/매도 타이밍 스코어 도트 (✓ 좋음 / △ 보통 / ✕ 아쉬움) + 요약 문장
  - `judgeTiming(type, zz)` 독립 모듈로 분리 → 프로덕션 이식 가능
- [x] 무차원 ZigZag 패턴 차트 (적응형 템플릿) — `mild_pattern_chart.html`
  - Large progress → x 좌표, Small progress → 작은 산 위치 역산
  - 매수/매도 평균 위치 마커 각 1개

### 다음 단계

- [ ] 프로덕션 이식: 버블차트 hover/click ↔ 타이밍 평가 연동 (`judgeTiming` 활용)
- [ ] Gemini API 연동 → 자연어 평가 텍스트 생성 (`mild_pattern_chart.html` insight 영역)
- [ ] 종목별 차트 표시 기간 자동 결정 (사용자의 첫 거래일 ~ 마지막 거래일 포함)
- [ ] DB 연동 (t_transactions, m_asset 스키마 연결)
- [ ] 배치 처리 구현 (가격 캐시 → ZigZag 분석 → 결과 저장)
