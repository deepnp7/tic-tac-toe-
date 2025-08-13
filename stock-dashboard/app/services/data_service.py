from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import csv

# Optional heavy deps
try:
    import yfinance as yf  # type: ignore
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - environment without heavy deps
    yf = None  # type: ignore
    pd = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from sqlalchemy import delete, select, func
from sqlalchemy.orm import Session

from ..config import PERIOD_TO_DAYS, CACHE_MAX_AGE_DAYS
from ..models import PriceBar
from ..schemas import PriceBarOut


def _period_days(period: str) -> int:
    return PERIOD_TO_DAYS.get(period, PERIOD_TO_DAYS["1y"])  # default 1y


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fetch_prices_yf(ticker: str, period: str, interval: str) -> List[PriceBarOut]:
    if yf is None:
        return []
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    if df is None or getattr(df, "empty", True):
        return []

    # Normalize dataframe
    df = df.dropna()
    if pd is not None:
        df.index = pd.to_datetime(df.index, utc=True)
    bars: List[PriceBarOut] = []
    for ts, row in df.iterrows():
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        bars.append(
            PriceBarOut(
                ticker=ticker,
                interval=interval,
                timestamp=ts_dt,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0)),
            )
        )
    return bars


def cache_prices(session: Session, bars: List[PriceBarOut]) -> None:
    if not bars:
        return
    ticker = bars[0].ticker
    interval = bars[0].interval
    # Replace cache for ticker+interval with new data to keep it simple
    session.execute(
        delete(PriceBar).where(PriceBar.ticker == ticker, PriceBar.interval == interval)
    )
    session.bulk_save_objects(
        [
            PriceBar(
                ticker=b.ticker,
                interval=b.interval,
                timestamp=b.timestamp.replace(tzinfo=None),
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
            )
            for b in bars
        ]
    )
    session.commit()


def load_cached_prices(
    session: Session, ticker: str, period: str, interval: str
) -> List[PriceBarOut]:
    days = _period_days(period)
    earliest_needed = _now_utc() - timedelta(days=days + 2)
    freshness_cutoff = _now_utc() - timedelta(days=CACHE_MAX_AGE_DAYS)

    # Check coverage and freshness
    latest_ts = session.execute(
        select(func.max(PriceBar.timestamp)).where(
            PriceBar.ticker == ticker, PriceBar.interval == interval
        )
    ).scalar()

    if latest_ts is None:
        return []

    if latest_ts.replace(tzinfo=timezone.utc) < freshness_cutoff:
        return []

    rows = session.execute(
        select(PriceBar)
        .where(
            PriceBar.ticker == ticker,
            PriceBar.interval == interval,
            PriceBar.timestamp >= earliest_needed.replace(tzinfo=None),
        )
        .order_by(PriceBar.timestamp.asc())
    ).scalars().all()

    if not rows:
        return []

    return [
        PriceBarOut(
            ticker=r.ticker,
            interval=r.interval,
            timestamp=r.timestamp.replace(tzinfo=timezone.utc),
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
        )
        for r in rows
    ]


def _read_sample_csv(ticker: str, interval: str) -> List[PriceBarOut]:
    path = "data/sample_ohlc.csv"
    bars: List[PriceBarOut] = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("ticker", "").upper() != ticker.upper():
                    continue
                ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                bars.append(
                    PriceBarOut(
                        ticker=ticker,
                        interval=interval,
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
        bars.sort(key=lambda b: b.timestamp)
    except FileNotFoundError:
        return []
    return bars


def get_prices(session: Session, ticker: str, period: str, interval: str) -> List[PriceBarOut]:
    # Try cache first
    cached = load_cached_prices(session, ticker, period, interval)
    if cached:
        return cached

    # Fetch fresh
    bars = fetch_prices_yf(ticker, period, interval)
    if not bars:
        # Fallback: try local sample dataset without pandas
        return _read_sample_csv(ticker, interval)

    # Save to cache
    cache_prices(session, bars)
    return bars


def compute_summary(bars: List[PriceBarOut], period: str, interval: str, ticker: str):
    if not bars:
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "high_52w": None,
            "low_52w": None,
            "average_volume": None,
        }

    if np is None:
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]
        high_52w = max(highs)
        low_52w = min(lows)
        avg_vol = sum(volumes) / len(volumes)
    else:
        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        volumes = np.array([b.volume for b in bars], dtype=float)
        high_52w = float(np.max(highs))
        low_52w = float(np.min(lows))
        avg_vol = float(np.mean(volumes))

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "high_52w": round(float(high_52w), 2),
        "low_52w": round(float(low_52w), 2),
        "average_volume": round(float(avg_vol), 2),
    }


def predict_next_close(bars: List[PriceBarOut]) -> Tuple[float, float | None]:
    if len(bars) < 2:
        return (bars[-1].close if bars else 0.0, None)

    if np is not None:
        N = min(60, len(bars))
        recent = bars[-N:]
        y = np.array([b.close for b in recent], dtype=float)
        x = np.arange(len(recent), dtype=float)
        coef = np.polyfit(x, y, 1)
        slope, intercept = coef[0], coef[1]
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
        next_pred = float(slope * (len(recent)) + intercept)
        return next_pred, float(r2) if r2 is not None else None

    # Pure-Python simple linear regression on last N points
    N = min(60, len(bars))
    recent = bars[-N:]
    y_vals = [b.close for b in recent]
    x_vals = list(range(N))
    n = float(N)
    sum_x = float(sum(x_vals))
    sum_y = float(sum(y_vals))
    sum_xx = float(sum(x * x for x in x_vals))
    sum_xy = float(sum(x_vals[i] * y_vals[i] for i in range(N)))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return y_vals[-1], None
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    y_pred_vals = [slope * x + intercept for x in x_vals]
    mean_y = sum_y / n
    ss_res = sum((y_vals[i] - y_pred_vals[i]) ** 2 for i in range(N))
    ss_tot = sum((y - mean_y) ** 2 for y in y_vals)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    next_pred = slope * N + intercept
    return float(next_pred), (float(r2) if r2 is not None else None)