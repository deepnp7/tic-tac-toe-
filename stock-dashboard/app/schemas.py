from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class CompanyOut(BaseModel):
    symbol: str
    name: str


class PriceBarOut(BaseModel):
    ticker: str
    interval: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class SummaryOut(BaseModel):
    ticker: str
    period: str
    interval: str
    high_52w: float
    low_52w: float
    average_volume: float


class PredictionOut(BaseModel):
    ticker: str
    next_day_close_prediction: float
    method: str
    r2: Optional[float] = None


class OHLCResponse(BaseModel):
    ticker: str
    period: str
    interval: str
    bars: List[PriceBarOut]