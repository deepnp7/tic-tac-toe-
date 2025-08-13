from sqlalchemy import Column, Integer, String, Float, DateTime, Index, UniqueConstraint
from .db import Base


class PriceBar(Base):
    __tablename__ = "price_bars"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    interval = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "interval", "timestamp", name="uq_ticker_interval_ts"),
        Index("ix_ticker_interval_ts", "ticker", "interval", "timestamp"),
    )