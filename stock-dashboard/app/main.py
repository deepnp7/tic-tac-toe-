from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .config import COMPANIES, DEFAULT_INTERVAL, DEFAULT_PERIOD
from .db import SessionLocal, engine
from .models import PriceBar
from .schemas import CompanyOut, OHLCResponse, PriceBarOut, SummaryOut, PredictionOut
from .services.data_service import get_prices, compute_summary, predict_next_close

# Create tables
from .db import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Stock Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/api/companies", response_model=list[CompanyOut])
async def list_companies():
    return [CompanyOut(**c) for c in COMPANIES]


@app.get("/api/ohlc", response_model=OHLCResponse)
async def get_ohlc(
    ticker: str = Query(..., description="Stock symbol e.g. AAPL"),
    period: str = Query(DEFAULT_PERIOD),
    interval: str = Query(DEFAULT_INTERVAL),
    db: Session = Depends(get_db),
):
    ticker = ticker.upper()
    bars = get_prices(db, ticker=ticker, period=period, interval=interval)
    if not bars:
        raise HTTPException(status_code=404, detail="No data available for the requested ticker.")
    return OHLCResponse(ticker=ticker, period=period, interval=interval, bars=bars)


@app.get("/api/summary", response_model=SummaryOut)
async def get_summary(
    ticker: str = Query(...),
    period: str = Query(DEFAULT_PERIOD),
    interval: str = Query(DEFAULT_INTERVAL),
    db: Session = Depends(get_db),
):
    ticker = ticker.upper()
    bars = get_prices(db, ticker=ticker, period=period, interval=interval)
    summary = compute_summary(bars, period=period, interval=interval, ticker=ticker)
    return SummaryOut(**summary)


@app.get("/api/predict", response_model=PredictionOut)
async def get_prediction(
    ticker: str = Query(...),
    period: str = Query(DEFAULT_PERIOD),
    interval: str = Query(DEFAULT_INTERVAL),
    db: Session = Depends(get_db),
):
    ticker = ticker.upper()
    bars = get_prices(db, ticker=ticker, period=period, interval=interval)
    pred, r2 = predict_next_close(bars)
    return PredictionOut(ticker=ticker, next_day_close_prediction=round(float(pred), 2), method="linear_regression_polyfit", r2=r2)


# Serve frontend
app.mount(
    "/", StaticFiles(directory="frontend", html=True), name="frontend"
)