## Stock Market Dashboard

A clean, responsive web app to explore stock prices with a FastAPI backend, SQLite caching, and a Chart.js frontend. Includes basic AI prediction for next-day close using linear regression.

### Features
- Responsive UI with a left sidebar listing 10+ companies
- FastAPI REST API: `/api/companies`, `/api/ohlc`, `/api/summary`, `/api/predict`
- Historical OHLC data via `yfinance`, cached in SQLite
- Chart.js line chart for close prices
- Summary stats: 52-week high/low, average volume
- Simple AI prediction (next-day close) using numpy linear regression
- Optional fallback to sample dataset in `data/sample_ohlc.csv`

### Tech Stack
- Backend: FastAPI, SQLAlchemy, SQLite, yfinance, pandas, numpy
- Frontend: HTML/CSS/JavaScript, Chart.js
- DevOps: Uvicorn, Dockerfile provided

### Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
3. Open the app at `http://localhost:8000`.

### Docker
```bash
docker build -t stock-dashboard .
docker run -p 8000:8000 stock-dashboard
```

### Notes (200–300 words)
This project was built with a pragmatic full‑stack approach: a lightweight FastAPI backend powers a static, responsive frontend. The backend exposes REST endpoints to list companies and serve historical OHLC data, which is sourced from Yahoo Finance via `yfinance`. To keep the app fast and resilient, results are cached in a local SQLite database with a simple replace‑on‑refresh strategy per ticker and interval. When live data is unavailable, the API can fall back to a small sample dataset to keep the UI functional.

On the frontend, the focus is clarity and responsiveness. A minimal layout with a left navigation column and a main chart panel keeps the experience simple. Chart.js renders the closing price series, while a compact stats bar highlights 52‑week high/low and average volume. For a bonus feature, a tiny AI component estimates the next day’s closing price using linear regression over recent closes; it is intentionally simple to remain transparent and dependency‑light.

Key challenges included normalizing data across different intervals, choosing a cache policy that stays fresh without complicating the schema, and keeping the UI fast while handling potentially large time series. The solution favors straightforward code and well‑named structures so the app can be extended with technical indicators, more sophisticated models, authentication, or a switch to PostgreSQL for multi‑user deployments.