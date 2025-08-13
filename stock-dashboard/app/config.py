import os
from datetime import timedelta

# Database URL can be overridden by env
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stocks.db")

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
CACHE_MAX_AGE_DAYS = 2

# At least 10 companies
COMPANIES = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc. (Class A)"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc."},
    {"symbol": "META", "name": "Meta Platforms, Inc."},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "NFLX", "name": "Netflix, Inc."},
    {"symbol": "DIS", "name": "The Walt Disney Company"},
    {"symbol": "INTC", "name": "Intel Corporation"},
]

PERIOD_TO_DAYS = {
    "1mo": 31,
    "3mo": 92,
    "6mo": 183,
    "1y": 366,
    "2y": 731,
    "5y": 1826,
}