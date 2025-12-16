from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import anthropic
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Noema Financial API - Placeholder Mode")

# CORS - Allow your frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
import os

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Validate that Claude API key is set
if not CLAUDE_API_KEY:
    logger.warning("CLAUDE_API_KEY environment variable not set! AI summaries will not work.")
    CLAUDE_API_KEY = None

# Cache for quotes
quote_cache: Dict[str, dict] = {}
last_update: Optional[datetime] = None

# Cache for AI summary
summary_cache = {
    "text": None,
    "timestamp": None
}

# Stock name mappings
STOCK_NAMES = {
    'NVDA': 'NVIDIA Corporation',
    'AAPL': 'Apple Inc',
    'TSLA': 'Tesla Inc',
    'MSFT': 'Microsoft Corporation',
    'AMD': 'Advanced Micro Devices',
    'GOOGL': 'Alphabet Inc',
    'META': 'Meta Platforms',
    'AMZN': 'Amazon.com Inc',
    'NFLX': 'Netflix Inc',
    'INTC': 'Intel Corporation',
    'TSM': 'Taiwan Semiconductor',
    'SMCI': 'Super Micro Computer',
    'ASML': 'ASML Holding NV',
    'DELL': 'Dell Technologies',
    'PLTR': 'Palantir Technologies',
    'COIN': 'Coinbase Global',
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF',
    'UVXY': 'VIX ETF',
    'NBIS': 'Nebius Group',
    'NET': 'Cloudflare Inc',
    'SNOW': 'Snowflake Inc',
    'RKLB': 'Rocket Lab USA',
    'BTC/USD': 'Bitcoin'
}

def generate_placeholder_quote(symbol: str) -> dict:
    """Generate placeholder quote data"""
    # Base prices (realistic starting points)
    base_prices = {
        'NVDA': 140.0,
        'AAPL': 195.0,
        'TSLA': 380.0,
        'MSFT': 425.0,
        'AMD': 145.0,
        'GOOGL': 165.0,
        'META': 580.0,
        'AMZN': 215.0,
        'NFLX': 880.0,
        'INTC': 25.0,
        'TSM': 195.0,
        'SMCI': 45.0,
        'ASML': 850.0,
        'DELL': 130.0,
        'PLTR': 75.0,
        'COIN': 285.0,
        'SPY': 600.0,
        'QQQ': 525.0,
        'UVXY': 35.0,
        'NBIS': 22.0,
        'NET': 115.0,
        'SNOW': 165.0,
        'RKLB': 28.0,
        'BTC/USD': 102000.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    
    # Add small random variation (-2% to +2%)
    variation = random.uniform(-0.02, 0.02)
    price = base_price * (1 + variation)
    
    # Calculate change
    change = price - base_price
    change_percent = (change / base_price) * 100
    
    return {
        "symbol": symbol,
        "name": STOCK_NAMES.get(symbol, symbol),
        "price": round(price, 2),
        "change": round(change, 2),
        "changePercent": round(change_percent, 2),
        "volume": random.randint(10000000, 100000000),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Noema Financial API - PLACEHOLDER MODE",
        "provider": "No real data - placeholder only",
        "last_update": last_update.isoformat() if last_update else None,
        "cached_symbols": len(quote_cache),
        "note": "Using placeholder data - API integration pending"
    }


@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Get quote for a single symbol"""
    symbol = symbol.upper()
    
    # Generate placeholder quote
    quote = generate_placeholder_quote(symbol)
    quote_cache[symbol] = quote
    
    return {
        "quote": quote,
        "last_update": datetime.now().isoformat(),
        "source": "Placeholder data"
    }


@app.get("/api/quotes")
async def get_quotes(symbols: str):
    """Get quotes for multiple symbols (comma-separated)"""
    symbol_list = [s.strip().upper() for s in symbols.split(',')]
    
    quotes = {}
    for symbol in symbol_list:
        quote = generate_placeholder_quote(symbol)
        quotes[symbol] = quote
        quote_cache[symbol] = quote
    
    global last_update
    last_update = datetime.now()
    
    logger.info(f"Generated {len(quotes)} placeholder quotes")
    
    return {
        "quotes": quotes,
        "last_update": last_update.isoformat(),
        "source": "Placeholder data"
    }


@app.get("/api/watchlist")
async def get_watchlist():
    """Get default watchlist quotes"""
    default_symbols = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMD']
    
    quotes = {}
    for symbol in default_symbols:
        quote = generate_placeholder_quote(symbol)
        quotes[symbol] = quote
        quote_cache[symbol] = quote
    
    global last_update
    last_update = datetime.now()
    
    return {
        "quotes": quotes,
        "last_update": last_update.isoformat(),
        "source": "Placeholder data"
    }


@app.get("/api/market-indices")
async def get_market_indices():
    """Get market indices (S&P, Nasdaq, VIX, Bitcoin)"""
    indices = {
        'SPY': generate_placeholder_quote('SPY'),
        'QQQ': generate_placeholder_quote('QQQ'),
        'UVXY': generate_placeholder_quote('UVXY'),
        'BTC/USD': generate_placeholder_quote('BTC/USD')
    }
    
    global last_update
    last_update = datetime.now()
    
    return {
        "indices": indices,
        "last_update": last_update.isoformat(),
        "source": "Placeholder data"
    }


@app.get("/api/suggested")
async def get_suggested_stocks():
    """Get suggested stocks"""
    suggested = ['TSM', 'SMCI', 'ASML', 'DELL', 'PLTR', 'COIN']
    
    quotes = {}
    for symbol in suggested:
        quote = generate_placeholder_quote(symbol)
        quotes[symbol] = quote
    
    return {
        "quotes": quotes,
        "last_update": datetime.now().isoformat(),
        "source": "Placeholder data"
    }


@app.post("/api/refresh")
async def refresh_quotes():
    """Force refresh all cached quotes"""
    global last_update
    
    # Clear cache
    quote_cache.clear()
    last_update = datetime.now()
    
    logger.info("Quote cache cleared")
    
    return {
        "status": "ok",
        "message": "Cache cleared - new placeholder data will be generated on next request",
        "last_update": last_update.isoformat()
    }


@app.get("/api/market-summary")
async def get_market_summary():
    """Get AI-generated market summary"""
    global summary_cache
    
    # Check if we have a cached summary less than 4 hours old
    if summary_cache["text"] and summary_cache["timestamp"]:
        age = datetime.now() - summary_cache["timestamp"]
        if age < timedelta(hours=4):
            logger.info("Returning cached summary")
            return {
                "summary": summary_cache["text"],
                "updated": summary_cache["timestamp"].isoformat(),
                "source": "Claude AI (cached)"
            }
    
    # Generate new summary (placeholder for now)
    summary_cache["text"] = "Market summary temporarily unavailable while API integration is being updated. Please check back soon."
    summary_cache["timestamp"] = datetime.now()
    
    return {
        "summary": summary_cache["text"],
        "updated": summary_cache["timestamp"].isoformat(),
        "source": "Placeholder"
    }


@app.post("/api/refresh-summary")
async def refresh_summary():
    """Force refresh the AI summary"""
    global summary_cache
    
    summary_cache["text"] = None
    summary_cache["timestamp"] = None
    
    logger.info("Summary cache cleared")
    
    return await get_market_summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
