from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
import logging
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Noema Financial API - Polygon.io + Claude AI")

# CORS - Allow your frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
import os

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "IggApxeIGzJ7fPmwYtS0YOhMgpOJZ8c1")
POLYGON_BASE_URL = "https://api.polygon.io"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Validate that Claude API key is set
if not CLAUDE_API_KEY:
    logger.warning("CLAUDE_API_KEY environment variable not set! AI summaries will not work.")
    CLAUDE_API_KEY = None

# Cache for quotes
quote_cache: Dict[str, dict] = {}
last_update: Optional[datetime] = None
update_in_progress = False

# Cache for AI summary
summary_cache = {
    "text": None,
    "timestamp": None
}

# Default symbols to track
DEFAULT_WATCHLIST = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMD']
MARKET_INDICES = ['SPY', 'QQQ', 'UVXY']
SUGGESTED_STOCKS = ['TSM', 'SMCI', 'ASML', 'DELL', 'PLTR', 'COIN']

# Bitcoin ticker on Polygon
CRYPTO_SYMBOLS = ['X:BTCUSD']  # Polygon crypto format


def is_market_hours() -> bool:
    """Check if it's during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)"""
    now = datetime.now()
    
    # Check if it's a weekday (0 = Monday, 6 = Sunday)
    if now.weekday() > 4:  # Saturday or Sunday
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    # For simplicity, we'll keep updating during extended hours too
    market_open = time(4, 0)  # 4 AM ET (pre-market)
    market_close = time(20, 0)  # 8 PM ET (after-hours)
    
    current_time = now.time()
    return market_open <= current_time <= market_close


async def fetch_stock_quote(symbol: str) -> Optional[dict]:
    """Fetch a single stock quote from Polygon (previous close - 15min delayed)"""
    try:
        async with httpx.AsyncClient() as client:
            # Use snapshot endpoint for current quote
            response = await client.get(
                f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
                params={"apiKey": POLYGON_API_KEY},
                timeout=10.0
            )
            
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code} for {symbol}")
                return None
            
            data = response.json()
            
            if data.get("status") != "OK" or "ticker" not in data:
                logger.error(f"Invalid response for {symbol}")
                return None
            
            ticker = data["ticker"]
            
            # Get current day data
            day = ticker.get("day", {})
            prev_close = ticker.get("prevDay", {}).get("c", 0)
            
            current_price = day.get("c", prev_close)  # Close price or last price
            
            # Calculate change
            change = current_price - prev_close if prev_close else 0
            change_percent = (change / prev_close * 100) if prev_close else 0
            
            return {
                "symbol": ticker["ticker"],
                "price": float(current_price),
                "change": float(change),
                "changePercent": float(change_percent),
                "name": ticker.get("name", symbol),
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        return None


async def fetch_crypto_quote(symbol: str) -> Optional[dict]:
    """Fetch crypto quote from Polygon"""
    try:
        async with httpx.AsyncClient() as client:
            # Polygon crypto format: X:BTCUSD
            response = await client.get(
                f"{POLYGON_BASE_URL}/v2/snapshot/locale/global/markets/crypto/tickers/{symbol}",
                params={"apiKey": POLYGON_API_KEY},
                timeout=10.0
            )
            
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code} for {symbol}")
                return None
            
            data = response.json()
            
            if data.get("status") != "OK" or "ticker" not in data:
                logger.error(f"Invalid crypto response for {symbol}")
                return None
            
            ticker = data["ticker"]
            day = ticker.get("day", {})
            prev_close = ticker.get("prevDay", {}).get("c", 0)
            
            current_price = day.get("c", prev_close)
            change = current_price - prev_close if prev_close else 0
            change_percent = (change / prev_close * 100) if prev_close else 0
            
            # Convert X:BTCUSD to BTC/USD for display
            display_symbol = symbol.replace("X:", "").replace("USD", "/USD")
            
            return {
                "symbol": display_symbol,
                "price": float(current_price),
                "change": float(change),
                "changePercent": float(change_percent),
                "name": "Bitcoin",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching crypto {symbol}: {str(e)}")
        return None


async def fetch_batch_quotes(symbols: List[str]) -> Dict[str, dict]:
    """Fetch multiple quotes concurrently (Polygon doesn't have true batch, so we use concurrent requests)"""
    
    # Separate stocks and crypto
    stock_symbols = [s for s in symbols if not s.startswith("X:")]
    crypto_symbols = [s for s in symbols if s.startswith("X:")]
    
    # Fetch stocks
    stock_tasks = [fetch_stock_quote(symbol) for symbol in stock_symbols]
    stock_results = await asyncio.gather(*stock_tasks)
    
    # Fetch crypto
    crypto_tasks = [fetch_crypto_quote(symbol) for symbol in crypto_symbols]
    crypto_results = await asyncio.gather(*crypto_tasks)
    
    # Combine results
    all_results = stock_results + crypto_results
    
    # Build dictionary of successful quotes
    quotes = {}
    for result in all_results:
        if result:
            quotes[result["symbol"]] = result
    
    return quotes


async def update_all_quotes():
    """Update all quotes in cache"""
    global quote_cache, last_update, update_in_progress
    
    if update_in_progress:
        logger.info("Update already in progress, skipping...")
        return
    
    update_in_progress = True
    logger.info("Updating quotes from Polygon.io...")
    
    try:
        # Combine all symbols we need to track
        all_symbols = list(set(DEFAULT_WATCHLIST + MARKET_INDICES + SUGGESTED_STOCKS + CRYPTO_SYMBOLS))
        
        # Fetch all quotes concurrently
        new_quotes = await fetch_batch_quotes(all_symbols)
        
        if new_quotes:
            quote_cache.update(new_quotes)
            last_update = datetime.now()
            logger.info(f"Updated {len(new_quotes)} quotes from Polygon.io")
        else:
            logger.warning("No quotes fetched from Polygon")
            
    except Exception as e:
        logger.error(f"Error updating quotes: {str(e)}")
    finally:
        update_in_progress = False


# ==================== AI MARKET SUMMARY FUNCTIONS ====================

async def fetch_polygon_news(limit: int = 10) -> List[dict]:
    """Fetch top financial news from Polygon"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{POLYGON_BASE_URL}/v2/reference/news",
                params={
                    "limit": limit,
                    "order": "desc",
                    "sort": "published_utc",
                    "apiKey": POLYGON_API_KEY
                },
                timeout=15.0
            )
            
            if response.status_code != 200:
                logger.error(f"Polygon news API error: {response.status_code}")
                return []
            
            data = response.json()
            
            if data.get("status") == "OK" and "results" in data:
                return data["results"]
            
            return []
            
    except Exception as e:
        logger.error(f"Error fetching Polygon news: {str(e)}")
        return []


def format_headlines(articles: List[dict]) -> str:
    """Format news articles into headlines for Claude"""
    headlines = []
    for article in articles[:10]:  # Top 10 only
        title = article.get("title", "")
        publisher = article.get("publisher", {}).get("name", "Unknown")
        headlines.append(f"- {title} ({publisher})")
    
    return "\n".join(headlines)


async def generate_ai_summary() -> str:
    """Generate AI market summary using Claude"""
    try:
        # Check if API key is configured
        if not CLAUDE_API_KEY:
            logger.error("Claude API key not configured")
            return "AI summary unavailable - API key not configured. Please contact support."
        
        # Fetch news from Polygon
        logger.info("Fetching news from Polygon for AI summary...")
        articles = await fetch_polygon_news(limit=10)
        
        if not articles:
            logger.warning("No articles fetched from Polygon")
            return "Market summary temporarily unavailable. Please check back soon."
        
        # Format headlines
        headlines_text = format_headlines(articles)
        logger.info(f"Formatted {len(articles)} headlines for Claude")
        
        # Create Claude client
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        
        # Generate summary
        logger.info("Sending headlines to Claude API...")
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Summarize these top financial headlines into 2-3 professional sentences for retail investors. Focus on major market moves, key catalysts, and overall sentiment:

{headlines_text}"""
            }]
        )
        
        summary = message.content[0].text
        logger.info("AI summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        return "Market summary temporarily unavailable. Please check back soon."


def summary_cache_expired() -> bool:
    """Check if summary cache is older than 4 hours"""
    if not summary_cache["timestamp"]:
        return True
    
    age = datetime.now() - summary_cache["timestamp"]
    return age > timedelta(hours=4)


# ==================== BACKGROUND TASKS ====================

async def background_updater():
    """Background task to update quotes"""
    while True:
        if is_market_hours():
            await update_all_quotes()
            # Update every 60 seconds during market hours (15-min delayed is fine)
            await asyncio.sleep(60)
        else:
            # Outside market hours, update every 5 minutes
            await update_all_quotes()
            await asyncio.sleep(300)


@app.on_event("startup")
async def startup_event():
    """Initialize cache and start background updater"""
    logger.info("Starting Noema Financial API with Polygon.io + Claude AI...")
    
    # Initial quote fetch
    await update_all_quotes()
    
    # Start background updater
    asyncio.create_task(background_updater())
    
    logger.info("API started successfully!")


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Noema Financial API",
        "provider": "Polygon.io (15-min delayed) + Claude AI",
        "last_update": last_update.isoformat() if last_update else None,
        "cached_symbols": len(quote_cache),
        "ai_summary_cached": summary_cache["text"] is not None
    }


@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Get a single quote"""
    symbol = symbol.upper()
    
    # Check if it's crypto
    if symbol == "BTC/USD":
        symbol = "X:BTCUSD"
    
    if symbol in quote_cache:
        return quote_cache[symbol]
    
    # If not in cache, fetch it directly
    if symbol.startswith("X:"):
        quote = await fetch_crypto_quote(symbol)
    else:
        quote = await fetch_stock_quote(symbol)
        
    if quote:
        quote_cache[quote["symbol"]] = quote
        return quote
    
    raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")


@app.get("/api/quotes")
async def get_quotes(symbols: str):
    """Get multiple quotes (comma-separated symbols)"""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    # Convert BTC/USD to Polygon format
    symbol_list = ["X:BTCUSD" if s == "BTC/USD" else s for s in symbol_list]
    
    result = {}
    missing = []
    
    # Get from cache first
    for symbol in symbol_list:
        # Try both formats for display
        display_symbol = symbol.replace("X:", "").replace("USD", "/USD") if symbol.startswith("X:") else symbol
        
        if symbol in quote_cache:
            result[display_symbol] = quote_cache[symbol]
        elif display_symbol in quote_cache:
            result[display_symbol] = quote_cache[display_symbol]
        else:
            missing.append(symbol)
    
    # Fetch missing quotes
    if missing:
        new_quotes = await fetch_batch_quotes(missing)
        result.update(new_quotes)
        quote_cache.update(new_quotes)
    
    return {
        "quotes": result,
        "last_update": last_update.isoformat() if last_update else None
    }


@app.get("/api/watchlist")
async def get_watchlist(symbols: Optional[str] = None):
    """Get quotes for a watchlist"""
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = DEFAULT_WATCHLIST
    
    result = []
    for symbol in symbol_list:
        if symbol in quote_cache:
            result.append(quote_cache[symbol])
    
    return {
        "quotes": result,
        "last_update": last_update.isoformat() if last_update else None
    }


@app.get("/api/market-indices")
async def get_market_indices():
    """Get market indices (SPY, QQQ, UVXY) + Bitcoin"""
    result = {}
    
    # Get stock indices
    for symbol in MARKET_INDICES:
        if symbol in quote_cache:
            result[symbol] = quote_cache[symbol]
    
    # Get Bitcoin
    if "BTC/USD" in quote_cache:
        result["BTC/USD"] = quote_cache["BTC/USD"]
    
    return {
        "indices": result,
        "last_update": last_update.isoformat() if last_update else None
    }


@app.get("/api/suggested")
async def get_suggested():
    """Get suggested stocks"""
    result = []
    for symbol in SUGGESTED_STOCKS:
        if symbol in quote_cache:
            result.append(quote_cache[symbol])
    
    return {
        "quotes": result,
        "last_update": last_update.isoformat() if last_update else None
    }


@app.get("/api/market-summary")
async def get_market_summary():
    """Get AI-generated market summary"""
    try:
        # Check if cache is expired
        if summary_cache_expired():
            logger.info("Summary cache expired, generating new summary...")
            summary = await generate_ai_summary()
            summary_cache["text"] = summary
            summary_cache["timestamp"] = datetime.now()
        else:
            logger.info("Returning cached summary")
        
        return {
            "summary": summary_cache["text"],
            "updated": summary_cache["timestamp"].isoformat() if summary_cache["timestamp"] else None,
            "source": "Claude AI + Polygon News"
        }
        
    except Exception as e:
        logger.error(f"Error in market summary endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating market summary")


@app.post("/api/refresh")
async def force_refresh():
    """Force refresh all quotes (useful for testing)"""
    await update_all_quotes()
    return {
        "status": "refreshed",
        "last_update": last_update.isoformat() if last_update else None,
        "cached_symbols": len(quote_cache)
    }


@app.post("/api/refresh-summary")
async def force_refresh_summary():
    """Force refresh AI summary (useful for testing)"""
    summary = await generate_ai_summary()
    summary_cache["text"] = summary
    summary_cache["timestamp"] = datetime.now()
    
    return {
        "status": "refreshed",
        "summary": summary,
        "updated": summary_cache["timestamp"].isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
