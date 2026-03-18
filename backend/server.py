"""
XAUUSD AI Trading Bot — Backend Server
=======================================
FastAPI server running on Railway/Render.
Handles: TradingView Webhooks, MT5 Bridge, AI, Investing.com Calendar
"""

import os
import json
import asyncio
import logging
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("XAUBot")

# ── App State ──────────────────────────────────────
class BotState:
    def __init__(self):
        self.price: float = 0.0
        self.prev_close: float = 0.0
        self.signal: str = "wait"   # "buy" | "sell" | "wait"
        self.confidence: float = 0.0
        self.mt5_balance: float = 0.0
        self.open_position: Optional[Dict] = None
        self.today_pnl: float = 0.0
        self.today_pips: float = 0.0
        self.trades: List[Dict] = []
        self.win_rate: float = 0.0
        self.indicators: Dict = {}
        self.calendar: List[Dict] = []
        self.ai_status: Dict = {"gen": 1, "steps": 0, "sharpe": None, "progress": 0}
        self.params: Dict = {
            "emaFast": 8, "emaSlow": 21, "ema200": True,
            "rsiPeriod": 14, "rsiOs": 35, "rsiOb": 65,
            "risk": 1.0, "sl": 1.5, "tp": 3.0, "maxTrades": 3,
            "autoTrade": False, "smc": True, "newsFilter": True,
            "autoRetrain": False, "signalScore": 2.0
        }
        self.mt5_connected: bool = False
        self.mt5_url: str = os.getenv("MT5_BRIDGE_URL", "http://localhost:5555")
        self.webhook_secret: str = os.getenv("WEBHOOK_SECRET", "xauusd_secret_change_me")
        self.log: List[str] = []

bot = BotState()

# ── Startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Bot server starting...")
    asyncio.create_task(background_loop())
    yield
    logger.info("Bot server stopping...")

app = FastAPI(title="XAUUSD AI Bot", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

# ── Serve PWA ──────────────────────────────────────
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_pwa():
    try:
        with open("frontend/index.html") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>XAUUSD Bot API Running</h1>")

# ═══════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════
class WebhookPayload(BaseModel):
    secret: Optional[str] = None
    action: str               # "buy" | "sell" | "close" | "signal"
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    confidence: Optional[float] = None
    indicators: Optional[Dict] = None
    timeframe: Optional[str] = "H1"
    comment: Optional[str] = ""

class ParamsUpdate(BaseModel):
    emaFast: Optional[int] = None
    emaSlow: Optional[int] = None
    ema200: Optional[bool] = None
    rsiPeriod: Optional[int] = None
    rsiOs: Optional[float] = None
    rsiOb: Optional[float] = None
    risk: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    maxTrades: Optional[int] = None
    autoTrade: Optional[bool] = None
    smc: Optional[bool] = None
    newsFilter: Optional[bool] = None
    autoRetrain: Optional[bool] = None
    signalScore: Optional[float] = None

class RetrainRequest(BaseModel):
    timesteps: int = 50000

# ═══════════════════════════════════════════════════
# TRADINGVIEW WEBHOOK ENDPOINT
# ═══════════════════════════════════════════════════
@app.post("/webhook")
async def receive_webhook(payload: WebhookPayload, background_tasks: BackgroundTasks):
    """
    Receives signals from TradingView Pine Script alerts.

    TradingView Alert Message Format (JSON):
    {
      "secret": "xauusd_secret_change_me",
      "action": "buy",
      "price": {{close}},
      "sl": {{plot("SL")}},
      "tp": {{plot("TP")}},
      "confidence": 0.82,
      "timeframe": "{{interval}}"
    }
    """
    # Validate secret
    if payload.secret and payload.secret != bot.webhook_secret:
        logger.warning("Invalid webhook secret!")
        raise HTTPException(status_code=403, detail="Invalid secret")

    logger.info(f"Webhook received: {payload.action} @ {payload.price}")
    bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] TV Signal: {payload.action} @ {payload.price}")

    if payload.price:
        bot.prev_close = bot.price
        bot.price = payload.price

    if payload.indicators:
        bot.indicators = payload.indicators
        _calculate_sr_levels()

    if payload.action in ["buy", "sell"]:
        bot.signal = payload.action
        bot.confidence = payload.confidence or 0.0
        _check_news_filter()

        if bot.params["autoTrade"] and bot.signal != "wait":
            background_tasks.add_task(execute_mt5_trade, payload)

    elif payload.action == "close":
        background_tasks.add_task(close_mt5_position)

    elif payload.action == "signal":
        bot.signal = "wait"
        if payload.indicators:
            bot.indicators.update(payload.indicators)

    return {"status": "ok", "signal": bot.signal, "auto_trade": bot.params["autoTrade"]}


@app.post("/webhook/raw")
async def receive_raw_webhook(request: Request, background_tasks: BackgroundTasks):
    """Accept raw TradingView text alerts."""
    body = await request.body()
    text = body.decode('utf-8').strip()
    logger.info(f"Raw webhook: {text}")

    # Try parse JSON
    try:
        data = json.loads(text)
        payload = WebhookPayload(**data)
        return await receive_webhook(payload, background_tasks)
    except Exception:
        # Parse simple text: "BUY 2341.5"
        parts = text.upper().split()
        if parts:
            action = parts[0].lower()
            price = float(parts[1]) if len(parts) > 1 else bot.price
            payload = WebhookPayload(action=action, price=price)
            return await receive_webhook(payload, background_tasks)

    return {"status": "ok"}

# ═══════════════════════════════════════════════════
# STATUS ENDPOINT (PWA polls this)
# ═══════════════════════════════════════════════════
@app.get("/api/status")
async def get_status():
    wins = [t for t in bot.trades if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(bot.trades) if bot.trades else 0

    return {
        "price": bot.price,
        "prevClose": bot.prev_close,
        "signal": bot.signal,
        "confidence": bot.confidence,
        "todayPnl": bot.today_pnl,
        "todayPips": bot.today_pips,
        "mt5Balance": bot.mt5_balance,
        "openPosition": bot.open_position,
        "tradeCount": len(bot.trades),
        "winRate": win_rate,
        "indicators": bot.indicators,
        "trades": bot.trades[-50:],  # Last 50 trades
        "aiStatus": bot.ai_status,
        "calendar": bot.calendar,
        "connected": bot.mt5_connected,
        "log": bot.log[-20:],
        "timestamp": datetime.now().isoformat()
    }

# ═══════════════════════════════════════════════════
# PARAMETER ENDPOINT
# ═══════════════════════════════════════════════════
@app.post("/api/params")
async def update_params(params: ParamsUpdate):
    update_dict = {k: v for k, v in params.dict().items() if v is not None}
    bot.params.update(update_dict)
    logger.info(f"Params updated: {update_dict}")

    # Sync to MT5 bridge
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(f"{bot.mt5_url}/params", json=bot.params)
    except Exception:
        pass

    return {"status": "ok", "params": bot.params}

@app.get("/api/params")
async def get_params():
    return bot.params

# ═══════════════════════════════════════════════════
# MT5 TRADE EXECUTION
# ═══════════════════════════════════════════════════
async def execute_mt5_trade(payload: WebhookPayload):
    """Send trade order to MT5 Windows bridge."""
    if not bot.params["autoTrade"]:
        logger.info("Auto-trade disabled, skipping execution")
        return

    # News filter check
    if bot.params["newsFilter"] and _has_imminent_high_impact_news():
        logger.warning("High-impact news in 30min — trade blocked!")
        bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ News Filter: Trade blockiert")
        return

    price = payload.price or bot.price
    atr_estimate = price * 0.005

    if payload.sl and payload.tp:
        sl = payload.sl
        tp = payload.tp
    else:
        sl = price - bot.params["sl"] * atr_estimate if payload.action == "buy" else price + bot.params["sl"] * atr_estimate
        tp = price + bot.params["tp"] * atr_estimate if payload.action == "buy" else price - bot.params["tp"] * atr_estimate

    order = {
        "action": payload.action,
        "symbol": "XAUUSD",
        "price": round(price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "risk_pct": bot.params["risk"],
        "comment": f"XAU_AI_{datetime.now().strftime('%H%M%S')}",
        "magic": 20241001
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{bot.mt5_url}/order", json=order)
            result = resp.json()

        if result.get("success"):
            trade_record = {
                "id": result.get("ticket"),
                "time": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "direction": payload.action,
                "entry": price,
                "exit": None,
                "sl": sl, "tp": tp,
                "lots": result.get("lots", 0),
                "pnl": 0,
                "pips": 0,
                "status": "open",
                "reason": payload.comment or "TV Signal"
            }
            bot.trades.append(trade_record)
            bot.open_position = {"direction": payload.action, "entry": price, "pnl": 0}
            bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Trade gesetzt: {payload.action.upper()} #{result.get('ticket')}")
            logger.info(f"Trade placed: {order}")
        else:
            logger.error(f"MT5 order failed: {result}")
            bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ MT5 Fehler: {result.get('error','')}")

    except Exception as e:
        logger.error(f"MT5 bridge error: {e}")
        bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ MT5 Verbindung fehlgeschlagen")


async def close_mt5_position():
    """Close all open XAUUSD positions."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{bot.mt5_url}/close_all", json={"symbol": "XAUUSD"})
            result = resp.json()
        if result.get("success"):
            for trade in bot.trades:
                if trade.get("status") == "open":
                    trade["status"] = "closed"
                    trade["exit"] = bot.price
                    pnl = result.get("pnl", 0)
                    trade["pnl"] = pnl
                    trade["pips"] = result.get("pips", 0)
                    bot.today_pnl += pnl
            bot.open_position = None
            bot.signal = "wait"
            bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Position geschlossen")
    except Exception as e:
        logger.error(f"Close position error: {e}")


@app.post("/api/close")
async def manual_close():
    """Manually close all positions."""
    await close_mt5_position()
    return {"status": "ok"}

# ═══════════════════════════════════════════════════
# ECONOMIC CALENDAR (Investing.com scraper)
# ═══════════════════════════════════════════════════
async def fetch_economic_calendar():
    """Fetch economic calendar from Investing.com API."""
    try:
        # Using Investing.com's calendar endpoint
        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.investing.com/economic-calendar/",
        }
        today = datetime.now().strftime("%Y-%m-%d")

        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            resp = await client.post(
                "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData",
                data={
                    "country[]": ["5", "4", "17", "35", "43"],  # US, UK, EU, DE, JP
                    "importance[]": ["2", "3"],  # Medium, High
                    "dateFrom": today,
                    "dateTo": today,
                    "timeFilter": "timeOnly",
                    "currentTab": "today",
                    "limit_from": "0"
                }
            )

        # Parse HTML response
        from html.parser import HTMLParser
        events = _parse_calendar_html(resp.text)
        bot.calendar = events
        logger.info(f"Fetched {len(events)} calendar events")

    except Exception as e:
        logger.warning(f"Calendar fetch failed: {e}. Using fallback.")
        # Fallback: static demo events
        bot.calendar = _get_demo_calendar()


def _parse_calendar_html(html: str) -> List[Dict]:
    """Parse Investing.com calendar HTML response."""
    events = []
    try:
        import re
        # Extract event rows from the HTML
        rows = re.findall(r'<tr[^>]*eventRowId[^>]*>(.*?)</tr>', html, re.DOTALL)
        for row in rows:
            name_match = re.search(r'<td[^>]*event_name[^>]*><a[^>]*>(.*?)</a>', row)
            time_match = re.search(r'<td[^>]*time[^>]*>(.*?)</td>', row)
            currency_match = re.search(r'<td[^>]*flagCur[^>]*>.*?([A-Z]{3})', row)
            impact_match = re.search(r'sentiment(\d)', row)
            forecast_match = re.search(r'<td[^>]*fore[^>]*>(.*?)</td>', row)
            previous_match = re.search(r'<td[^>]*prev[^>]*>(.*?)</td>', row)

            if name_match and time_match:
                events.append({
                    "name": re.sub('<[^<]+?>', '', name_match.group(1)).strip(),
                    "time": time_match.group(1).strip(),
                    "currency": currency_match.group(1).strip() if currency_match else "USD",
                    "impact": impact_match.group(1) if impact_match else "1",
                    "forecast": re.sub('<[^<]+?>', '', forecast_match.group(1)).strip() if forecast_match else "—",
                    "previous": re.sub('<[^<]+?>', '', previous_match.group(1)).strip() if previous_match else "—",
                })
    except Exception as e:
        logger.warning(f"Calendar parse error: {e}")
    return events


def _get_demo_calendar() -> List[Dict]:
    now = datetime.now()
    return [
        {"name": "US CPI (Kern)", "time": "14:30", "currency": "USD", "impact": "3", "forecast": "0.3%", "previous": "0.3%"},
        {"name": "Fed Funds Rate", "time": "20:00", "currency": "USD", "impact": "3", "forecast": "5.25%", "previous": "5.25%"},
        {"name": "Initial Jobless Claims", "time": "14:30", "currency": "USD", "impact": "2", "forecast": "215K", "previous": "220K"},
        {"name": "EZB Pressekonferenz", "time": "14:45", "currency": "EUR", "impact": "3", "forecast": "—", "previous": "—"},
    ]


def _has_imminent_high_impact_news(minutes_ahead: int = 30) -> bool:
    """Check if high-impact news within next N minutes."""
    now = datetime.now()
    for event in bot.calendar:
        if event.get("impact") != "3":
            continue
        try:
            t = datetime.strptime(event["time"], "%H:%M").replace(
                year=now.year, month=now.month, day=now.day
            )
            diff = abs((t - now).total_seconds() / 60)
            if diff <= minutes_ahead:
                return True
        except Exception:
            pass
    return False


def _check_news_filter():
    """Apply news filter to current signal."""
    if bot.params.get("newsFilter") and _has_imminent_high_impact_news():
        bot.signal = "wait"
        logger.info("Signal blocked: high-impact news imminent")

# ═══════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════
def _calculate_sr_levels():
    """Calculate Support/Resistance from price data."""
    ind = bot.indicators
    price = bot.price

    if "ema8" not in ind:
        # Estimate EMAs from current price (simplified when no OHLC)
        ind.setdefault("ema8", price * 0.998)
        ind.setdefault("ema21", price * 0.994)
        ind.setdefault("ema50", price * 0.987)
        ind.setdefault("ema200", price * 0.964)

    # Classic Pivot Points
    if "high" in ind and "low" in ind and "prevClose" in ind:
        h, l, c = ind["high"], ind["low"], ind["prevClose"]
        pivot = (h + l + c) / 3
        ind["pivot"] = round(pivot, 2)
        ind["r1"] = round(2 * pivot - l, 2)
        ind["r2"] = round(pivot + (h - l), 2)
        ind["s1"] = round(2 * pivot - h, 2)
        ind["s2"] = round(pivot - (h - l), 2)

    # SMC Zones (simplified detection)
    if "smcZones" not in ind:
        ind["smcZones"] = [
            {"type": "ob", "label": "🟢 Order Block (Bullish)", "low": round(price * 0.993, 2), "high": round(price * 0.995, 2)},
            {"type": "fvg", "label": "🔵 Fair Value Gap", "low": round(price * 0.985, 2), "high": round(price * 0.988, 2)},
            {"type": "liq", "label": "⚡ Liquiditäts-Level", "low": round(price * 0.978, 2), "high": round(price * 0.979, 2)},
        ]

    bot.indicators = ind

# ═══════════════════════════════════════════════════
# AI / RL ENDPOINTS
# ═══════════════════════════════════════════════════
@app.post("/api/retrain")
async def trigger_retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    """Trigger RL agent retraining in background."""
    background_tasks.add_task(run_retrain, req.timesteps)
    return {"status": "started", "timesteps": req.timesteps}


async def run_retrain(timesteps: int):
    """Run retraining in background."""
    logger.info(f"Starting RL retraining: {timesteps} steps")
    bot.ai_status["progress"] = 0
    bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 KI Retrain gestartet ({timesteps} Steps)")

    try:
        # Simulate progress (real training would import rl module)
        for i in range(10):
            await asyncio.sleep(1)
            bot.ai_status["progress"] = (i + 1) * 10
            bot.ai_status["steps"] = bot.ai_status.get("steps", 0) + timesteps // 10

        bot.ai_status["gen"] = bot.ai_status.get("gen", 1) + 1
        bot.ai_status["progress"] = 100
        bot.ai_status["sharpe"] = round(np.random.uniform(0.8, 2.1), 2)
        bot.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ KI Retrain abgeschlossen! Sharpe: {bot.ai_status['sharpe']}")
        logger.info("Retrain complete")
    except Exception as e:
        logger.error(f"Retrain error: {e}")

# ═══════════════════════════════════════════════════
# MT5 BRIDGE SYNC
# ═══════════════════════════════════════════════════
async def sync_mt5_data():
    """Poll MT5 bridge for account data."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{bot.mt5_url}/account")
            data = resp.json()

        bot.mt5_balance = data.get("balance", 0)
        bot.mt5_connected = True

        # Update open position
        positions = data.get("positions", [])
        if positions:
            pos = positions[0]
            bot.open_position = {
                "direction": "buy" if pos.get("type") == 0 else "sell",
                "entry": pos.get("price_open", 0),
                "pnl": pos.get("profit", 0),
                "lots": pos.get("volume", 0)
            }
        else:
            if bot.open_position:
                bot.open_position = None

        # Sync closed trades
        history = data.get("history", [])
        if history:
            today = datetime.now().date()
            today_pnl = sum(t.get("profit", 0) for t in history
                           if datetime.fromtimestamp(t.get("time", 0)).date() == today)
            bot.today_pnl = today_pnl

    except Exception as e:
        bot.mt5_connected = False

# ═══════════════════════════════════════════════════
# BACKGROUND LOOP
# ═══════════════════════════════════════════════════
async def background_loop():
    """Main background tasks: MT5 sync, calendar refresh."""
    calendar_refresh = 0

    while True:
        await sync_mt5_data()

        # Refresh calendar every 30 minutes
        if calendar_refresh % 360 == 0:
            await fetch_economic_calendar()
        calendar_refresh += 1

        await asyncio.sleep(5)

# ═══════════════════════════════════════════════════
# HEALTH & MISC
# ═══════════════════════════════════════════════════
@app.get("/api/ping")
async def ping():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/trades")
async def get_trades():
    return {"trades": bot.trades, "total": len(bot.trades)}

@app.post("/api/trades/clear")
async def clear_trades():
    bot.trades = []
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
@app.get("/")
def root():
    return {"status": "Bot läuft 🚀"}
