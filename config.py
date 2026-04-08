"""NEXUS Configuration."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "nexus.db"

# Google AI — Primary Project
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Google AI — Secondary Project (Phase 2: Multi-Key Rotation for true physical redundancy)
GEMINI_API_KEY_SECONDARY = os.getenv("GEMINI_API_KEY_SECONDARY", "")
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
GEMINI_PRO_MODEL = "gemini-2.5-flash"  # same model — free tier has separate per-model quotas

# Tavily Web Search (optional — enables web_search + LinkedIn tools)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Agent
MAX_AGENT_STEPS = 20
AGENT_TIMEOUT_SECONDS = 120
MAX_PLAN_STEPS = 3              # Hard cap on plan steps to limit total Gemini calls per request

# Rate Limiting (Gemini Free Tier: 15 RPM, 1M TPM, 1500 RPD)
GEMINI_MAX_CONCURRENT = 1           # Semaphore: only 1 Gemini call at a time
GEMINI_MIN_CALL_SPACING_S = 6.0     # Minimum seconds between API calls (safe for 2.5 Free Tier)
GEMINI_429_BASE_DELAY_S = 10.0      # Initial backoff on 429 (before reading retryDelay)
GEMINI_429_MAX_DELAY_S = 60.0       # Max backoff cap on 429
GEMINI_JITTER_S = 0.5               # Random jitter added to each call spacing

# Circuit Breaker (Phase 1: Graceful Degradation)
CIRCUIT_BREAKER_COOLDOWN_S = 300.0  # 5 minutes — after all providers exhaust, skip cascade
CIRCUIT_BREAKER_TRIP_THRESHOLD = 2  # Trip after N consecutive full-cascade failures

# ── Logging Setup ──────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging():
    """Configure structured logging for production."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


setup_logging()
