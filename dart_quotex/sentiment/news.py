"""
dart_quotex/sentiment/news.py
News sentiment integration — no paid API required.

Data sources (free, no API key)
--------------------------------
1. Forex Factory RSS     — major economic news calendar
2. Investing.com RSS     — currency and commodity news
3. FXStreet RSS          — forex news
4. DailyFX RSS           — forex education / news
5. Fallback: local cache

Sentiment pipeline
------------------
1. Fetch RSS feed (async, cached 10 minutes)
2. Parse headlines and descriptions
3. Score each item using a financial lexicon (positive/negative words)
4. Map items to relevant currency pairs (asset relevance scoring)
5. Decay older news items (half-life = 2 hours)
6. Return: sentiment score (-1 to +1), impact score (0-1), items list

Usage
-----
    from dart_quotex.sentiment.news import NewsSentimentEngine
    engine = NewsSentimentEngine()
    score, impact, items = await engine.get_sentiment("EURUSD_OTC")
    # score: +0.65 = bullish, -0.3 = slightly bearish
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Financial sentiment lexicon
# ──────────────────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    # macro
    "growth", "recovery", "expansion", "rally", "surge", "jump", "gain",
    "rise", "rising", "rose", "increase", "increased", "soar", "soaring",
    "boost", "boosted", "strong", "stronger", "strength", "beat", "beats",
    "exceed", "exceeded", "outperform", "upbeat", "optimism", "optimistic",
    "bullish", "bull", "buying", "demand", "hot", "robust", "solid",
    "surprise", "upside", "upgrade", "upgraded", "accelerate", "acceleration",
    # economic indicators
    "gdp beat", "nfp beat", "cpi lower", "unemployment fell", "rates hike",
    "hawkish", "tightening", "above expectations", "better than expected",
}

NEGATIVE_WORDS = {
    # macro
    "decline", "fall", "fell", "falling", "drop", "dropped", "slump",
    "slumped", "slide", "weakness", "weak", "weaker", "miss", "missed",
    "below", "disappoint", "disappointing", "disappoint", "recession",
    "contraction", "slowdown", "concern", "risk", "uncertainty", "uncertain",
    "worry", "worried", "bearish", "bear", "selling", "oversold",
    "downgrade", "downgraded", "cut", "cuts", "warning", "downside",
    "deficit", "debt", "crisis", "inflation", "stagflation",
    # economic
    "gdp miss", "nfp miss", "cpi higher", "unemployment rose", "rates cut",
    "dovish", "easing", "below expectations", "worse than expected",
}

HIGH_IMPACT_WORDS = {
    "fed", "fomc", "ecb", "boe", "bank of japan", "boj", "rba", "rbnz",
    "nfp", "non-farm", "cpi", "gdp", "interest rate", "rate decision",
    "payroll", "unemployment", "inflation", "central bank",
}

# ──────────────────────────────────────────────────────────────────────────────
# Currency relevance map
# ──────────────────────────────────────────────────────────────────────────────

CURRENCY_KEYWORDS: Dict[str, List[str]] = {
    "USD": ["dollar", "usd", "fed", "fomc", "us economy", "united states",
            "nfp", "payroll", "treasury", "federal reserve"],
    "EUR": ["euro", "eur", "ecb", "european", "eurozone", "germany", "france",
            "italy", "spain", "draghi", "lagarde"],
    "GBP": ["pound", "gbp", "boe", "bank of england", "britain", "uk",
            "united kingdom", "brexit", "bailey"],
    "JPY": ["yen", "jpy", "boj", "bank of japan", "japan", "japanese",
            "kuroda", "ueda"],
    "AUD": ["aud", "aussie", "rba", "reserve bank australia", "australia",
            "australian"],
    "CAD": ["cad", "loonie", "boc", "bank of canada", "canada", "canadian",
            "oil", "crude"],
    "CHF": ["chf", "franc", "snb", "swiss", "switzerland"],
    "NZD": ["nzd", "kiwi", "rbnz", "reserve bank new zealand", "new zealand"],
    "GOLD": ["gold", "xau", "precious metal", "bullion", "safe haven"],
    "OIL":  ["oil", "crude", "brent", "wti", "opec", "petroleum"],
}


# ──────────────────────────────────────────────────────────────────────────────
# News item data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title:      str
    summary:    str
    published:  float     # unix timestamp
    source:     str
    sentiment:  float = 0.0    # -1 to +1
    impact:     float = 0.0    # 0 to 1
    currencies: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# News Sentiment Engine
# ──────────────────────────────────────────────────────────────────────────────

class NewsSentimentEngine:
    """
    Async news sentiment engine.

    Parameters
    ----------
    cache_ttl   : seconds to cache fetched feeds
    decay_half  : half-life in seconds for news relevance decay
    max_items   : max news items to keep in memory
    """

    RSS_FEEDS = [
        ("ForexFactory",  "https://forexfactory.com/ff_calendar_thisweek.xml"),
        ("Investing",     "https://www.investing.com/rss/news_301.rss"),
        ("FXStreet",      "https://www.fxstreet.com/rss/news"),
        ("DailyFX",       "https://feeds.dailyfx.com/forex-market-news"),
        ("MarketWatch",   "https://feeds.marketwatch.com/marketwatch/forex"),
    ]

    def __init__(
        self,
        cache_ttl: int = 600,       # 10 minutes
        decay_half: float = 7200.0, # 2 hours half-life
        max_items: int = 100,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cache_ttl  = cache_ttl
        self.decay_half = decay_half
        self.max_items  = max_items
        self.cache_dir  = Path(cache_dir or "data/news_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._items: List[NewsItem] = []
        self._last_fetch: float = 0.0
        self._fetch_lock = asyncio.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    async def get_sentiment(
        self,
        asset: str,
        force_refresh: bool = False,
    ) -> Tuple[float, float, List[NewsItem]]:
        """
        Get sentiment score for an asset.

        Parameters
        ----------
        asset         : e.g. "EURUSD_OTC" or "GBPUSD"
        force_refresh : bypass cache

        Returns
        -------
        (sentiment_score, impact_score, relevant_items)
        sentiment_score : -1 (very bearish) to +1 (very bullish)
        impact_score    : 0 (no news) to 1 (major news)
        """
        await self._ensure_fresh(force_refresh)

        currencies = self._asset_to_currencies(asset)
        relevant   = self._filter_relevant(currencies)
        now        = time.time()

        if not relevant:
            return 0.0, 0.0, []

        # Compute time-decayed sentiment
        weighted_sentiment = 0.0
        total_weight       = 0.0

        for item in relevant:
            age_hours = (now - item.published) / 3600.0
            decay     = 0.5 ** (age_hours / (self.decay_half / 3600.0))
            w         = decay * item.impact
            weighted_sentiment += item.sentiment * w
            total_weight       += w

        if total_weight < 1e-9:
            return 0.0, 0.0, relevant

        sentiment = float(weighted_sentiment / total_weight)
        sentiment = float(max(-1.0, min(1.0, sentiment)))

        # Impact = normalised count of high-impact items (recent 2h)
        recent_high = sum(
            1 for it in relevant
            if it.impact > 0.6 and (now - it.published) < 7200
        )
        impact = float(min(1.0, recent_high / 3.0))

        return sentiment, impact, relevant[:10]

    async def refresh(self) -> int:
        """Force-refresh all feeds. Returns number of new items."""
        async with self._fetch_lock:
            new_count = 0
            for name, url in self.RSS_FEEDS:
                try:
                    items = await asyncio.wait_for(
                        self._fetch_rss(name, url), timeout=8.0
                    )
                    new_count += len(items)
                    self._items.extend(items)
                except Exception as exc:
                    logger.debug("Feed %s failed: %s", name, exc)

            # Deduplicate and sort
            seen_titles: set = set()
            unique: List[NewsItem] = []
            for it in sorted(self._items, key=lambda x: -x.published):
                sig = hashlib.md5(it.title.encode()).hexdigest()
                if sig not in seen_titles:
                    seen_titles.add(sig)
                    unique.append(it)

            self._items = unique[: self.max_items]
            self._last_fetch = time.time()
            logger.debug("News refresh: %d total items (%d new)", len(self._items), new_count)
            return new_count

    def score_text(self, text: str) -> Tuple[float, float]:
        """
        Score arbitrary text for sentiment and impact.
        Returns (sentiment, impact) each -1..1 / 0..1
        """
        text_lower = text.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
        hi  = sum(1 for w in HIGH_IMPACT_WORDS if w in text_lower)

        total = pos + neg
        sentiment = (pos - neg) / total if total > 0 else 0.0
        impact    = min(1.0, (hi * 0.3 + total * 0.05))

        return float(sentiment), float(impact)

    # ── internal ──────────────────────────────────────────────────────────────

    async def _ensure_fresh(self, force: bool) -> None:
        age = time.time() - self._last_fetch
        if force or age > self.cache_ttl or not self._items:
            # Try to load from cache first (offline)
            loaded = self._load_cache()
            if not loaded or force:
                try:
                    await self.refresh()
                except Exception as exc:
                    logger.warning("News refresh failed: %s — using cache", exc)
                    if not self._items:
                        self._load_cache()

    async def _fetch_rss(self, name: str, url: str) -> List[NewsItem]:
        """Fetch and parse an RSS feed."""
        import aiohttp
        items = []
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8)
            ) as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    if resp.status != 200:
                        return []
                    text = await resp.text(errors="replace")

            root = ET.fromstring(text)
            channel = root.find("channel")
            if channel is None:
                channel = root

            for entry in channel.findall("item")[:20]:
                title   = (entry.findtext("title") or "").strip()
                summary = (entry.findtext("description") or "").strip()
                pub_str = entry.findtext("pubDate") or ""
                try:
                    import email.utils
                    pub_ts = float(email.utils.mktime_tz(
                        email.utils.parsedate_tz(pub_str)
                    ))
                except Exception:
                    pub_ts = time.time() - 3600  # assume 1h ago

                sentiment, impact = self.score_text(title + " " + summary)
                currencies = self._extract_currencies(title + " " + summary)

                items.append(NewsItem(
                    title=title, summary=summary[:200],
                    published=pub_ts, source=name,
                    sentiment=sentiment, impact=impact,
                    currencies=currencies,
                ))

            self._save_cache(items, name)
        except Exception as exc:
            logger.debug("RSS fetch %s: %s", name, exc)
            # Try loading from cache
            cached = self._load_cache_source(name)
            if cached:
                items = cached

        return items

    def _filter_relevant(self, currencies: List[str]) -> List[NewsItem]:
        """Return items relevant to at least one of the given currencies."""
        relevant = []
        for item in self._items:
            if any(ccy in item.currencies for ccy in currencies):
                relevant.append(item)
        return sorted(relevant, key=lambda x: -x.published)

    @staticmethod
    def _asset_to_currencies(asset: str) -> List[str]:
        """Extract currency codes from an asset name."""
        clean = asset.upper().replace("_OTC", "").replace("_otc", "")
        currencies = []
        for ccy in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
            if ccy in clean:
                currencies.append(ccy)
        if "XAU" in clean:
            currencies.extend(["GOLD", "USD"])
        if "XTI" in clean or "XBR" in clean:
            currencies.extend(["OIL", "USD"])
        return currencies

    @staticmethod
    def _extract_currencies(text: str) -> List[str]:
        """Identify which currencies are mentioned in text."""
        text_lower = text.lower()
        found = []
        for ccy, keywords in CURRENCY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.append(ccy)
        return found

    # ── disk cache ────────────────────────────────────────────────────────────

    def _save_cache(self, items: List[NewsItem], source: str) -> None:
        try:
            fp = self.cache_dir / f"news_{source}.json"
            data = [
                {
                    "title": it.title, "summary": it.summary,
                    "published": it.published, "source": it.source,
                    "sentiment": it.sentiment, "impact": it.impact,
                    "currencies": it.currencies,
                }
                for it in items
            ]
            fp.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    def _load_cache_source(self, source: str) -> List[NewsItem]:
        try:
            fp = self.cache_dir / f"news_{source}.json"
            if not fp.exists():
                return []
            data = json.loads(fp.read_text(encoding="utf-8"))
            return [
                NewsItem(
                    title=d["title"], summary=d.get("summary", ""),
                    published=d["published"], source=d["source"],
                    sentiment=d.get("sentiment", 0), impact=d.get("impact", 0),
                    currencies=d.get("currencies", []),
                )
                for d in data
            ]
        except Exception:
            return []

    def _load_cache(self) -> bool:
        items = []
        for name, _ in self.RSS_FEEDS:
            items.extend(self._load_cache_source(name))
        if items:
            self._items = sorted(items, key=lambda x: -x.published)[: self.max_items]
            logger.debug("Loaded %d items from disk cache", len(self._items))
            return True
        return False
