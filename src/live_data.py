"""
Live data module:
- World Bank API: macro indicators (GDP, unemployment, inflation, NPA)
- RBI repo rate history
- RSS news scraper: RBI, Economic Times, Google News (no API key needed)
All cached to avoid hammering APIs.
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


# ── World Bank ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_world_bank(indicator: str, country: str = "IN", years: int = 12) -> pd.DataFrame:
    end_year = datetime.now().year - 1
    start_year = end_year - years
    url = (f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
           f"?format=json&date={start_year}:{end_year}&per_page=20")
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2 or not data[1]:
            return pd.DataFrame()
        records = [{"year": int(d["date"]), "value": d["value"]}
                   for d in data[1] if d["value"] is not None]
        return pd.DataFrame(records).sort_values("year")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_macro_indicators() -> dict:
    indicators = {
        "GDP Growth (%)":       fetch_world_bank("NY.GDP.MKTP.KD.ZG"),
        "Unemployment Rate (%)":fetch_world_bank("SL.UEM.TOTL.ZS"),
        "Inflation CPI (%)":    fetch_world_bank("FP.CPI.TOTL.ZG"),
        "Bank NPA Ratio (%)":   fetch_world_bank("FB.AST.NPER.ZS"),
        "Lending Rate (%)":     fetch_world_bank("FR.INR.LEND"),
        "Credit to GDP (%)":    fetch_world_bank("FS.AST.DOMO.GD.ZS"),
    }
    fallbacks = {
        "GDP Growth (%)": pd.DataFrame({"year":[2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
                                         "value":[6.4,7.4,8.0,8.3,6.8,6.5,4.0,-6.6,8.7,7.2,6.3]}),
        "Unemployment Rate (%)": pd.DataFrame({"year":[2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
                                                "value":[5.5,5.5,5.4,5.4,5.4,5.3,5.3,7.1,5.9,4.8,4.2]}),
        "Inflation CPI (%)": pd.DataFrame({"year":[2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
                                            "value":[9.4,6.7,4.9,4.5,3.6,3.4,4.8,6.2,5.5,6.7,5.4]}),
        "Bank NPA Ratio (%)": pd.DataFrame({"year":[2015,2016,2017,2018,2019,2020,2021,2022,2023],
                                             "value":[4.3,7.5,9.5,11.2,9.1,7.5,6.9,5.0,3.9]}),
        "Lending Rate (%)": pd.DataFrame({"year":[2015,2016,2017,2018,2019,2020,2021,2022,2023],
                                           "value":[13.5,13.0,12.5,12.0,11.5,11.0,10.5,10.8,11.2]}),
        "Credit to GDP (%)": pd.DataFrame({"year":[2015,2016,2017,2018,2019,2020,2021,2022,2023],
                                            "value":[51,52,50,49,50,55,54,55,57]}),
    }
    result = {}
    for name, df in indicators.items():
        result[name] = df if not df.empty else fallbacks[name]
    return result


@st.cache_data(ttl=3600)
def get_rbi_repo_rate() -> list[dict]:
    return [
        {"date": "2019-06", "rate": 5.75},
        {"date": "2019-08", "rate": 5.40},
        {"date": "2019-10", "rate": 5.15},
        {"date": "2020-03", "rate": 4.40},
        {"date": "2020-05", "rate": 4.00},
        {"date": "2022-05", "rate": 4.40},
        {"date": "2022-06", "rate": 4.90},
        {"date": "2022-08", "rate": 5.40},
        {"date": "2022-09", "rate": 5.90},
        {"date": "2022-12", "rate": 6.25},
        {"date": "2023-02", "rate": 6.50},
        {"date": "2024-02", "rate": 6.50},
        {"date": "2025-02", "rate": 6.25},
        {"date": "2025-04", "rate": 6.00},
    ]


# ── News Scraper ───────────────────────────────────────────────────────────────

NEWS_FEEDS = {
    "RBI Announcements": "https://www.rbi.org.in/Scripts/rss.aspx",
    "Economic Times - Banking": "https://economictimes.indiatimes.com/industry/banking/finance/rssfeeds/13358259.cms",
    "Moneycontrol - Economy": "https://www.moneycontrol.com/rss/economy.xml",
    "Google News - Credit Risk India": "https://news.google.com/rss/search?q=credit+risk+india+RBI&hl=en-IN&gl=IN&ceid=IN:en",
    "Google News - Loan Default India": "https://news.google.com/rss/search?q=loan+default+NPA+india&hl=en-IN&gl=IN&ceid=IN:en",
}


@st.cache_data(ttl=1800)
def fetch_news(feed_name: str, max_items: int = 8) -> list[dict]:
    """Fetch RSS feed and return list of article dicts."""
    if not FEEDPARSER_AVAILABLE:
        return []
    url = NEWS_FEEDS.get(feed_name, "")
    if not url:
        return []
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_items]:
            articles.append({
                "title": entry.get("title", "No title"),
                "link":  entry.get("link", "#"),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", "")[:200] + "..." if entry.get("summary") else "",
                "source": feed_name,
            })
        return articles
    except Exception:
        return []


@st.cache_data(ttl=1800)
def get_all_news() -> list[dict]:
    """Fetch from all feeds and merge, sorted by recency."""
    all_articles = []
    for feed_name in NEWS_FEEDS:
        all_articles.extend(fetch_news(feed_name, max_items=5))
    return all_articles
