"""
Fetches live macroeconomic indicators from free public APIs.
- World Bank API: GDP growth, unemployment (India)
- RBI / data.gov.in: fallback static data if API unavailable
All calls are cached for 1 hour to avoid rate limits.
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime


@st.cache_data(ttl=3600)
def fetch_world_bank(indicator: str, country: str = "IN", years: int = 10) -> pd.DataFrame:
    """Fetch indicator from World Bank Open Data API."""
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
    """
    Returns dict of DataFrames for key India macro indicators.
    Falls back to static data if API fails.
    """
    indicators = {
        "GDP Growth (%)": fetch_world_bank("NY.GDP.MKTP.KD.ZG"),
        "Unemployment Rate (%)": fetch_world_bank("SL.UEM.TOTL.ZS"),
        "Inflation (CPI %)": fetch_world_bank("FP.CPI.TOTL.ZG"),
        "Bank NPA Ratio (%)": fetch_world_bank("FB.AST.NPER.ZS"),
    }
    # Fill any failed fetches with static fallback
    fallbacks = {
        "GDP Growth (%)": pd.DataFrame({
            "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "value": [8.0, 8.3, 6.8, 6.5, 4.0, -6.6, 8.7, 7.2, 6.3]
        }),
        "Unemployment Rate (%)": pd.DataFrame({
            "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "value": [5.4, 5.4, 5.4, 5.3, 5.3, 7.1, 5.9, 4.8, 4.2]
        }),
        "Inflation (CPI %)": pd.DataFrame({
            "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "value": [4.9, 4.5, 3.6, 3.4, 4.8, 6.2, 5.5, 6.7, 5.4]
        }),
        "Bank NPA Ratio (%)": pd.DataFrame({
            "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "value": [4.3, 7.5, 9.5, 11.2, 9.1, 7.5, 6.9, 5.0, 3.9]
        }),
    }
    result = {}
    for name, df in indicators.items():
        result[name] = df if not df.empty else fallbacks[name]
    return result


@st.cache_data(ttl=3600)
def get_rbi_repo_rate() -> list[dict]:
    """Static RBI repo rate history (updated manually — RBI has no public JSON API)."""
    return [
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
