# =========================================================
# TITAN – Match Intelligence & Probability Lab
# =========================================================
# CÉL:
# - Futballmérkőzések valószínűségi és kockázati elemzése
# - Understat xG + Poisson modell
# - Piaci odds csak benchmarkként (implicit probability)
# - Hírek (GDELT + Google RSS) → pszichológiai stressz index
# - NINCS tippadás, NINCS kiválasztás, NINCS odds-szorzás
#
# FUTÁS:
#   python -m streamlit run app.py
# RENDER:
#   python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
# =========================================================

import os
import math
import time
import sqlite3
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import requests
import feedparser

# ---------------- SAFE IMPORT: aiohttp -------------------
try:
    import aiohttp
except Exception:
    st.set_page_config(page_title="Missing dependency", layout="wide")
    st.error("Hiányzó csomag: aiohttp")
    st.code("pip install aiohttp\npip install -r requirements.txt", language="bash")
    st.stop()

# ---------------- SAFE IMPORT: understat -----------------
try:
    from understat import Understat
except Exception:
    st.set_page_config(page_title="Missing dependency", layout="wide")
    st.error("Hiányzó csomag: understat")
    st.code("pip install understat\npip install -r requirements.txt", language="bash")
    st.stop()

# =========================================================
# STREAMLIT CONFIG (EGYSZER!)
# =========================================================
st.set_page_config(
    page_title="⚽ TITAN – Probability Lab",
    layout="wide",
    page_icon="🧠"
)

# =========================================================
# ENV / SECRETS
# =========================================================
def secret(name: str) -> str:
    return (os.getenv(name) or st.secrets.get(name, "") or "").strip()

ODDS_API_KEY = secret("ODDS_API_KEY")      # opcionális (benchmark)
NEWS_API_KEY = secret("NEWS_API_KEY")      # opcionális
MYMEMORY_EMAIL = secret("MYMEMORY_EMAIL")  # fordítás stabilizálás

# =========================================================
# KONFIG
# =========================================================
LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}

DAYS_AHEAD = 3
MAX_GOALS = 10

DB_PATH = "titan_probability.db"

# =========================================================
# DB
# =========================================================
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return con

def init_db():
    con = db()
    con.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            league TEXT,
            match TEXT,
            kickoff_utc TEXT,
            xg_home REAL,
            xg_away REAL,
            p_home REAL,
            p_draw REAL,
            p_away REAL,
            p_over25 REAL,
            p_btts REAL,
            stress_index INTEGER,
            notes TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

# =========================================================
# MATEMATIKA – POISSON
# =========================================================
def poisson_prob(lmbda: float, k: int) -> float:
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)

def match_probabilities(xg_home: float, xg_away: float) -> Dict[str, float]:
    p_home = p_draw = p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_prob(xg_home, hg) * poisson_prob(xg_away, ag)
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p
            if hg + ag > 2:
                p_over25 += p
            if hg > 0 and ag > 0:
                p_btts += p

    return {
        "home": p_home,
        "draw": p_draw,
        "away": p_away,
        "over25": p_over25,
        "btts": p_btts,
    }

# =========================================================
# HÍREK + STRESSZ INDEX
# =========================================================
NEGATIVE_KEYWORDS = [
    "injury", "ban", "suspension", "crisis", "conflict",
    "arrest", "investigation", "sacked", "fined"
]

def translate_hu(text: str) -> str:
    if not text:
        return ""
    try:
        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": text[:500],
            "langpair": "en|hu"
        }
        if MYMEMORY_EMAIL:
            params["de"] = MYMEMORY_EMAIL
        r = requests.get(url, params=params, timeout=6)
        return r.json().get("responseData", {}).get("translatedText", text)
    except Exception:
        return text

def news_stress(team: str) -> Tuple[int, List[str]]:
    score = 0
    lines = []

    try:
        rss = feedparser.parse(
            f"https://news.google.com/rss/search?q={team}+football"
        )
        for e in rss.entries[:8]:
            title = e.title.lower()
            if any(k in title for k in NEGATIVE_KEYWORDS):
                score -= 1
            lines.append(translate_hu(e.title))
    except Exception:
        pass

    score = max(-5, min(5, score))
    return score, lines

# =========================================================
# UNDERSTAT
# =========================================================
async def fetch_understat_matches(league: str):
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        return await us.get_league_fixtures(league, 2024)

def get_matches(league_key: str):
    try:
        return asyncio.run(fetch_understat_matches(league_key))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(fetch_understat_matches(league_key))

# =========================================================
# UI
# =========================================================
st.title("🧠 TITAN – Match Probability & Risk Lab")
st.caption("Valószínűség-alapú elemzés | nincs tippadás | kutatási cél")

selected_leagues = st.multiselect(
    "Ligák",
    list(LEAGUES.keys()),
    default=["epl"]
)

if st.button("Elemzés futtatása"):
    with st.spinner("Adatok feldolgozása…"):
        for lg in selected_leagues:
            matches = get_matches(lg)
            for m in matches:
                kickoff = datetime.fromtimestamp(int(m["datetime"]), tz=timezone.utc)
                if kickoff > datetime.now(timezone.utc) + timedelta(days=DAYS_AHEAD):
                    continue

                home = m["h"]["title"]
                away = m["a"]["title"]

                xg_home = float(m["xG"]["h"])
                xg_away = float(m["xG"]["a"])

                probs = match_probabilities(xg_home, xg_away)

                stress_h, _ = news_stress(home)
                stress_a, _ = news_stress(away)
                stress_index = stress_h + stress_a

                con = db()
                con.execute("""
                    INSERT INTO analyses
                    (created_at, league, match, kickoff_utc,
                     xg_home, xg_away,
                     p_home, p_draw, p_away,
                     p_over25, p_btts,
                     stress_index, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    datetime.utcnow().isoformat(),
                    lg,
                    f"{home} vs {away}",
                    kickoff.isoformat(),
                    xg_home,
                    xg_away,
                    probs["home"],
                    probs["draw"],
                    probs["away"],
                    probs["over25"],
                    probs["btts"],
                    stress_index,
                    "Model: Understat xG + Poisson"
                ))
                con.commit()
                con.close()

        st.success("Elemzés kész.")

# =========================================================
# MEGJELENÍTÉS
# =========================================================
con = db()
df = pd.read_sql("SELECT * FROM analyses ORDER BY id DESC LIMIT 50", con)
con.close()

if not df.empty:
    df_show = df.copy()
    for c in ["p_home", "p_draw", "p_away", "p_over25", "p_btts"]:
        df_show[c] = (df_show[c] * 100).round(1)

    st.dataframe(df_show, use_container_width=True)
else:
    st.info("Még nincs elemzés lefuttatva.")

