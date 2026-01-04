# ============================================================
# TITAN – Probability & Risk Intelligence Lab
# ============================================================
# PURPOSE:
# - Football match probability analysis
# - NO betting, NO picks, NO odds multiplication
# - Understat xG + Poisson
# - News-based psychological stress model
# - Dashboard UI (charts)
# - SQLite local persistence
#
# TECH:
# - Streamlit
# - Python (latest)
# - Render compatible
# ============================================================

import os
import math
import time
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import requests

# =========================
# SAFE OPTIONAL IMPORTS
# =========================
try:
    import aiohttp
except Exception:
    st.error("Missing dependency: aiohttp")
    st.stop()

try:
    from understat import Understat
except Exception:
    st.error("Missing dependency: understat")
    st.stop()

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import feedparser
    FEED_OK = True
except Exception:
    FEED_OK = False

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="⚽ TITAN – Probability Lab",
    layout="wide",
    page_icon="🧠"
)

# =========================
# ENV / SECRETS
# =========================
def secret(name: str) -> str:
    return (os.getenv(name) or st.secrets.get(name, "") or "").strip()

MYMEMORY_EMAIL = secret("MYMEMORY_EMAIL")

# =========================
# CONFIG
# =========================
LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}

DAYS_AHEAD = 4
MAX_GOALS = 10
DB_PATH = "titan_lab.db"

NEGATIVE_KEYWORDS = [
    "injury", "ban", "suspension", "crisis", "conflict",
    "arrest", "investigation", "sacked", "fined"
]

# =========================
# DATABASE
# =========================
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
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
            confidence TEXT,
            notes TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

# =========================
# MATHEMATICS – POISSON
# =========================
def poisson_prob(lmbda: float, k: int) -> float:
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)

def compute_probabilities(xgh: float, xga: float) -> Dict[str, float]:
    p_home = p_draw = p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_prob(xgh, hg) * poisson_prob(xga, ag)
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

# =========================
# TRANSLATION (SAFE)
# =========================
def translate_hu(text: str) -> str:
    if not text:
        return ""
    try:
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={
                "q": text[:500],
                "langpair": "en|hu",
                "de": MYMEMORY_EMAIL or None
            },
            timeout=6
        )
        return r.json().get("responseData", {}).get("translatedText", text)
    except Exception:
        return text

# =========================
# STRESS MODEL
# =========================
def stress_from_news(team: str) -> Tuple[int, List[str]]:
    if not FEED_OK:
        return 0, []

    score = 0
    titles = []

    try:
        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={team}+football"
        )
        for e in feed.entries[:8]:
            t = e.title.lower()
            if any(k in t for k in NEGATIVE_KEYWORDS):
                score -= 1
            titles.append(translate_hu(e.title))
    except Exception:
        pass

    return max(-5, min(5, score)), titles

# =========================
# UNDERSTAT FETCH
# =========================
async def fetch_understat(league: str):
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        return await us.get_league_fixtures(league)

def get_matches(league: str):
    try:
        return asyncio.run(fetch_understat(league))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(fetch_understat(league))

# =========================
# UI
# =========================
st.title("🧠 TITAN – Match Probability Intelligence")
st.caption("Valószínűség • Kockázat • Kontextus | nincs tippadás")

leagues = st.multiselect(
    "Ligák",
    list(LEAGUES.keys()),
    default=["epl"]
)

run = st.button("Elemzés futtatása")

if run:
    with st.spinner("Elemzés folyamatban…"):
        for lg in leagues:
            matches = get_matches(lg)
            for m in matches:
                kickoff = datetime.fromtimestamp(
                    int(m["datetime"]), tz=timezone.utc
                )
                if kickoff > datetime.now(timezone.utc) + timedelta(days=DAYS_AHEAD):
                    continue

                home = m["h"]["title"]
                away = m["a"]["title"]

                xgh = float(m["xG"]["h"])
                xga = float(m["xG"]["a"])

                probs = compute_probabilities(xgh, xga)

                sh, _ = stress_from_news(home)
                sa, _ = stress_from_news(away)
                stress = sh + sa

                confidence = (
                    "HIGH" if abs(xgh - xga) > 0.6 else
                    "MEDIUM" if abs(xgh - xga) > 0.3 else
                    "LOW"
                )

                con = db()
                con.execute("""
                    INSERT INTO analyses VALUES (
                        NULL,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                """, (
                    datetime.utcnow().isoformat(),
                    lg,
                    f"{home} vs {away}",
                    kickoff.isoformat(),
                    xgh,
                    xga,
                    probs["home"],
                    probs["draw"],
                    probs["away"],
                    probs["over25"],
                    probs["btts"],
                    stress,
                    confidence,
                    "Understat xG + Poisson + stress"
                ))
                con.commit()
                con.close()

        st.success("Elemzés elkészült.")

# =========================
# DATA VIEW
# =========================
con = db()
df = pd.read_sql(
    "SELECT * FROM analyses ORDER BY id DESC LIMIT 100",
    con
)
con.close()

if not df.empty:
    df_show = df.copy()
    for c in ["p_home", "p_draw", "p_away", "p_over25", "p_btts"]:
        df_show[c] = (df_show[c] * 100).round(1)

    st.subheader("Eredmények")
    st.dataframe(df_show, use_container_width=True)

    if PLOTLY_OK:
        st.subheader("Valószínűségi megoszlás")
        fig = px.histogram(
            df,
            x="p_home",
            nbins=20,
            title="Hazai győzelem valószínűség eloszlás"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Még nincs adat.")



