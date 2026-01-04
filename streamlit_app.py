import os
import re
import math
import csv
import sqlite3
import asyncio
import aiohttp
import feedparser
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from understat import Understat

# =========================================================
#  1. GLOBÁLIS KONFIGURÁCIÓ ÉS SECRETS
# =========================================================
st.set_page_config(page_title="TITAN – Universal Strategic Engine", layout="wide", page_icon="🛰️")

# GitHub Secrets / Environment variables betöltése
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))
# További kulcsok: NEWS_API_KEY, WEATHER_API_KEY stb. a secret-ben vannak

DB_PATH = "titan_vault.db"
TARGET_TOTAL_ODDS = 2.00
PICKS_LOG_CSV = Path("titan_history.csv")

# Ligák megfeleltetése (Understat vs Odds API)
LEAGUE_MAP = {
    "epl": {"odds": "soccer_epl", "name": "Premier League"},
    "la_liga": {"odds": "soccer_spain_la_liga", "name": "La Liga"},
    "bundesliga": {"odds": "soccer_germany_bundesliga", "name": "Bundesliga"},
    "serie_a": {"odds": "soccer_italy_serie_a", "name": "Serie A"},
    "ligue_1": {"odds": "soccer_france_ligue_1", "name": "Ligue 1"},
}

# =========================================================
#  2. CSS DESIGN (Professional Dark Mode)
# =========================================================
st.markdown("""
<style>
    .stApp { background-color: #06070c; color: #e0e0e0; }
    .titan-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(121, 166, 255, 0.2);
        border-radius: 15px; padding: 20px; margin-bottom: 20px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #79a6ff; }
    .status-ok { color: #4ef0a3; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================================================
#  3. MAG (CORE) OSZTÁLYOK
# =========================================================

class StatisticsEngine:
    @staticmethod
    def poisson_prob(lmb, k):
        if lmb <= 0: return 1.0 if k == 0 else 0.0
        return (math.exp(-lmb) * (lmb**k)) / math.factorial(k)

    @classmethod
    def calculate_1x2(cls, lh, la):
        p1, px, p2 = 0.0, 0.0, 0.0
        for i in range(10):
            for j in range(10):
                prob = cls.poisson_prob(lh, i) * cls.poisson_prob(la, j)
                if i > j: p1 += prob
                elif i == j: px += prob
                else: p2 += prob
        return p1, px, p2

class NetworkIntelligence:
    def __init__(self, session):
        self.session = session

    async def fetch_json(self, url, params=None):
        try:
            async with self.session.get(url, params=params, timeout=10) as resp:
                return await resp.json() if resp.status == 200 else None
        except: return None

    async def get_market_odds(self, league_odds_key):
        if not ODDS_API_KEY: return None
        url = f"https://api.the-odds-api.com/v4/sports/{league_odds_key}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h", "bookmakers": "pinnacle,betfair_ex"}
        return await self.fetch_json(url, params)

    async def get_narrative(self, query):
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": 5}
        data = await self.fetch_json(url, params)
        return data.get("articles", []) if data else []

# =========================================================
#  4. A TITAN FŐ FOLYAMAT
# =========================================================

async def main():
    st.markdown('<h1 style="text-align:center; color:#79a6ff;">🛰️ TITAN UNIVERSAL</h1>', unsafe_allow_html=True)
    
    async with aiohttp.ClientSession() as session:
        net = NetworkIntelligence(session)
        u = Understat(session)
        
        all_candidates = []
        progress = st.progress(0)
        
        for idx, (l_key, l_info) in enumerate(LEAGUE_MAP.items()):
            # 1. Adatlekérés (Understat + Odds) párhuzamosan
            fixtures_task = u.get_league_fixtures(l_key, 2025) # Aktuális szezon
            odds_task = net.get_market_odds(l_info['odds'])
            
            fixtures, odds_data = await asyncio.gather(fixtures_task, odds_task)
            
            # 2. xG Profil építése (Results alapján)
            results = await u.get_league_results(l_key, 2025)
            # (Itt az xG számítási logika az eredeti streamlit_app.py-ból...)
            
            # 3. Összefésülés és elemzés
            now = datetime.now(timezone.utc)
            for f in fixtures[:10]: # Csak a legközelebbi 10 meccs
                f_date = datetime.strptime(f['datetime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                if now < f_date < now + timedelta(days=4):
                    home, away = f['h']['title'], f['a']['title']
                    
                    # Poisson számítás (lh, la kinyerése...)
                    lh, la = 1.5, 1.2 # Placeholder
                    p1, px, p2 = StatisticsEngine.calculate_1x2(lh, la)
                    
                    # Narratív szűrés
                    news = await net.get_narrative(f'"{home}" OR "{away}"')
                    stress = len([n for n in news if "injury" in n['title'].lower()])
                    
                    # Piaci Odds illesztés
                    match_odds = None
                    if odds_data:
                        # Név-egyeztetés logika (SequenceMatcher vagy direkt)
                        pass 

                    all_candidates.append({
                        "match": f"{home} - {away}",
                        "prob": max(p1, p2),
                        "pick": home if p1 > p2 else away,
                        "stress": stress,
                        "league": l_info['name']
                    })
            
            progress.progress((idx + 1) / len(LEAGUE_MAP))

        # 5. Stratégiai Duó választás
        st.subheader("🎯 TITAN Strategic Duo (Target Odds: 2.00)")
        if len(all_candidates) >= 2:
            final_picks = sorted(all_candidates, key=lambda x: x['prob'], reverse=True)[:2]
            
            c1, c2 = st.columns(2)
            for i, p in enumerate(final_picks):
                with (c1 if i==0 else c2):
                    st.markdown(f"""
                    <div class="titan-card">
                        <h3>{p['match']}</h3>
                        <p>{p['league']}</p>
                        <div class="metric-value">{p['pick']}</div>
                        <p>Bizalmi szint: {p['prob']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

# =========================================================
#  5. RUN
# =========================================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
