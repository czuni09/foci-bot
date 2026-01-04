# -*- coding: utf-8 -*-
"""
TITAN UNIVERSAL STRATEGIC INTELLIGENCE - FULL ENTERPRISE EDITION v5.0
---------------------------------------------------------------------
ARCHITEKTÚRA: ASZINKRON ADAT-FÚZIÓ ÉS PSZICHOLÓGIAI NARRATÍVA
MODULOK: Understat (xG), Odds API v4, GDELT, MyMemory Translator, SQLite
TARGET: 1100-1300 SORNYI LOGIKAI MÉLYSÉG ÉS HIBAKEZELÉS
---------------------------------------------------------------------
"""

import os
import re
import math
import csv
import json
import time
import sqlite3
import asyncio
import logging
import hashlib
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional, Tuple, Union
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from understat import Understat

# =========================================================
#  1. KRITIKUS FÜGGŐSÉG KEZELŐ (RENDER FIX)
# =========================================================
def check_dependencies():
    missing = []
    try: import aiohttp
    except ImportError: missing.append("aiohttp")
    try: import feedparser
    except ImportError: missing.append("feedparser")
    
    if missing:
        st.set_page_config(page_title="TITAN - CRITICAL ERROR", page_icon="🚫")
        st.error(f"❌ HIÁNYZÓ MODULOK: {', '.join(missing)}")
        st.info("Kérlek, frissítsd a requirements.txt-t a következőre:")
        st.code("streamlit\npandas\nnumpy\naiohttp\nfeedparser\nunderstat\nplotly\nrequests", language="text")
        st.stop()

check_dependencies()
import aiohttp
import feedparser

# =========================================================
#  2. KONFIGURÁCIÓ ÉS SECRETS (GITHUB SECRETS COMPATIBLE)
# =========================================================
st.set_page_config(page_title="TITAN MISSION CONTROL", page_icon="🛰️", layout="wide")

ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
DB_PATH = "titan_persistence_vault.db"

# Statisztikai & Stratégiai korlátok
TARGET_TOTAL_ODDS = 2.00
ODDS_TOLERANCE = 0.15
MIN_HISTORICAL_GAMES = 6
LOOKAHEAD_DAYS = 4

LEAGUES = {
    "epl": {"odds": "soccer_epl", "name": "Premier League", "country": "England"},
    "la_liga": {"odds": "soccer_spain_la_liga", "name": "La Liga", "country": "Spain"},
    "bundesliga": {"odds": "soccer_germany_bundesliga", "name": "Bundesliga", "country": "Germany"},
    "serie_a": {"odds": "soccer_italy_serie_a", "name": "Serie A", "country": "Italy"},
    "ligue_1": {"odds": "soccer_france_ligue_1", "name": "Ligue 1", "country": "France"}
}

DERBY_MAP = [
    "Manchester City-Manchester United", "Arsenal-Tottenham", "Liverpool-Everton",
    "Real Madrid-Barcelona", "Atletico Madrid-Real Madrid", "AC Milan-Inter",
    "Lazio-Roma", "Bayern Munich-Borussia Dortmund", "Marseille-Paris Saint Germain"
]

# =========================================================
#  3. PERSISTENCE LAYER (ADATBÁZIS ÉS NAPLÓZÁS)
# =========================================================
class TitanVault:
    @staticmethod
    def init_db():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT UNIQUE,
                    date TEXT,
                    league TEXT,
                    match_name TEXT,
                    pick TEXT,
                    odds REAL,
                    confidence REAL,
                    stress_score INTEGER,
                    status TEXT DEFAULT 'PENDING'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match ON predictions(match_id)")

    @staticmethod
    def save_prediction(p: dict):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions 
                    (match_id, date, league, match_name, pick, odds, confidence, stress_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (p['match_id'], p['date'], p['league'], p['match_name'], p['pick'], p['odds'], p['prob'], p['stress']))
        except Exception as e:
            logging.error(f"DB Error: {e}")

# =========================================================
#  4. MATEMATIKAI MAG (POISSON & KELLY)
# =========================================================
class QuantEngine:
    @staticmethod
    def poisson_pdf(lmb: float, k: int) -> float:
        if lmb <= 0: return 1.0 if k == 0 else 0.0
        return (math.exp(-lmb) * (lmb**k)) / math.factorial(k)

    @classmethod
    def matrix_1x2(cls, home_exp: float, away_exp: float) -> Tuple[float, float, float]:
        p_h, p_d, p_a = 0.0, 0.0, 0.0
        for i in range(12):
            for j in range(12):
                prob = cls.poisson_pdf(home_exp, i) * cls.poisson_pdf(away_exp, j)
                if i > j: p_h += prob
                elif i == j: p_d += prob
                else: p_a += prob
        norm = p_h + p_d + p_a
        return p_h/norm, p_d/norm, p_a/norm

    @staticmethod
    def kelly_criterion(prob: float, odds: float) -> float:
        if odds <= 1: return 0.0
        b = odds - 1
        q = 1 - prob
        f = (b * prob - q) / b
        return max(0, f * 0.2) # 0.2-es frakcionális Kelly (biztonsági puffer)

# =========================================================
#  5. INTELLIGENCE SERVICE (API & NEWS)
# =========================================================
class IntelligenceService:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.cache = {}

    async def fetch_json(self, url: str, params: dict = None) -> Optional[dict]:
        try:
            async with self.session.get(url, params=params, timeout=15) as resp:
                if resp.status == 200: return await resp.json()
                return None
        except Exception: return None

    async def translate(self, text: str) -> str:
        if not text or len(text) < 3: return text
        if text in self.cache: return self.cache[text]
        url = f"https://api.mymemory.translated.net/get?q={quote_plus(text)}&langpair=en|hu"
        data = await self.fetch_json(url)
        res = data["responseData"]["translatedText"] if data else text
        self.cache[text] = res
        return res

    async def get_gdelt_stress(self, team: str) -> Tuple[int, List[str]]:
        query = f'"{team}" (injury OR crisis OR missing OR suspended OR internal conflict)'
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": 6}
        data = await self.fetch_json(url, params)
        if not data or "articles" not in data: return 0, []
        
        stress = 0
        titles = []
        for art in data["articles"]:
            stress += 1
            t = await self.translate(art['title'])
            titles.append(t)
        return stress, titles

# =========================================================
#  6. DATA FUSION ENGINE (KÓD SZÍVE)
# =========================================================
class TitanCore:
    def __init__(self, session: aiohttp.ClientSession):
        self.u = Understat(session)
        self.intel = IntelligenceService(session)

    async def build_xg_profile(self, league: str) -> Dict:
        results = await self.u.get_league_results(league, 2025)
        stats = {}
        for r in results:
            h, a = r['h']['title'], r['a']['title']
            xh, xa = float(r['xG']['h']), float(r['xG']['a'])
            for t, f, ag, side in [(h, xh, xa, 'h'), (a, xa, xh, 'a')]:
                if t not in stats: stats[t] = {'hf':[], 'ha':[], 'af':[], 'aa':[]}
                if side == 'h':
                    stats[t]['hf'].append(f); stats[t]['ha'].append(ag)
                else:
                    stats[t]['af'].append(f); stats[t]['aa'].append(ag)
        return stats

    async def get_market_odds(self, league_key: str) -> Dict:
        if not ODDS_API_KEY: return {}
        url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}
        data = await self.intel.fetch_json(url, params)
        return {f"{m['home_team']}_{m['away_team']}": m for m in data} if data else {}

    async def analyze_match(self, f: dict, stats: dict, odds_data: dict, l_name: str) -> Optional[dict]:
        h, a = f['h']['title'], f['a']['title']
        
        # Derby szűrő
        for d in DERBY_MAP:
            if h in d and a in d: return None

        # xG Alapú Poisson
        h_f = np.mean(stats.get(h, {}).get('hf', [1.4]))
        h_a = np.mean(stats.get(h, {}).get('ha', [1.3]))
        a_f = np.mean(stats.get(a, {}).get('af', [1.2]))
        a_a = np.mean(stats.get(a, {}).get('aa', [1.5]))
        
        exp_h = (h_f + a_a) / 2
        exp_a = (a_f + h_a) / 2
        p1, px, p2 = QuantEngine.matrix_1x2(exp_h, exp_a)
        
        # Pszichológiai Narratíva
        stress, news = await self.intel.get_gdelt_stress(h if p1 > p2 else a)
        
        # Odds API Matching (SequenceMatcher)
        price = 1.90
        found_odds = False
        for key, val in odds_data.items():
            if SequenceMatcher(None, h, val['home_team']).ratio() > 0.8:
                for b in val.get('bookmakers', []):
                    for outcome in b['markets'][0]['outcomes']:
                        if (p1 > p2 and outcome['name'] == val['home_team']):
                            price = outcome['price']; found_odds = True
                        elif (p2 > p1 and outcome['name'] == val['away_team']):
                            price = outcome['price']; found_odds = True
                if found_odds: break

        return {
            "match_id": f['id'], "date": f['datetime'], "league": l_name,
            "match_name": f"{h} - {a}", "pick": h if p1 > p2 else a,
            "prob": max(p1, p2), "odds": price, "stress": stress, "news": news,
            "ev": QuantEngine.calculate_ev(max(p1, p2), price)
        }

# =========================================================
#  7. UI & STRATÉGIAI DUÓ OPTIMALIZÁLÓ
# =========================================================
async def main():
    TitanVault.init_db()
    st.markdown("""
        <style>
        .stApp { background: #07090f; color: #ecf0f1; }
        .main-card { background: rgba(255,255,255,0.02); border: 1px solid #79a6ff33; border-radius: 12px; padding: 20px; margin-bottom: 15px; }
        .pick-val { font-size: 26px; font-weight: 900; color: #79a6ff; }
        .odds-val { font-size: 20px; color: #b387ff; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align:center;'>🛰️ TITAN UNIVERSAL v5.0</h1>", unsafe_allow_html=True)
    
    async with aiohttp.ClientSession() as session:
        core = TitanCore(session)
        all_candidates = []
        
        progress_bar = st.progress(0)
        league_items = list(LEAGUES.items())
        
        for idx, (l_id, l_info) in enumerate(league_items):
            st.write(f"⚙️ Elemzés: {l_info['name']}...")
            stats = await core.build_xg_profile(l_id)
            odds_data = await core.get_market_odds(l_info['odds'])
            fixtures = await core.u.get_league_fixtures(l_id, 2025)
            
            now = datetime.now(timezone.utc)
            tasks = []
            for f in fixtures[:12]:
                f_dt = datetime.strptime(f['datetime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                if now < f_dt < now + timedelta(days=LOOKAHEAD_DAYS):
                    tasks.append(core.analyze_match(f, stats, odds_data, l_info['name']))
            
            res = await asyncio.gather(*tasks)
            all_candidates.extend([r for r in res if r is not None])
            progress_bar.progress((idx + 1) / len(league_items))

        # DUÓ KIVÁLASZTÁS
        st.divider()
        if len(all_candidates) >= 2:
            best_duo = None
            max_quality = -1.0
            
            for i in range(len(all_candidates)):
                for j in range(i + 1, len(all_candidates)):
                    c1, c2 = all_candidates[i], all_candidates[j]
                    total_odds = c1['odds'] * c2['odds']
                    
                    if abs(total_odds - TARGET_TOTAL_ODDS) <= ODDS_TOLERANCE:
                        quality = (c1['prob'] + c2['prob']) / (1 + (c1['stress'] + c2['stress'])*0.1)
                        if quality > max_quality:
                            max_quality = quality
                            best_duo = (c1, c2, total_odds)

            if best_duo:
                st.subheader("🎯 TITAN STRATEGIC DUO (2.00 TARGET)")
                cols = st.columns(2)
                for i, match in enumerate([best_duo[0], best_duo[1]]):
                    with cols[i]:
                        st.markdown(f"""
                            <div class='main-card'>
                                <p>{match['league']} | {match['date']}</p>
                                <h3>{match['match_name']}</h3>
                                <div class='pick-val'>{match['pick']}</div>
                                <div class='odds-val'>Odds: {match['odds']:.2f}</div>
                                <p>Bizalom: {match['prob']*100:.1f}% | Stressz: {match['stress']}</p>
                                <details><summary>Narratív hírek</summary>
                                <ul>{"".join([f"<li>{t}</li>" for t in match['news']])}</ul>
                                </details>
                            </div>
                        """, unsafe_allow_html=True)
                        TitanVault.save_prediction(match)
                
                st.metric("Összesített Odds", f"{best_duo[2]:.2f}", delta=f"{best_duo[2]-2.0:.2f}")
            else:
                st.warning("Nincs az odds-kritériumnak (1.90-2.15) megfelelő meccspár.")
        
        # TELJES LISTA
        st.subheader("📋 További Kvantitatív Lehetőségek")
        if all_candidates:
            df = pd.DataFrame(all_candidates).drop(columns=['news', 'match_id'])
            st.dataframe(df.sort_values(by='score', ascending=False), use_container_width=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
