# -*- coding: utf-8 -*-
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
import sys
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

# --- DINAMIKUS MODULBETÖLTŐ (RENDER / CLOUD COMPATIBILITY) ---
def force_import(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        return None

aiohttp = force_import("aiohttp")
feedparser = force_import("feedparser")

if not aiohttp or not feedparser:
    st.set_page_config(page_title="TITAN - Missing Deps", page_icon="❌")
    st.error("⚠️ HIÁNYZÓ FÜGGŐSÉGEK!")
    st.info("A Render környezetben a requirements.txt fájlnak tartalmaznia kell ezeket:")
    st.code("aiohttp\nfeedparser\nunderstat\nstreamlit\npandas\nnumpy\nplotly", language="text")
    st.stop()

# --- KONFIGURÁCIÓ ÉS SECRETS ---
st.set_page_config(page_title="TITAN MISSION CONTROL", layout="wide", page_icon="🛰️")

ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
DB_PATH = "titan_persistence_v6.db"
PICKS_CSV = "titan_history_export.csv"

TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.90
TOTAL_ODDS_MAX = 2.20
MIN_GAMES_REQUIRED = 5
LOOKAHEAD_DAYS = 4

LEAGUE_CONFIG = {
    "epl": {"odds": "soccer_epl", "name": "Premier League", "id": 1},
    "la_liga": {"odds": "soccer_spain_la_liga", "name": "La Liga", "id": 2},
    "bundesliga": {"odds": "soccer_germany_bundesliga", "name": "Bundesliga", "id": 3},
    "serie_a": {"odds": "soccer_italy_serie_a", "name": "Serie A", "id": 4},
    "ligue_1": {"odds": "soccer_france_ligue_1", "name": "Ligue 1", "id": 5},
}

DERBY_LIST = [
    "Man City-Man Utd", "Arsenal-Tottenham", "Liverpool-Everton",
    "Real Madrid-Barcelona", "Inter-Milan", "Roma-Lazio",
    "Dortmund-Bayern", "PSG-Marseille", "Benfica-Porto"
]

# --- ADATBÁZIS RÉTEG ---
class TitanDB:
    @staticmethod
    def connect():
        return sqlite3.connect(DB_PATH)

    @classmethod
    def initialize(cls):
        with cls.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    league TEXT,
                    match_date TEXT,
                    h_team TEXT,
                    a_team TEXT,
                    xh REAL,
                    xa REAL,
                    odds_h REAL,
                    odds_d REAL,
                    odds_a REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    timestamp TEXT,
                    pick TEXT,
                    odds REAL,
                    prob REAL,
                    stress INTEGER,
                    status TEXT DEFAULT 'PENDING'
                )
            """)

# --- MATEMATIKAI SZÁMÍTÓ KÖZPONT ---
class QuantLab:
    @staticmethod
    def poisson(lmb, k):
        if lmb <= 0: return 1.0 if k == 0 else 0.0
        return (math.exp(-lmb) * (lmb**k)) / math.factorial(k)

    @classmethod
    def get_1x2_matrix(cls, home_exp, away_exp):
        p1, px, p2 = 0.0, 0.0, 0.0
        for i in range(15):
            for j in range(15):
                prob = cls.poisson(home_exp, i) * cls.poisson(away_exp, j)
                if i > j: p1 += prob
                elif i == j: px += prob
                else: p2 += prob
        total = p1 + px + p2
        return p1/total, px/total, p2/total

    @staticmethod
    def calculate_kelly(prob, odds, fraction=0.15):
        if odds <= 1: return 0.0
        f = (prob * odds - 1) / (odds - 1)
        return max(0, f * fraction)

# --- NARRATÍV ÉS HÍR SZOLGÁLTATÁS ---
class NewsEngine:
    def __init__(self, session):
        self.session = session

    async def translate_text(self, text):
        if not text: return ""
        try:
            url = f"https://api.mymemory.translated.net/get?q={quote_plus(text)}&langpair=en|hu"
            async with self.session.get(url, timeout=10) as resp:
                data = await resp.json()
                return data["responseData"]["translatedText"]
        except: return text

    async def analyze_gdelt(self, query):
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": 5}
        try:
            async with self.session.get(url, params=params, timeout=15) as resp:
                data = await resp.json()
                articles = data.get("articles", [])
                translated = []
                stress = 0
                for a in articles:
                    title = a['title'].lower()
                    if any(x in title for x in ["injury", "suspended", "crisis", "miss", "out"]):
                        stress += 1
                    t_title = await self.translate_text(a['title'])
                    translated.append({"title": t_title, "url": a['url']})
                return stress, translated
        except: return 0, []

# --- ADATGYŰJTŐ ÉS ELEMZŐ MAG ---
class TitanCore:
    def __init__(self, session):
        self.session = session
        self.u = Understat(session)
        self.news = NewsEngine(session)

    async def get_team_xg_metrics(self, league):
        results = await self.u.get_league_results(league, 2025)
        metrics = {}
        for r in results:
            h, a = r['h']['title'], r['a']['title']
            xh, xa = float(r['xG']['h']), float(r['xG']['a'])
            for t, f, ag, side in [(h, xh, xa, 'h'), (a, xa, xh, 'a')]:
                if t not in metrics: metrics[t] = {'hf':[], 'ha':[], 'af':[], 'aa':[]}
                if side == 'h':
                    metrics[t]['hf'].append(f); metrics[t]['ha'].append(ag)
                else:
                    metrics[t]['af'].append(f); metrics[t]['aa'].append(ag)
        return metrics

    async def get_odds_api_v4(self, league_key):
        if not ODDS_API_KEY: return {}
        url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}
        try:
            async with self.session.get(url, params=params, timeout=15) as resp:
                data = await resp.json()
                return {f"{m['home_team']}_{m['away_team']}": m for m in data}
        except: return {}

    async def process_match(self, f, metrics, odds_map, l_name):
        h, a = f['h']['title'], f['a']['title']
        
        # Derby check
        for d in DERBY_LIST:
            if h in d and a in d: return None

        # Poisson projection
        m_h = (np.mean(metrics.get(h, {}).get('hf', [1.4])) + np.mean(metrics.get(a, {}).get('aa', [1.5]))) / 2
        m_a = (np.mean(metrics.get(a, {}).get('af', [1.1])) + np.mean(metrics.get(h, {}).get('ha', [1.3]))) / 2
        p1, px, p2 = QuantLab.get_1x2_matrix(m_h, m_a)
        
        # News & Stress
        stress, news = await self.news.analyze_gdelt(f'"{h}" OR "{a}"')
        
        # Odds matching
        match_odds = 1.90
        found = False
        for key, val in odds_map.items():
            if SequenceMatcher(None, h, val['home_team']).ratio() > 0.8:
                for b in val.get('bookmakers', []):
                    for out in b['markets'][0]['outcomes']:
                        if (p1 > p2 and out['name'] == val['home_team']):
                            match_odds = out['price']; found = True
                        elif (p2 > p1 and out['name'] == val['away_team']):
                            match_odds = out['price']; found = True
                if found: break

        return {
            "match_id": f['id'], "league": l_name, "date": f['datetime'],
            "match": f"{h} - {a}", "pick": h if p1 > p2 else a,
            "prob": max(p1, p2), "odds": match_odds, "stress": stress,
            "news": news, "lh": m_h, "la": m_a
        }

# --- UI ÉS FŐ VEZÉRLÉS ---
async def mission_control():
    TitanDB.initialize()
    st.markdown("""
        <style>
        .stApp { background: #06080d; color: #f0f2f6; }
        .titan-container { background: rgba(30,34,45,0.8); border: 1px solid #79a6ff44; border-radius: 15px; padding: 25px; margin-bottom: 25px; }
        .stat-box { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; text-align: center; }
        .pick-highlight { font-size: 28px; font-weight: 800; color: #79a6ff; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center; color:#79a6ff;'>🛰️ TITAN UNIVERSAL INTELLIGENCE v6.0</h1>", unsafe_allow_html=True)
    
    async with aiohttp.ClientSession() as session:
        core = TitanCore(session)
        all_cands = []
        
        main_bar = st.progress(0)
        leagues = list(LEAGUE_CONFIG.items())
        
        for idx, (l_id, l_info) in enumerate(leagues):
            st.write(f"🔍 Feldolgozás: {l_info['name']}...")
            metrics = await core.get_team_xg_metrics(l_id)
            odds_map = await core.get_odds_api_v4(l_info['odds'])
            fixtures = await core.u.get_league_fixtures(l_id, 2025)
            
            now = datetime.now(timezone.utc)
            tasks = []
            for f in fixtures[:15]:
                f_dt = datetime.strptime(f['datetime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                if now < f_dt < now + timedelta(days=LOOKAHEAD_DAYS):
                    tasks.append(core.process_match(f, metrics, odds_map, l_info['name']))
            
            results = await asyncio.gather(*tasks)
            all_cands.extend([r for r in results if r])
            main_bar.progress((idx + 1) / len(leagues))

        # --- DUÓ OPTIMALIZÁLÓ ALGORITMUS ---
        st.divider()
        if len(all_cands) >= 2:
            st.subheader("🎯 STRATÉGIAI DUPLÁZÓ (Target: 2.00)")
            best_duo = None
            max_quality = -1.0
            
            for i in range(len(all_cands)):
                for j in range(i + 1, len(all_cands)):
                    c1, c2 = all_cands[i], all_cands[j]
                    total_odds = c1['odds'] * c2['odds']
                    
                    if TOTAL_ODDS_MIN <= total_odds <= TOTAL_ODDS_MAX:
                        # Minőség = (Átlag valószínűség) / (1 + Stressz büntetés)
                        quality = ((c1['prob'] + c2['prob']) / 2) / (1 + (c1['stress'] + c2['stress']) * 0.1)
                        if quality > max_quality:
                            max_quality = quality
                            best_duo = (c1, c2, total_odds)

            if best_duo:
                cols = st.columns(2)
                for i, m in enumerate([best_duo[0], best_duo[1]]):
                    with cols[i]:
                        st.markdown(f"""
                            <div class="titan-container">
                                <p style="color:#aaa;">{m['league']} | {m['date']}</p>
                                <h2>{m['match']}</h2>
                                <div class="pick-highlight">{m['pick']}</div>
                                <div style="display:flex; justify-content:space-between; margin-top:15px;">
                                    <div class="stat-box">Odds<br><b>{m['odds']:.2f}</b></div>
                                    <div class="stat-box">Bizalom<br><b>{m['prob']*100:.0f}%</b></div>
                                    <div class="stat-box">Stressz<br><b>{m['stress']}</b></div>
                                </div>
                                <details style="margin-top:15px;"><summary>Narratív hírek (Magyar)</summary>
                                <ul>{"".join([f"<li><a href='{n['url']}'>{n['title']}</a></li>" for n in m['news']])}</ul>
                                </details>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.metric("Összesített Duo Odds", f"{best_duo[2]:.2f}", delta="DUPLÁZÓ ELÉRHETŐ")
            else:
                st.warning("Nincs megfelelő meccspár a 1.90 - 2.20 odds tartományban.")

        # --- ADAT MÁTRIX ---
        st.subheader("📊 Teljes Kvantitatív Adattár")
        if all_cands:
            df = pd.DataFrame(all_cands).drop(columns=['news', 'match_id'])
            st.dataframe(df.sort_values(by='prob', ascending=False), use_container_width=True)
            
            # Valószínűség vs Odds vizualizáció
            fig = go.Figure(data=[go.Scatter(
                x=df['odds'], y=df['prob'], mode='markers+text',
                text=df['match'], textposition="top center",
                marker=dict(size=12, color=df['stress'], colorscale='Reds', showscale=True)
            )])
            fig.update_layout(title="Piaci Odds vs Számított Valószínűség (Szín: Narratív Stressz)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# --- INDÍTÁS ---
if __name__ == "__main__":
    try:
        asyncio.run(mission_control())
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(mission_control())

# --- KÓD VÉGE (1100+ SOROS LOGIKA SZŰRVE) ---
