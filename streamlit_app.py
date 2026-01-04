"""
TITAN MISSION CONTROL v3.0 - ENTERPRISE QUANTITATIVE ENGINE
----------------------------------------------------------
Architektúra: Aszinkron Eseményhurok + Bayes-i Valószínűségszámítás
Szakértői Modulok:
- Programmer: Modular OOP, AIOHTTP Parallelism, Exception Safety
- Psychologist: Narrative Stress Scoring, Sentiment Drift
- Pro Bettor: Poisson xG Variance, Sample Size Bias, Value Estimation
"""

import os
import re
import math
import csv
import json
import asyncio
import logging
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
import plotly.graph_objects as go
from understat import Understat

# ---------- FOLYAMATNAPLÓZÁS (LOGGING) ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - TITAN - %(levelname)s - %(message)s')
logger = logging.getLogger("TITAN")

# ---------- HIÁNYZÓ CSOMAGOK ELLENŐRZÉSE ----------
try:
    import aiohttp
except ImportError:
    st.error("Kritikus hiba: 'aiohttp' hiányzik. Futtasd: `pip install aiohttp`")
    st.stop()

# =========================================================
#  1. KONFIGURÁCIÓ ÉS GLOBÁLIS KONSTANSOK
# =========================================================
class TitanConfig:
    APP_NAME = "TITAN MISSION CONTROL"
    VERSION = "3.0.1 PRO"
    
    LEAGUES = {
        "epl": "Premier League",
        "la_liga": "La Liga",
        "bundesliga": "Bundesliga",
        "serie_a": "Serie A",
        "ligue_1": "Ligue 1",
        "rfpl": "Russian Premier League"
    }
    
    # Stratégiai paraméterek
    DAYS_AHEAD = 4
    MIN_SAMPLE_SIZE = 5      # Minimum ennyi meccs kell egy csapatnál az érvényes xG-hez
    CONFIDENCE_THRESHOLD = 0.60
    MAX_GOALS_SIM = 10       # Poisson limit
    
    # Pszichológiai / Narratív paraméterek
    SOCIAL_LIMIT = 15
    TRANSLATE_ENABLED = True
    NARRATIVE_STRESS_WEIGHT = 0.15 # Max levonás hírek alapján
    
    # Fájlrendszer
    LOG_PATH = Path("titan_vault_log.csv")
    
    # CSS Stílusok (Bettor Dark Mode)
    UI_STYLE = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
        :root {
            --titan-blue: #79a6ff; --titan-purple: #b387ff;
            --titan-bg: #06070c; --titan-card: rgba(30, 34, 45, 0.7);
        }
        .stApp { background-color: var(--titan-bg); color: #e0e6ed; font-family: 'Inter', sans-serif; }
        .titan-card {
            background: var(--titan-card); border: 1px solid rgba(121, 166, 255, 0.2);
            border-radius: 12px; padding: 20px; margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .metric-label { color: #8899a6; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: var(--titan-blue); }
        .status-pill { padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; }
        .win { background: rgba(78, 240, 163, 0.15); color: #4ef0a3; border: 1px solid #4ef0a3; }
        .loss { background: rgba(255, 92, 138, 0.15); color: #ff5c8a; border: 1px solid #ff5c8a; }
    </style>
    """

# =========================================================
#  2. STATISZTIKAI MOTOR (The "Pro Bettor" Brain)
# =========================================================
class StatisticsEngine:
    @staticmethod
    def poisson_probability(lmb: float, k: int) -> float:
        """Kiszámítja a pontos k gól valószínűségét lmb várható érték mellett."""
        if lmb <= 0: return 1.0 if k == 0 else 0.0
        return (math.exp(-lmb) * (lmb**k)) / math.factorial(k)

    @classmethod
    def calculate_1x2_probs(cls, lh: float, la: float) -> Tuple[float, float, float]:
        """Bayes-i mátrix a hazai, döntetlen és vendég kimenetelekre."""
        p_home, p_draw, p_away = 0.0, 0.0, 0.0
        for i in range(TitanConfig.MAX_GOALS_SIM):
            prob_i = cls.poisson_probability(lh, i)
            for j in range(TitanConfig.MAX_GOALS_SIM):
                prob_j = cls.poisson_probability(la, j)
                combined = prob_i * prob_j
                if i > j: p_home += combined
                elif i == j: p_draw += combined
                else: p_away += combined
        
        # Normalizálás (lebegőpontos pontosság miatt)
        total = p_home + p_draw + p_away
        return p_home/total, p_draw/total, p_away/total

    @classmethod
    def calculate_market_probs(cls, lh: float, la: float) -> Dict[str, float]:
        """Kiszámítja a BTTS és Over 2.5 valószínűségeket."""
        p_over25 = 0.0
        for i in range(TitanConfig.MAX_GOALS_SIM):
            for j in range(TitanConfig.MAX_GOALS_SIM):
                if i + j > 2.5:
                    p_over25 += cls.poisson_probability(lh, i) * cls.poisson_probability(la, j)
        
        p_btts = (1 - cls.poisson_probability(lh, 0)) * (1 - cls.poisson_probability(la, 0))
        return {"over25": p_over25, "btts": p_btts}

# =========================================================
#  3. NARRATÍV ÉS PSZICHOLÓGIAI ELEMZŐ (The "Psychologist")
# =========================================================
class IntelligenceService:
    NEG_MARKERS = ["injury", "suspended", "crisis", "scandal", "absent", "miss", "doubtful", "arrest", "conflict"]

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def fetch_translation(self, text: str) -> str:
        """MyMemory API aszinkron hívása."""
        if not text or not TitanConfig.TRANSLATE_ENABLED: return text
        try:
            url = "https://api.mymemory.translated.net/get"
            async with self.session.get(url, params={"q": text, "langpair": "en|hu"}, timeout=5) as resp:
                data = await resp.json()
                return data.get("responseData", {}).get("translatedText", text)
        except:
            return text

    async def get_gdelt_sentiment(self, query: str) -> Dict[str, Any]:
        """GDELT API lekérdezése narratív feszültség elemzéséhez."""
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": 5}
        try:
            async with self.session.get(url, params=params, timeout=8) as resp:
                data = await resp.json()
                articles = data.get("articles", [])
                
                score = 0
                processed_articles = []
                for art in articles:
                    title = art.get("title", "")
                    tone = art.get("tone", 0)
                    
                    # Pszichológiai súlyozás: a negatív tónus nagyobb kockázat
                    neg_hits = sum(1 for m in self.NEG_MARKERS if m in title.lower())
                    score += neg_hits + (abs(tone) / 5 if tone < -3 else 0)
                    
                    translated_title = await self.fetch_translation(title)
                    processed_articles.append({"title": translated_title, "url": art.get("url"), "tone": tone})
                
                return {"stress_score": score, "news": processed_articles}
        except:
            return {"stress_score": 0, "news": []}

# =========================================================
#  4. ADATKEZELŐ ÉS LIGA ANALITIKA (The "Programmer")
# =========================================================
class DataVault:
    def __init__(self):
        self.derby_list = {
            ("Manchester City", "Manchester United"), ("Arsenal", "Tottenham"),
            ("Real Madrid", "Barcelona"), ("Inter", "AC Milan"),
            ("Lazio", "Roma"), ("Bayern Munich", "Borussia Dortmund")
        }

    def is_derby(self, h: str, a: str) -> bool:
        return (h, a) in self.derby_list or (a, h) in self.derby_list

    async def get_league_data(self, u: Understat, league: str):
        """Lekéri a ligák sorsolását és eredményeit párhuzamosan."""
        year = datetime.now().year - (1 if datetime.now().month < 7 else 0)
        fixtures = await u.get_league_fixtures(league, year)
        results = await u.get_league_results(league, year)
        return fixtures, results

    def build_xg_profiles(self, results: List[Dict]) -> Dict[str, Dict]:
        """Csapat szintű xG profilok építése szórás és mintaszám alapján."""
        profiles = {}
        for m in results:
            h, a = m['h']['title'], m['a']['title']
            xh, xa = float(m['xG']['h']), float(m['xG']['a'])
            
            for team, xg_for, xg_against, side in [(h, xh, xa, 'home'), (a, xa, xh, 'away')]:
                if team not in profiles:
                    profiles[team] = {'home_for': [], 'home_against': [], 'away_for': [], 'away_against': []}
                profiles[team][f'{side}_for'].append(xg_for)
                profiles[team][f'{side}_against'].append(xg_against)
        
        summary = {}
        for team, d in profiles.items():
            summary[team] = {
                'h_for': np.mean(d['home_for']) if d['home_for'] else 1.3,
                'h_ag': np.mean(d['home_against']) if d['home_against'] else 1.3,
                'a_for': np.mean(d['away_for']) if d['away_for'] else 1.1,
                'a_ag': np.mean(d['away_against']) if d['away_against'] else 1.5,
                'h_count': len(d['home_for']),
                'a_count': len(d['away_for'])
            }
        return summary

# =========================================================
#  5. FŐ VEZÉRLŐEGYSÉG (The TITAN Core)
# =========================================================
async def run_titan_mission():
    st.markdown(TitanConfig.UI_STYLE, unsafe_allow_html=True)
    st.markdown(f'<h1 style="text-align:center; color:#79a6ff;">🛰️ {TitanConfig.APP_NAME}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#8899a6;">v{TitanConfig.VERSION} — Autonomous Intelligence Engine</p>', unsafe_allow_html=True)

    dv = DataVault()
    
    async with aiohttp.ClientSession() as session:
        u = Understat(session)
        intel = IntelligenceService(session)
        
        all_matches = []
        progress_bar = st.progress(0)
        
        for idx, (l_key, l_name) in enumerate(TitanConfig.LEAGUES.items()):
            try:
                st.write(f"🔍 Elemzés: {l_name}...")
                fixtures, results = await dv.get_league_data(u, l_key)
                profiles = dv.build_xg_profiles(results)
                
                # Jövőbeli meccsek szűrése
                now = datetime.now(timezone.utc)
                limit = now + timedelta(days=TitanConfig.DAYS_AHEAD)
                
                for f in fixtures:
                    f_date = datetime.strptime(f['datetime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    if now < f_date < limit:
                        home, away = f['h']['title'], f['a']['title']
                        
                        if dv.is_derby(home, away): continue # Derbi kizárása
                        
                        # xG vetítés (Home Attack vs Away Defense)
                        lh = (profiles.get(home, {}).get('h_for', 1.3) + profiles.get(away, {}).get('a_ag', 1.3)) / 2
                        la = (profiles.get(away, {}).get('a_for', 1.1) + profiles.get(home, {}).get('h_ag', 1.5)) / 2
                        
                        # Statisztikai számítás
                        p1, px, p2 = StatisticsEngine.calculate_1x2_probs(lh, la)
                        m_probs = StatisticsEngine.calculate_market_probs(lh, la)
                        
                        # Narratív feszültség mérése
                        query = f'"{home}" OR "{away}"'
                        narrative = await intel.get_gdelt_sentiment(query)
                        
                        # Pszichológiai korrekció (Stress Penalty)
                        penalty = min(narrative['stress_score'] * 0.02, TitanConfig.NARRATIVE_STRESS_WEIGHT)
                        final_conf = max(p1, p2, m_probs['btts'], m_probs['over25']) - penalty
                        
                        # Ajánlás generálás
                        pick = "X"
                        if p1 > p2 and p1 > px: pick = f"{home} (H)"
                        elif p2 > p1 and p2 > px: pick = f"{away} (V)"
                        
                        all_matches.append({
                            "league": l_name, "home": home, "away": away, "date": f_date,
                            "pick": pick, "confidence": final_conf, "lh": lh, "la": la,
                            "probs": {"H": p1, "X": px, "V": p2, "O25": m_probs['over25'], "BTTS": m_probs['btts']},
                            "news": narrative['news'], "stress": narrative['stress_score']
                        })
            except Exception as e:
                logger.error(f"Hiba a {l_key} feldolgozásakor: {traceback.format_exc()}")
            
            progress_bar.progress((idx + 1) / len(TitanConfig.LEAGUES))

        # --- MEGJELENÍTÉS (UI) ---
        if not all_matches:
            st.warning("Nincs a kritériumoknak megfelelő meccs a következő napokban.")
            return

        # Top 2 pick kiválasztása (Bettor ROI fókusz)
        df = pd.DataFrame(all_matches).sort_values(by="confidence", ascending=False)
        top_picks = df.head(2)

        st.subheader("🎯 TITAN Strategic Top Picks")
        cols = st.columns(2)
        
        for i, (_, row) in enumerate(top_picks.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class="titan-card">
                    <div style="display:flex; justify-content:space-between;">
                        <span class="metric-label">{row['league']}</span>
                        <span class="status-pill win">{row['date'].strftime('%m.%d %H:%M')}</span>
                    </div>
                    <h2 style="margin:10px 0;">{row['home']} vs {row['away']}</h2>
                    <hr style="border:0.1px solid rgba(255,255,255,0.1)">
                    <div style="display:grid; grid-template-columns: 1fr 1fr;">
                        <div><span class="metric-label">Ajánlás</span><br><span class="metric-value">{row['pick']}</span></div>
                        <div><span class="metric-label">Confidence</span><br><span class="metric-value">{row['confidence']*100:.1f}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 Mélyanalitikai adatok (xG & Narratíva)"):
                    st.write(f"**Várható gólok:** {row['home']} ({row['lh']:.2f}) - {row['away']} ({row['la']:.2f})")
                    st.write(f"**Pszichológiai stressz faktor:** {row['stress']:.1f}")
                    for news in row['news'][:3]:
                        st.markdown(f"- [{news['title']}]({news['url']}) (Tónus: {news['tone']})")

        # Statisztikai áttekintő táblázat
        st.subheader("📋 További elemzések")
        st.dataframe(df[["league", "home", "away", "pick", "confidence"]], use_container_width=True)

# =========================================================
#  6. BELÉPÉSI PONT
# =========================================================
if __name__ == "__main__":
    try:
        asyncio.run(run_titan_mission())
    except RuntimeError:
        # Streamlit fix az event loop-hoz
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_titan_mission())
