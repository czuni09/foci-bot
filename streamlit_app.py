import os
import re
import math
import time
import sqlite3
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from difflib import SequenceMatcher
import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
from scipy.stats import poisson
from scipy.optimize import minimize
# Aiohttp import with fallback
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    st.error("⚠️ **Hiányzó csomag: aiohttp**")
    st.code("pip install aiohttp understat", language="bash")
    st.stop()
try:
    from understat import Understat
except ImportError:
    st.error("⚠️ **Hiányzó csomag: understat**")
    st.code("pip install understat", language="bash")
    st.stop()

# ============================================================================
# DIXON-COLES MODEL INTEGRATION (Reddit/algobetting based: time-weighted, xG instead of goals, rho correction for low scores)
# ============================================================================
def rho_correction(x, y, lambda_x, lambda_y, rho):
    if x == 0 and y == 0:
        return 1 - (lambda_x * lambda_y * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_x * rho)
    elif x == 1 and y == 0:
        return 1 + (lambda_y * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def dixon_coles_log_likelihood(params, home_goals, away_goals, home_team, away_team, time_diff=None, xi=0.0018):
    num_teams = len(np.unique(np.concatenate([home_team, away_team])))
    attack = params[:num_teams]
    defence = params[num_teams:2*num_teams]
    home_adv = params[2*num_teams]
    rho = params[2*num_teams + 1] if len(params) > 2*num_teams + 1 else 0.0

    team_idx = {team: i for i, team in enumerate(np.unique(np.concatenate([home_team, away_team])))}

    log_lik = 0.0
    for i in range(len(home_goals)):
        h_idx = team_idx[home_team[i]]
        a_idx = team_idx[away_team[i]]

        lambda_h = np.exp(attack[h_idx] + defence[a_idx] + home_adv)
        lambda_a = np.exp(attack[a_idx] + defence[h_idx])

        weight = np.exp(-xi * time_diff[i]) if time_diff is not None and xi > 0 else 1.0

        prob = rho_correction(home_goals[i], away_goals[i], lambda_h, lambda_a, rho)
        log_lik += weight * (poisson.logpmf(home_goals[i], lambda_h) + poisson.logpmf(away_goals[i], lambda_a) + np.log(max(prob, 0.0001)))

    return -log_lik

def fit_dixon_coles_model(df, xi=0.0018):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    max_date = df['date'].max()
    time_diff = (max_date - df['date']).dt.days.values

    teams = np.sort(np.unique(np.concatenate([df['home_team'], df['away_team']])))

    initial_attack = np.zeros(len(teams))
    initial_defence = np.zeros(len(teams))
    initial_home = 0.0
    initial_rho = 0.0

    initial_params = np.concatenate([initial_attack, initial_defence, [initial_home, initial_rho]])

    bounds = [(None, None)] * len(initial_params)
    bounds[-1] = (-0.2, 0.2)  # rho korlát

    result = minimize(dixon_coles_log_likelihood, initial_params,
                      args=(df['home_goals'].values, df['away_goals'].values,
                            df['home_team'].values, df['away_team'].values, time_diff, xi),
                      method='L-BFGS-B', bounds=bounds)

    params = result.x
    return params, teams

def predict_match(params, teams, home_team, away_team, max_goals=8):
    team_idx = {team: i for i, team in enumerate(teams)}
    num_teams = len(teams)
    attack = params[:num_teams]
    defence = params[num_teams:2*num_teams]
    home_adv = params[2*num_teams]
    rho = params[2*num_teams + 1] if len(params) > 2*num_teams + 1 else 0.0

    h_idx = team_idx.get(home_team, 0)  # Fallback
    a_idx = team_idx.get(away_team, 0)

    lambda_home = np.exp(attack[h_idx] + defence[a_idx] + home_adv)
    lambda_away = np.exp(attack[a_idx] + defence[h_idx])

    home_probs = poisson.pmf(np.arange(max_goals+1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals+1), lambda_away)

    score_matrix = np.outer(home_probs, away_probs)

    for x in range(max_goals+1):
        for y in range(max_goals+1):
            score_matrix[x, y] *= rho_correction(x, y, lambda_home, lambda_away, rho)

    score_matrix /= score_matrix.sum()

    home_win_prob = np.sum(np.tril(score_matrix, -1))
    draw_prob = np.sum(np.diag(score_matrix))
    away_win_prob = np.sum(np.triu(score_matrix, 1))

    expected_home_goals = lambda_home
    expected_away_goals = lambda_away
    btts_prob = np.sum(score_matrix[1:, 1:])
    over25_prob = 1 - np.sum(score_matrix[:3, :3])  # Összesen 0,1,2 gól

    return {
        'home_win': home_win_prob,
        'draw': draw_prob,
        'away_win': away_win_prob,
        'expected_home_goals': expected_home_goals,
        'expected_away_goals': expected_away_goals,
        'btts': btts_prob,
        'over_2.5': over25_prob,
        'score_matrix': score_matrix
    }

class FootballPredictor:
    def __init__(self):
        self.params = None
        self.teams = None
        self.xi = 0.0018  # Time decay ~3-4 hónap félértékidő (Reddit ajánlás)

    def prepare_data(self, matches_df):
        # Understat xG adatokkal, xG mint "goals" a modelben
        df = matches_df[['date', 'home_team', 'away_team', 'home_xg', 'away_xg']].copy()
        df.rename(columns={'home_xg': 'home_goals', 'away_xg': 'away_goals'}, inplace=True)
        df = df.dropna()
        return df

    def train(self, historical_df):
        df = self.prepare_data(historical_df)
        if len(df) < 10:
            raise ValueError("Túl kevés adat a modellhez!")
        self.params, self.teams = fit_dixon_coles_model(df, xi=self.xi)

    def predict(self, home_team, away_team):
        if self.params is None:
            raise ValueError("Modell nincs betanítva!")
        return predict_match(self.params, self.teams, home_team, away_team)

# ============================================================================
# CONFIGURATION (fórumok alapján: target odds finomhangolás, exclude big matches)
# ============================================================================
st.set_page_config(
    page_title="⚽ TITAN - Strategic Intelligence",
    page_icon="⚽",
    layout="wide"
)
# Database
DB_PATH = "titan_bot.db"
# Betting strategy (Reddit/SoccerBetting: value betting fókusz, 1.85-2.15 range)
TARGET_TOTAL_ODDS = 2.00 # Dupla target
TOTAL_ODDS_MIN = 1.85
TOTAL_ODDS_MAX = 2.15
TARGET_LEG_ODDS = math.sqrt(2) # ~1.414
# Understat leagues
UNDERSTAT_LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}
# Odds API leagues
ODDS_API_LEAGUES = {
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
}
# Time window
DAYS_AHEAD = 4
MAX_GOALS = 10
# Social signals
USE_GOOGLE_NEWS = True
USE_GDELT = True
TRANSLATE_TO_HU = True
SOCIAL_MAX_ITEMS = 10
# API Keys (from secrets/env)
def get_secret(name: str) -> str:
    return (os.getenv(name) or st.secrets.get(name, "") or "").strip()
ODDS_API_KEY = get_secret("ODDS_API_KEY")
WEATHER_API_KEY = get_secret("WEATHER_API_KEY")  # Ha kell később
NEWS_API_KEY = get_secret("NEWS_API_KEY")  # Ha kell
FOOTBALL_DATA_KEY = get_secret("FOOTBALL_DATA_TOKEN")
# ============================================================================
# EXCLUDED DERBIES & BIG RIVALRIES (fórumok: kerülni a volatilis meccseket)
# ============================================================================
EXCLUDED_MATCHUPS = {
    # EPL
    ("Manchester City", "Chelsea"), ("Chelsea", "Manchester City"),
    ("Manchester City", "Manchester United"), ("Manchester United", "Manchester City"),
    ("Arsenal", "Tottenham"), ("Tottenham", "Arsenal"),
    ("Liverpool", "Everton"), ("Everton", "Liverpool"),
    ("Liverpool", "Manchester United"), ("Manchester United", "Liverpool"),
    ("Arsenal", "Chelsea"), ("Chelsea", "Arsenal"),
    ("Manchester United", "Chelsea"), ("Chelsea", "Manchester United"),
    ("Liverpool", "Manchester City"), ("Manchester City", "Liverpool"),
    # La Liga
    ("Real Madrid", "Barcelona"), ("Barcelona", "Real Madrid"),
    ("Atletico Madrid", "Real Madrid"), ("Real Madrid", "Atletico Madrid"),
    ("Barcelona", "Atletico Madrid"), ("Atletico Madrid", "Barcelona"),
    # Serie A
    ("Inter", "AC Milan"), ("AC Milan", "Inter"),
    ("Juventus", "Inter"), ("Inter", "Juventus"),
    ("Juventus", "AC Milan"), ("AC Milan", "Juventus"),
    ("Roma", "Lazio"), ("Lazio", "Roma"),
    # Bundesliga
    ("Bayern Munich", "Borussia Dortmund"), ("Borussia Dortmund", "Bayern Munich"),
    # Ligue 1
    ("PSG", "Marseille"), ("Marseille", "PSG"),
}
EPL_BIG6 = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham"}
def is_excluded_match(league_key: str, home: str, away: str) -> bool:
    if (home, away) in EXCLUDED_MATCHUPS:
        return True
    if league_key == "epl" and home in EPL_BIG6 and away in EPL_BIG6:
        return True
    return False
# ============================================================================
# DATABASE SETUP (backtesthez is: store historical preds)
# ============================================================================
def init_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            match TEXT,
            home TEXT,
            away TEXT,
            league TEXT,
            kickoff_utc TEXT,
            bet_type TEXT,
            market_key TEXT,
            selection TEXT,
            line REAL,
            bookmaker TEXT,
            odds REAL,
            score REAL,
            reasoning TEXT,
            xg_home REAL,
            xg_away REAL,
            football_data_match_id INTEGER,
            result TEXT DEFAULT 'PENDING',
            settled_at TEXT,
            home_goals INTEGER,
            away_goals INTEGER,
            opening_odds REAL,
            closing_odds REAL,
            clv_percent REAL,
            data_quality TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT,
            league TEXT,
            season INT,
            roi REAL,
            hit_rate REAL,
            num_bets INT,
            details TEXT
        )
    """)
    con.commit()
    con.close()
init_db()
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)
def season_from_today() -> int:
    t = datetime.now().date()
    return t.year - 1 if t.month < 7 else t.year
def parse_dt(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except:
            return None
def fmt_local(dt):
    if not dt:
        return "—"
    try:
        return dt.astimezone().strftime("%Y.%m.%d %H:%M")
    except:
        return dt.strftime("%Y.%m.%d %H:%M")
def safe_float(x, default=None):
    try:
        return float(x)
    except:
        return default
def clamp(x, a, b):
    return max(a, min(b, x))
def clean_team(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name
def norm_team(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    replacements = {
        "manchester utd": "manchester united",
        "man utd": "manchester united",
        "bayern munchen": "bayern munich",
        "internazionale": "inter",
        "psg": "paris saint germain",
    }
    return replacements.get(s, s)
def team_match_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a2, b2 = norm_team(a), norm_team(b)
    if a2 == b2:
        return 1.0
    at = set(a2.split())
    bt = set(b2.split())
    token_score = len(at & bt) / max(1, len(at | bt))
    seq_score = SequenceMatcher(None, a2, b2).ratio()
    return max(token_score, seq_score)
# ============================================================================
# UNDERSTAT (async xG data, Reddit: use for momentum, set-piece korrekció)
# ============================================================================
def run_async(coro):
    try:
        asyncio.get_running_loop()
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    except RuntimeError:
        return asyncio.run(coro)
@st.cache_data(ttl=600, show_spinner=False)
def understat_fetch(league_key: str, season: int, days_ahead: int):
    async def _run():
        async with aiohttp.ClientSession() as session:
            u = Understat(session)
            fixtures = await u.get_league_fixtures(league_key, season)
            results = await u.get_league_results(league_key, season)
            return fixtures or [], results or []
   
    fixtures, results = run_async(_run())
   
    now = now_utc()
    limit = now + timedelta(days=days_ahead)
    fx = []
    for m in fixtures:
        dt = parse_dt(m.get("datetime", ""))
        if not dt:
            continue
        if now <= dt <= limit:
            fx.append(m)
    fx.sort(key=lambda x: x.get("datetime", ""))
    return fx, results
def build_historical_df(results: list[dict]):
    data = []
    for m in results or []:
        h = clean_team(((m.get("h") or {}).get("title")))
        a = clean_team(((m.get("a") or {}).get("title")))
        dt = parse_dt(m.get("datetime", ""))
        xgh = safe_float(((m.get("xG") or {}).get("h")))
        xga = safe_float(((m.get("xG") or {}).get("a")))
        if h and a and dt and xgh is not None and xga is not None:
            data.append({
                'date': dt,
                'home_team': h,
                'away_team': a,
                'home_xg': xgh,
                'away_xg': xga,
                'home_goals': m.get("goals", {}).get("h"),
                'away_goals': m.get("goals", {}).get("a")
            })
    return pd.DataFrame(data)
# ============================================================================
# SOCIAL SIGNALS (News + GDELT, fórumok: negatív hírek penalty)
# ============================================================================
NEG_KEYWORDS = [
    "injury", "injured", "ruled out", "out", "doubtful", "sidelined",
    "suspended", "suspension", "ban", "scandal", "arrest", "police"
]
def count_neg_hits(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for k in NEG_KEYWORDS if k in t)
@st.cache_data(ttl=3600, show_spinner=False)
def translate_en_to_hu(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    try:
        url = "https://api.mymemory.translated.net/get"
        params = {"q": t, "langpair": "en|hu"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = ((data.get("responseData") or {}).get("translatedText") or "").strip()
        return out if out else t
    except:
        return t
def google_news_rss(query: str, limit=10):
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    out = []
    for e in (feed.entries or [])[:limit]:
        title = e.get("title", "")
        out.append({
            "title": title,
            "title_hu": translate_en_to_hu(title) if TRANSLATE_TO_HU else title,
            "link": e.get("link", ""),
            "published": e.get("published", ""),
        })
    return out
def gdelt_doc(query: str, maxrecords=10):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "sort": "HybridRel",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    arts = data.get("articles", []) or []
    out = []
    for a in arts:
        title = a.get("title", "")
        out.append({
            "title": title,
            "title_hu": translate_en_to_hu(title) if TRANSLATE_TO_HU else title,
            "url": a.get("url", ""),
            "tone": a.get("tone", None),
        })
    return out
def fetch_social_signals(home: str, away: str):
    neg_terms = ["injury", "suspended", "scandal", "arrest"]
    gnews_q = f'({home} OR "{away}") AND ({" OR ".join(neg_terms)})'
    gdelt_q = f'({home} OR "{away}") ({" OR ".join(neg_terms)})'
   
    social = {"gnews": [], "gdelt": [], "neg_hits": 0}
   
    try:
        if USE_GOOGLE_NEWS:
            social["gnews"] = google_news_rss(gnews_q, SOCIAL_MAX_ITEMS)
            social["neg_hits"] += sum(count_neg_hits(x.get("title", "")) for x in social["gnews"])
       
        if USE_GDELT:
            social["gdelt"] = gdelt_doc(gdelt_q, SOCIAL_MAX_ITEMS)
            for a in social["gdelt"]:
                social["neg_hits"] += count_neg_hits(a.get("title", ""))
                tone = a.get("tone")
                if isinstance(tone, (int, float)) and tone < -4:
                    social["neg_hits"] += 1
    except Exception as e:
        st.warning(f"Social signals error: {e}")
   
    return social
def social_penalty(neg_hits: int) -> float:
    if neg_hits <= 0: return 0.00
    if neg_hits == 1: return 0.05
    if neg_hits == 2: return 0.08
    if 3 <= neg_hits <= 4: return 0.12
    return 0.15
# ============================================================================
# THE ODDS API (best prices, CLV tracking: store opening, later update closing)
# ============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def odds_api_get(league_key: str):
    if not ODDS_API_KEY:
        return {"ok": False, "events": [], "msg": "No ODDS_API_KEY"}
   
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
    }
   
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return {"ok": False, "events": [], "msg": f"HTTP {r.status_code}"}
        return {"ok": True, "events": r.json(), "msg": "OK"}
    except Exception as e:
        return {"ok": False, "events": [], "msg": str(e)}
def extract_best_odds(match_data: dict, home: str, away: str):
    bookmakers = match_data.get("bookmakers", []) or []
   
    # H2H (1X2)
    h2h_prices = {"home": [], "draw": [], "away": []}
    for b in bookmakers:
        for mk in b.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            for o in mk.get("outcomes", []):
                nm = o.get("name", "")
                pr = safe_float(o.get("price"))
                if pr is None:
                    continue
                if team_match_score(nm, home) >= 0.7:
                    h2h_prices["home"].append(pr)
                elif team_match_score(nm, away) >= 0.7:
                    h2h_prices["away"].append(pr)
                elif "draw" in nm.lower():
                    h2h_prices["draw"].append(pr)
   
    best_h2h = {}
    for side, prices in h2h_prices.items():
        if prices:
            best_h2h[side] = max(prices)
   
    # Totals (Over/Under)
    totals = {}
    for b in bookmakers:
        for mk in b.get("markets", []):
            if mk.get("key") != "totals":
                continue
            for o in mk.get("outcomes", []):
                nm = (o.get("name") or "").lower()
                pt = safe_float(o.get("point"))
                pr = safe_float(o.get("price"))
                if pt is None or pr is None or nm not in ("over", "under"):
                    continue
                key = (float(pt), nm)
                totals.setdefault(key, []).append(pr)
   
    best_totals = {}
    for key, prices in totals.items():
        best_totals[key] = max(prices)
   
    return {"h2h": best_h2h, "totals": best_totals}
# ============================================================================
# FOOTBALL-DATA.ORG (match results for settling & backtest)
# ============================================================================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_KEY} if FOOTBALL_DATA_KEY else {}
@st.cache_data(ttl=300, show_spinner=False)
def fd_find_match_id(home: str, away: str, kickoff_utc: datetime):
    if not FOOTBALL_DATA_KEY or not kickoff_utc:
        return None
   
    date_from = (kickoff_utc.date() - timedelta(days=1)).isoformat()
    date_to = (kickoff_utc.date() + timedelta(days=1)).isoformat()
   
    try:
        url = "https://api.football-data.org/v4/matches"
        params = {"dateFrom": date_from, "dateTo": date_to}
        r = requests.get(url, headers=fd_headers(), params=params, timeout=15)
        r.raise_for_status()
        candidates = r.json().get("matches", [])
    except:
        return None
   
    best = (0.0, None)
    for m in candidates:
        try:
            fd_home = (m.get("homeTeam") or {}).get("name", "")
            fd_away = (m.get("awayTeam") or {}).get("name", "")
            fd_utc = parse_dt(m.get("utcDate"))
        except:
            continue
       
        if not fd_home or not fd_away or not fd_utc:
            continue
       
        if abs((fd_utc - kickoff_utc).total_seconds()) > 8 * 3600:
            continue
       
        score = (team_match_score(home, fd_home) + team_match_score(away, fd_away)) / 2.0
        if score > best[0]:
            best = (score, m.get("id"))
   
    return best[1] if best[0] >= 0.60 else None
@st.cache_data(ttl=300, show_spinner=False)
def fd_get_result(match_id: int):
    if not FOOTBALL_DATA_KEY or not match_id:
        return None
   
    try:
        url = f"https://api.football-data.org/v4/matches/{match_id}"
        r = requests.get(url, headers=fd_headers(), timeout=15)
        r.raise_for_status()
        m = r.json()
       
        status = m.get("status", "")
        if status not in ["FINISHED", "AWARDED"]:
            return None
       
        score_ft = (m.get("score") or {}).get("fullTime", {}) or {}
        hg = score_ft.get("home")
        ag = score_ft.get("away")
       
        if hg is None or ag is None:
            return None
       
        return {"home_goals": int(hg), "away_goals": int(ag)}
    except:
        return None
# ============================================================================
# SCORING & RECOMMENDATION ENGINE (upgraded: xG efficiency, social penalty, forums: hedge if low score)
# ============================================================================
def score_bet_candidate(bet: dict, pred: dict, social: dict) -> tuple[float, str]:
    odds = safe_float(bet.get("odds"), 1.0) or 1.0
    bet_type = bet.get("bet_type", "")
   
    # Base score: odds proximity to target
    odds_diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (odds_diff / 0.6)))
   
    # Model alignment (Dixon-Coles probs)
    model_score = 0.0
    if bet_type == "H2H":
        selection = bet.get("selection", "").lower()
        if "home" in selection:
            model_score = pred['home_win'] * 100
        elif "away" in selection:
            model_score = pred['away_win'] * 100
        elif "draw" in selection:
            model_score = pred['draw'] * 100
   
    elif bet_type == "TOTALS":
        line = safe_float(bet.get("line"), 2.5)
        selection = bet.get("selection", "").lower()
        if selection == "over" and line == 2.5:
            model_score = pred['over_2.5'] * 100
        elif selection == "under" and line == 2.5:
            model_score = (1 - pred['over_2.5']) * 100
   
    # Social penalty
    neg_hits = social.get("neg_hits", 0)
    social_pen = social_penalty(neg_hits) * 100
   
    # Final score
    raw_score = 50.0 + odds_score + model_score - social_pen
    final_score = clamp(raw_score, 0.0, 100.0)
   
    # Reasoning
    why_parts = []
    why_parts.append(f"**Odds:** {odds:.2f} (cél: {TARGET_LEG_ODDS:.2f})")
    why_parts.append(f"**Model Prob:** {model_score/100:.2%}")
   
    if model_score > 60:
        why_parts.append("✅ Modell erősen támogatja")
    elif model_score > 50:
        why_parts.append("⚖️ Modell enyhén támogatja")
    else:
        why_parts.append("⚠️ Modell nem támogatja erősen")
   
    if neg_hits > 0:
        why_parts.append(f"⚠️ Negatív hírek: {neg_hits} találat (-{social_pen:.0f} pont)")
    else:
        why_parts.append("✅ Nincs negatív hír")
   
    reasoning = "\n".join(why_parts)
   
    return final_score, reasoning
def pick_best_duo(candidates: list[dict]) -> tuple[list[dict], float]:
    """Select best 2-leg parlay targeting 2.00 combined odds (fórumok: hedge fallback top2)"""
    if len(candidates) < 2:
        return [], 0.0
   
    best = (None, None, -1e18, 0.0)
    n = len(candidates)
   
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
           
            # Don't combine same match
            if a.get("match_id") == b.get("match_id"):
                continue
           
            total_odds = float(a.get("odds", 0.0)) * float(b.get("odds", 0.0))
           
            # Filter by odds range
            if not (TOTAL_ODDS_MIN <= total_odds <= TOTAL_ODDS_MAX):
                continue
           
            # Score combination
            closeness = 1.0 - min(1.0, abs(total_odds - TARGET_TOTAL_ODDS) / 0.20)
            combined_score = float(a.get("score", 0.0)) + float(b.get("score", 0.0))
            utility = combined_score + 25.0 * closeness
           
            if utility > best[2]:
                best = (i, j, utility, total_odds)
   
    if best[0] is None:
        # Fallback: top2 by score
        top2 = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:2]
        if len(top2) == 2:
            a, b = top2
            total_odds = a["odds"] * b["odds"]
            return [a, b], total_odds
        else:
            return [], 0.0
    else:
        return [candidates[best[0]], candidates[best[1]]], best[3]
# ============================================================================
# BACKTEST SYSTEM (Reddit/algobetting: simulate historical, ROI calculation)
# ============================================================================
def run_backtest(league_key: str, season: int):
    _, results = understat_fetch(league_key, season, 0)  # Csak results
    df = build_historical_df(results)
    if len(df) < 50:
        return {"roi": 0, "hit_rate": 0, "num_bets": 0, "details": "Túl kevés adat"}
    
    predictor = FootballPredictor()
    bets = []
    for idx in range(10, len(df)):  # Első 10 meccs training
        train_df = df.iloc[:idx]
        test_row = df.iloc[idx]
        try:
            predictor.train(train_df)
            pred = predictor.predict(test_row['home_team'], test_row['away_team'])
            # Szimulált odds: model prob -> implied odds + margin
            over_prob = pred['over_2.5']
            if over_prob > 0.55:  # Threshold for value
                implied_odds = 1 / over_prob
                sim_odds = implied_odds * 1.05  # +5% margin
                outcome = (test_row['home_goals'] + test_row['away_goals']) > 2.5
                profit = (sim_odds - 1) if outcome else -1
                bets.append(profit)
        except:
            continue
    
    if not bets:
        return {"roi": 0, "hit_rate": 0, "num_bets": 0, "details": "Nincs érvényes fogadás"}
    
    num_bets = len(bets)
    roi = sum(bets) / num_bets * 100
    hit_rate = sum(1 for p in bets if p > 0) / num_bets * 100
    details = f"ROI: {roi:.2f}%, Hit: {hit_rate:.2f}%, Bets: {num_bets}"
    
    # Store to db
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO backtests (run_at, league, season, roi, hit_rate, num_bets, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now_utc().isoformat(), league_key, season, roi, hit_rate, num_bets, details))
    con.commit()
    con.close()
    
    return {"roi": roi, "hit_rate": hit_rate, "num_bets": num_bets, "details": details}
# ============================================================================
# SETTLE PENDING PREDICTIONS (CLV calculation)
# ============================================================================
def settle_pending():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    pending = cur.execute("""
        SELECT id, football_data_match_id, opening_odds, league, home, away, kickoff_utc, bet_type, selection, line
        FROM predictions WHERE result = 'PENDING'
    """).fetchall()
    
    settled = 0
    for row in pending:
        pid, mid, opening, league, home, away, ko_str, btype, sel, line = row
        if not mid:
            continue
        res = fd_get_result(mid)
        if not res:
            continue
        hg, ag = res['home_goals'], res['away_goals']
        
        # Fetch current odds as closing (ha van API historical, különben approx)
        odds_data = odds_api_get(ODDS_API_LEAGUES.get(league, ""))
        match_data = next((e for e in odds_data.get("events", []) if team_match_score(e.get("home_team"), home) > 0.7 and team_match_score(e.get("away_team"), away) > 0.7), None)
        closing = 0
        if match_data:
            best_odds = extract_best_odds(match_data, home, away)
            if btype == "TOTALS" and sel.lower() == "over" and line == 2.5:
                closing = best_odds.get("totals", {}).get((2.5, "over"), 0)
        
        clv = ((closing - opening) / opening * 100) if opening and closing else 0
        
        # Determine result
        if btype == "TOTALS" and sel.lower() == "over" and line == 2.5:
            outcome = "WIN" if hg + ag > 2.5 else "LOSS"
        else:
            outcome = "UNKNOWN"
        
        cur.execute("""
            UPDATE predictions SET 
                result = ?, settled_at = ?, home_goals = ?, away_goals = ?, 
                closing_odds = ?, clv_percent = ?
            WHERE id = ?
        """, (outcome, now_utc().isoformat(), hg, ag, closing, clv, pid))
        settled += 1
    
    con.commit()
    con.close()
    return settled
# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
st.title("⚽ TITAN - Advanced Football Betting Intelligence System")

tab1, tab2, tab3 = st.tabs(["Predikciók", "Backtest", "Pending Settlement"])

with tab1:
    leagues = list(UNDERSTAT_LEAGUES.keys())
    selected_leagues = st.multiselect("Liga kiválasztása", leagues, default=leagues)
    
    candidates = []
    predictor = FootballPredictor()
    
    for league_key in selected_leagues:
        fixtures, results = understat_fetch(league_key, season_from_today(), DAYS_AHEAD)
        historical_df = build_historical_df(results)
        predictor.train(historical_df)
        
        odds_resp = odds_api_get(league_key)
        if not odds_resp["ok"]:
            st.warning(f"{league_key} odds error: {odds_resp['msg']}")
            continue
        
        events = odds_resp["events"]
        
        for fix in fixtures:
            home = clean_team(fix['h']['title'])
            away = clean_team(fix['a']['title'])
            kickoff = parse_dt(fix['datetime'])
            if not kickoff or is_excluded_match(league_key, home, away):
                continue
            
            match_id = fd_find_match_id(home, away, kickoff)
            
            pred = predictor.predict(home, away)
            social = fetch_social_signals(home, away)
            
            # Find matching event in odds
            match_data = next((e for e in events if team_match_score(e.get("home_team"), home) > 0.7 and team_match_score(e.get("away_team"), away) > 0.7), None)
            if not match_data:
                continue
            
            best_odds = extract_best_odds(match_data, home, away)
            
            # Create candidates (pl. Over 2.5, ha van value)
            over25 = best_odds.get("totals", {}).get((2.5, "over"))
            if over25 and 1.4 < over25 < 1.8:  # Leg range
                bet = {
                    "match": f"{home} vs {away}",
                    "home": home,
                    "away": away,
                    "league": league_key,
                    "kickoff_utc": kickoff.isoformat(),
                    "bet_type": "TOTALS",
                    "selection": "Over",
                    "line": 2.5,
                    "odds": over25,
                    "opening_odds": over25,  # Current as opening
                    "xg_home": pred['expected_home_goals'],
                    "xg_away": pred['expected_away_goals'],
                    "football_data_match_id": match_id,
                    "match_id": match_data.get("id")  # For uniqueness
                }
                bet["score"], bet["reasoning"] = score_bet_candidate(bet, pred, social)
                if bet["score"] > 50:
                    candidates.append(bet)
    
    if candidates:
        best_duo, total_odds = pick_best_duo(candidates)
        if best_duo:
            st.subheader("Ajánlott Dupla (Target ~2.00)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{best_duo[0]['match']}** - {best_duo[0]['selection']} {best_duo[0]['line']} @ {best_duo[0]['odds']:.2f}")
                st.markdown(best_duo[0]['reasoning'])
            with col2:
                st.markdown(f"**{best_duo[1]['match']}** - {best_duo[1]['selection']} {best_duo[1]['line']} @ {best_duo[1]['odds']:.2f}")
                st.markdown(best_duo[1]['reasoning'])
            st.success(f"Össz odds: {total_odds:.2f}")
            
            if st.button("Mentés DB-be"):
                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                for b in best_duo:
                    cur.execute("""
                        INSERT INTO predictions (created_at, match, home, away, league, kickoff_utc, 
                                                 bet_type, selection, line, odds, score, reasoning, 
                                                 xg_home, xg_away, football_data_match_id, opening_odds)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (now_utc().isoformat(), b['match'], b['home'], b['away'], b['league'], b['kickoff_utc'],
                          b['bet_type'], b['selection'], b['line'], b['odds'], b['score'], b['reasoning'],
                          b['xg_home'], b['xg_away'], b['football_data_match_id'], b['opening_odds']))
                con.commit()
                con.close()
                st.success("Mentve!")
        else:
            st.info("Nincs megfelelő dupla kombináció.")
    else:
        st.info("Nincs jelölt fogadás.")

with tab2:
    st.subheader("Backtest Futtatás")
    bt_league = st.selectbox("Liga", leagues)
    bt_season = st.number_input("Szezon", min_value=2015, max_value=season_from_today(), value=season_from_today()-1)
    if st.button("Backtest Indítás"):
        with st.spinner("Backtest fut..."):
            res = run_backtest(bt_league, bt_season)
        st.markdown(res['details'])

with tab3:
    st.subheader("Pending Settlement & CLV")
    if st.button("Settle Pending"):
        settled = settle_pending()
        st.success(f"{settled} predikció rendezve!")
    
    # Show db
    con = sqlite3.connect(DB_PATH)
    df_preds = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", con)
    con.close()
    st.dataframe(df_preds)


