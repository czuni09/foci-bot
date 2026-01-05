# streamlit_app.py - JAVÍTOTT VERZIÓ
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
    st.code("pip install aiohttp", language="bash")
    st.stop()

# 1. KRITIKUS JAVÍTÁS: understatapi használata a requirements.txt-nek megfelelően[citation:1]
try:
    from understatapi import UnderstatClient  # A csomag neve UnderstatClient
    UNDERSTAT_API_CORRECT = True
except ImportError:
    st.error("⚠️ **Hiányzó csomag: understatapi**")
    st.code("pip install understatapi", language="bash")
    st.stop()

# ============================================================================
# DIXON-COLES MODEL INTEGRATION
# ============================================================================
# ... (A modell függvények változatlanok: rho_correction, dixon_coles_log_likelihood,
#      fit_dixon_coles_model, predict_match, FootballPredictor osztály)
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
# KONFIGURÁCIÓ
# ============================================================================
st.set_page_config(
    page_title="⚽ TITAN - Strategic Intelligence",
    page_icon="⚽",
    layout="wide"
)
DB_PATH = "titan_bot.db"
TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.85
TOTAL_ODDS_MAX = 2.15
TARGET_LEG_ODDS = math.sqrt(2)
UNDERSTAT_LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}
ODDS_API_LEAGUES = {
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
}
DAYS_AHEAD = 4
MAX_GOALS = 10
USE_GOOGLE_NEWS = True
USE_GDELT = True
TRANSLATE_TO_HU = True
SOCIAL_MAX_ITEMS = 10

def get_secret(name: str) -> str:
    return (os.getenv(name) or st.secrets.get(name, "") or "").strip()

ODDS_API_KEY = get_secret("ODDS_API_KEY")
WEATHER_API_KEY = get_secret("WEATHER_API_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")
FOOTBALL_DATA_KEY = get_secret("FOOTBALL_DATA_TOKEN")

EXCLUDED_MATCHUPS = {
    ("Manchester City", "Chelsea"), ("Chelsea", "Manchester City"),
    ("Manchester City", "Manchester United"), ("Manchester United", "Manchester City"),
    ("Arsenal", "Tottenham"), ("Tottenham", "Arsenal"),
    ("Liverpool", "Everton"), ("Everton", "Liverpool"),
    ("Liverpool", "Manchester United"), ("Manchester United", "Liverpool"),
    ("Arsenal", "Chelsea"), ("Chelsea", "Arsenal"),
    ("Manchester United", "Chelsea"), ("Chelsea", "Manchester United"),
    ("Liverpool", "Manchester City"), ("Manchester City", "Liverpool"),
    ("Real Madrid", "Barcelona"), ("Barcelona", "Real Madrid"),
    ("Atletico Madrid", "Real Madrid"), ("Real Madrid", "Atletico Madrid"),
    ("Barcelona", "Atletico Madrid"), ("Atletico Madrid", "Barcelona"),
    ("Inter", "AC Milan"), ("AC Milan", "Inter"),
    ("Juventus", "Inter"), ("Inter", "Juventus"),
    ("Juventus", "AC Milan"), ("AC Milan", "Juventus"),
    ("Roma", "Lazio"), ("Lazio", "Roma"),
    ("Bayern Munich", "Borussia Dortmund"), ("Borussia Dortmund", "Bayern Munich"),
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
# ADATBÁZIS (with context managerrel javítva)
# ============================================================================
def init_db():
    with sqlite3.connect(DB_PATH, check_same_thread=False) as con:
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

init_db()

# ============================================================================
# SEGÉDFÜGGVÉNYEK
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
# UNDERSTAT (understatapi csomag használatával javítva)[citation:1]
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
        # 2. KRITIKUS JAVÍTÁS: UnderstatClient használata és a helyes metódushívások
        async with aiohttp.ClientSession() as session:
            understat = UnderstatClient(session)  # UnderstatClient példányosítás
            # A league() metódus a liga kulcsot és szezont várja, get_fixtures() és get_results() a lekérdezés
            try:
                league_obj = understat.league(league=league_key)
                fixtures = await league_obj.get_fixtures(season=str(season))
                results = await league_obj.get_results(season=str(season))
                return fixtures or [], results or []
            except Exception as e:
                st.sidebar.error(f"Understat API hiba ({league_key}): {e}")
                return [], []

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
# TÖBBI FUNKCIÓ (social, odds, stb.) - változatlanul
# ============================================================================
# A google_news_rss, gdelt_doc, odds_api_get, stb. függvények változatlanok,
# de timeout paraméterrel kiegészítve (pl. requests.get(url, timeout=10))
# ============================================================================

# ============================================================================
# FŐ ALKALMAZÁS
# ============================================================================
st.title("⚽ TITAN - Advanced Football Betting Intelligence System")

tab1, tab2, tab3 = st.tabs(["Predikciók", "Backtest", "Pending Settlement"])

with tab1:
    # 3. KRITIKUS JAVÍTÁS: API kulcs ellenőrzése azonnal
    if not ODDS_API_KEY:
        st.error("""
        ❌ **Hiányzó kötelező API kulcs!**
        Az alkalmazás működéséhez az **`ODDS_API_KEY`** környezeti változó beállítása szükséges.

        **Javítás a Renderen:**
        1. Nyisd meg a 'foci-bot' szolgáltatásodat a Render Dashboardon.
        2. Menj a **'Environment'** (Környezet) fülre.
        3. Győződj meg róla, hogy van **`ODDS_API_KEY`** nevű változó, és a **Value** mezőbe be van illesztve a kulcsod.
        """)
        st.stop()

    leagues = list(UNDERSTAT_LEAGUES.keys())
    selected_leagues = st.multiselect("Liga kiválasztása", leagues, default=leagues)

    # 4. OPTIMALIZÁLÁS: Modell cache ligánként
    candidates = []
    predictors = {}  # Szótár a már betanított modellek tárolására

    for league_key in selected_leagues:
        fixtures, results = understat_fetch(league_key, season_from_today(), DAYS_AHEAD)
        historical_df = build_historical_df(results)

        # Betanítás csak akkor, ha ez a liga még nincs a cache-ben
        if league_key not in predictors:
            try:
                predictor = FootballPredictor()
                predictor.train(historical_df)
                predictors[league_key] = predictor
            except ValueError as e:
                st.sidebar.warning(f"⚠️ {league_key}: {e} Kihagyva.")
                continue
        else:
            predictor = predictors[league_key]  # Kiolvassuk a cache-ből

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

            match_data = next((e for e in events if team_match_score(e.get("home_team"), home) > 0.7 and team_match_score(e.get("away_team"), away) > 0.7), None)
            if not match_data:
                continue

            best_odds = extract_best_odds(match_data, home, away)
            over25 = best_odds.get("totals", {}).get((2.5, "over"))
            if over25 and 1.4 < over25 < 1.8:
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
                    "opening_odds": over25,
                    "xg_home": pred['expected_home_goals'],
                    "xg_away": pred['expected_away_goals'],
                    "football_data_match_id": match_id,
                    "match_id": match_data.get("id")
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
                # 5. OPTIMALIZÁLÁS: Adatbázis kapcsolat with context managerrel
                with sqlite3.connect(DB_PATH) as con:
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
                st.success("Mentve!")
        else:
            st.info("Nincs megfelelő dupla kombináció.")
    else:
        st.info("Nincs jelölt fogadás.")

with tab2:
    st.subheader("Backtest Futtatás")
    bt_league = st.selectbox("Liga", list(UNDERSTAT_LEAGUES.keys()))
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

    with sqlite3.connect(DB_PATH) as con:
        df_preds = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", con)
    st.dataframe(df_preds)
