import os
import re
import math
import sqlite3
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd
import requests

# =========================
# KÖTELEZŐ IMPORTOK (hiba esetén hard stop)
# =========================
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    st.error("⚠️ **Hiányzó csomag: aiohttp**")
    st.code("pip install aiohttp", language="bash")
    st.stop()

try:
    from understat import Understat
    UNDERSTAT_AVAILABLE = True
except ImportError:
    UNDERSTAT_AVAILABLE = False
    st.error("⚠️ **Hiányzó csomag: understat**")
    st.code("pip install understat", language="bash")
    st.stop()

# Feedparser - TELEPÍTETLEN KORLÁTOZÁS
FEEDPARSER_AVAILABLE = False
# Feedparser NEM kerül importálásra - helyette alternatív megoldás

# =========================
# STREAMLIT ALAPBEÁLLÍTÁS
# =========================
st.set_page_config(
    page_title="⚽ Football Intelligence System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# SECRETS / ENV
# =========================
def get_secret(name: str) -> str:
    return (os.getenv(name) or st.secrets.get(name, "") or "").strip()

ODDS_API_KEY = get_secret("ODDS_API_KEY")
WEATHER_API_KEY = get_secret("WEATHER_API_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")
FOOTBALL_DATA_TOKEN = get_secret("FOOTBALL_DATA_TOKEN")
MYMEMORY_EMAIL = get_secret("MYMEMORY_EMAIL")

# =========================
# GLOBÁLIS KONFIGURÁCIÓ
# =========================
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
TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.85
TOTAL_ODDS_MAX = 2.15
TARGET_LEG_ODDS = math.sqrt(2)

# Social & hírek - feedparser nélkül
USE_GOOGLE_NEWS = False  # Google RSS nem működik feedparser nélkül
USE_GDELT = True
TRANSLATE_TO_HU = True
SOCIAL_MAX_ITEMS = 8

NEGATIVE_KEYWORDS = [
    "injury", "injured", "ruled out", "out", "doubtful", "sidelined",
    "suspended", "suspension", "ban", "scandal", "arrest", "police",
    "court", "divorce", "wife", "girlfriend", "partner", "family",
]

DB_PATH = "football_bot.db"

# =========================
# DERBY / RANGADÓ KIZÁRÁS
# =========================
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

# =========================
# ADATBÁZIS
# =========================
def get_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db():
    con = get_db()
    cur = con.cursor()
    cur.execute("""
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
    con.commit()
    con.close()

init_db()

# =========================
# SEGÉDFÜGGVÉNYEK
# =========================
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
    s = re.sub(r"[^a-z0-9áéíóöőúüű\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    repl = {
        "manchester utd": "manchester united",
        "man utd": "manchester united",
        "bayern munchen": "bayern munich",
        "internazionale": "inter",
        "psg": "paris saint germain",
    }
    return repl.get(s, s)

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

# =========================
# POISSON MODELL
# =========================
def poisson_pmf(lmb, k):
    return math.exp(-lmb) * (lmb ** k) / math.factorial(k)

def prob_over_25(lh, la, max_goals=MAX_GOALS):
    p = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j >= 3:
                p += poisson_pmf(lh, i) * poisson_pmf(la, j)
    return clamp(p, 0.0, 1.0)

def prob_btts(lh, la):
    p = 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh + la))
    return clamp(p, 0.0, 1.0)

def prob_1x2(lh, la, max_goals=MAX_GOALS):
    ph = pdw = pa = 0.0
    for i in range(max_goals + 1):
        pi = poisson_pmf(lh, i)
        for j in range(max_goals + 1):
            pj = poisson_pmf(la, j)
            if i > j:
                ph += pi * pj
            elif i == j:
                pdw += pi * pj
            else:
                pa += pi * pj
    s = ph + pdw + pa
    if s > 0:
        ph, pdw, pa = ph / s, pdw / s, pa / s
    return clamp(ph, 0.0, 1.0), clamp(pdw, 0.0, 1.0), clamp(pa, 0.0, 1.0)

# =========================
# UNDERSTAT ASYNC FETCH
# =========================
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

def build_team_xg_profiles(results: list):
    prof = {}
    def ensure(team):
        prof.setdefault(team, {"home_for": [], "home_against": [], "away_for": [], "away_against": []})

    for m in results or []:
        h = clean_team(((m.get("h") or {}).get("title")))
        a = clean_team(((m.get("a") or {}).get("title")))
        xgh = safe_float(((m.get("xG") or {}).get("h")))
        xga = safe_float(((m.get("xG") or {}).get("a")))
        if not h or not a or xgh is None or xga is None:
            continue
        ensure(h); ensure(a)
        prof[h]["home_for"].append(xgh)
        prof[h]["home_against"].append(xga)
        prof[a]["away_for"].append(xga)
        prof[a]["away_against"].append(xgh)

    out = {}
    for team, d in prof.items():
        hf = d["home_for"]; ha = d["home_against"]
        af = d["away_for"]; aa = d["away_against"]
        out[team] = {
            "home_xg_for": sum(hf)/len(hf) if hf else None,
            "home_xg_against": sum(ha)/len(ha) if ha else None,
            "away_xg_for": sum(af)/len(af) if af else None,
            "away_xg_against": sum(aa)/len(aa) if aa else None,
            "n_home": len(hf),
            "n_away": len(af),
        }
    return out

def expected_goals_from_profiles(home: str, away: str, prof: dict, base=1.35):
    ph = prof.get(home, {})
    pa = prof.get(away, {})

    lh_parts = []
    if ph.get("home_xg_for") is not None: lh_parts.append(ph["home_xg_for"])
    if pa.get("away_xg_against") is not None: lh_parts.append(pa["away_xg_against"])
    lh = sum(lh_parts)/len(lh_parts) if lh_parts else base

    la_parts = []
    if pa.get("away_xg_for") is not None: la_parts.append(pa["away_xg_for"])
    if ph.get("home_xg_against") is not None: la_parts.append(ph["home_xg_against"])
    la = sum(la_parts)/len(la_parts) if la_parts else base

    lh = clamp(lh, 0.2, 3.5)
    la = clamp(la, 0.2, 3.5)

    n_home = int(ph.get("n_home", 0) or 0)
    n_away = int(pa.get("n_away", 0) or 0)
    return lh, la, n_home, n_away

# =========================
# MAGYAR FORDÍTÁS
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def translate_en_to_hu(text: str) -> str:
    t = (text or "").strip()
    if not t or not TRANSLATE_TO_HU:
        return t
    try:
        url = "https://api.mymemory.translated.net/get"
        params = {"q": t, "langpair": "en|hu"}
        if MYMEMORY_EMAIL:
            params["de"] = MYMEMORY_EMAIL
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        out = ((data.get("responseData") or {}).get("translatedText") or "").strip()
        return out if out else t
    except Exception:
        return t

# =========================
# SOCIAL SIGNALS - FEEDPARSER NÉLKÜL
# =========================
def count_neg_hits(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for k in NEGATIVE_KEYWORDS if k in t)

def google_news_api_fallback(query: str, limit=8):
    """Google News alternatíva RSS helyett - NewsAPI vagy GDELT"""
    # NewsAPI-t használunk ha van kulcs, különben csak GDELT
    if NEWS_API_KEY:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": limit,
                "apiKey": NEWS_API_KEY
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                articles = data.get("articles", [])
                out = []
                for a in articles:
                    title = a.get("title", "")
                    out.append({
                        "title": title,
                        "title_hu": translate_en_to_hu(title),
                        "link": a.get("url", ""),
                        "published": a.get("publishedAt", ""),
                        "source": a.get("source", {}).get("name", ""),
                    })
                return out
        except Exception:
            pass
    return []  # Ha nincs API kulcs vagy hiba

def gdelt_doc(query: str, maxrecords=8):
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": maxrecords,
            "sort": "HybridRel",
        }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", []) or []
        out = []
        for a in arts:
            title = a.get("title", "")
            out.append({
                "title": title,
                "title_hu": translate_en_to_hu(title),
                "url": a.get("url", ""),
                "domain": a.get("domain", ""),
                "seendate": a.get("seendate", ""),
                "tone": a.get("tone", None),
            })
        return out
    except Exception:
        return []

def fetch_social_signals(home: str, away: str):
    neg_terms = ["injury", "suspended", "scandal", "arrest", "divorce", "out", "doubt"]
    news_q = f'("{home}" OR "{away}") AND ({" OR ".join(neg_terms)})'
    gdelt_q = f'("{home}" OR "{away}") ({" OR ".join(neg_terms)})'

    social = {"news": [], "gdelt": [], "neg_hits": 0, "risk_penalty": 0.0}

    try:
        if USE_GOOGLE_NEWS:
            # Csak NewsAPI-t használunk
            social["news"] = google_news_api_fallback(news_q, SOCIAL_MAX_ITEMS)
            social["neg_hits"] += sum(count_neg_hits(x.get("title", "")) for x in social["news"])

        if USE_GDELT:
            social["gdelt"] = gdelt_doc(gdelt_q, SOCIAL_MAX_ITEMS)
            for a in social["gdelt"]:
                social["neg_hits"] += count_neg_hits(a.get("title", ""))
                tone = a.get("tone")
                if isinstance(tone, (int, float)) and tone < -4:
                    social["neg_hits"] += 1
    except Exception:
        pass

    neg = social["neg_hits"]
    if neg <= 0:
        social["risk_penalty"] = 0.0
    elif neg == 1:
        social["risk_penalty"] = 0.05
    elif neg == 2:
        social["risk_penalty"] = 0.08
    elif 3 <= neg <= 4:
        social["risk_penalty"] = 0.12
    else:
        social["risk_penalty"] = 0.15

    return social

# =========================
# IDŐJÁRÁS
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def get_weather_basic(city_guess="London"):
    if not WEATHER_API_KEY:
        return {"temp": None, "desc": "—", "wind": None, "humidity": None}
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city_guess,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "lang": "hu",
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return {
            "temp": safe_float((data.get("main") or {}).get("temp")),
            "desc": ((data.get("weather") or [{}])[0] or {}).get("description", "—"),
            "wind": safe_float((data.get("wind") or {}).get("speed")),
            "humidity": safe_float((data.get("main") or {}).get("humidity")),
        }
    except Exception:
        return {"temp": None, "desc": "—", "wind": None, "humidity": None}

# =========================
# ODDS API
# =========================
@st.cache_data(ttl=120, show_spinner=False)
def odds_api_get(league_key: str):
    if not ODDS_API_KEY:
        return {"ok": False, "events": [], "msg": "Nincs ODDS_API_KEY"}
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return {"ok": False, "events": [], "msg": f"HTTP {r.status_code}"}
        return {"ok": True, "events": r.json(), "msg": "OK"}
    except Exception as e:
        return {"ok": False, "events": [], "msg": str(e)}

def extract_best_odds(match_data: dict, home: str, away: str):
    bookmakers = match_data.get("bookmakers", []) or []

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

    spreads = {}
    for b in bookmakers:
        for mk in b.get("markets", []):
            if mk.get("key") != "spreads":
                continue
            for o in mk.get("outcomes", []):
                nm = o.get("name", "")
                pt = safe_float(o.get("point"))
                pr = safe_float(o.get("price"))
                if pt is None or pr is None:
                    continue
                key = (float(pt), nm)
                spreads.setdefault(key, []).append(pr)

    best_spreads = {}
    for key, prices in spreads.items():
        best_spreads[key] = max(prices)

    return {"h2h": best_h2h, "totals": best_totals, "spreads": best_spreads}

# =========================
# FOOTBALL-DATA.ORG
# =========================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_TOKEN} if FOOTBALL_DATA_TOKEN else {}

@st.cache_data(ttl=300, show_spinner=False)
def fd_find_match_id(home: str, away: str, kickoff_utc: datetime):
    if not FOOTBALL_DATA_TOKEN or not kickoff_utc:
        return None
    date_from = (kickoff_utc.date() - timedelta(days=1)).isoformat()
    date_to = (kickoff_utc.date() + timedelta(days=1)).isoformat()
    try:
        url = "https://api.football-data.org/v4/matches"
        params = {"dateFrom": date_from, "dateTo": date_to}
        r = requests.get(url, headers=fd_headers(), params=params, timeout=12)
        r.raise_for_status()
        candidates = r.json().get("matches", [])
    except Exception:
        return None
    best = (0.0, None)
    for m in candidates:
        try:
            fd_home = (m.get("homeTeam") or {}).get("name", "")
            fd_away = (m.get("awayTeam") or {}).get("name", "")
            fd_utc = parse_dt(m.get("utcDate"))
        except Exception:
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
    if not FOOTBALL_DATA_TOKEN or not match_id:
        return None
    try:
        url = f"https://api.football-data.org/v4/matches/{match_id}"
        r = requests.get(url, headers=fd_headers(), timeout=12)
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
    except Exception:
        return None

# =========================
# PONTOZÁS & AJÁNLÁS
# =========================
def score_bet_candidate(bet: dict, xg_home: float, xg_away: float, social: dict):
    odds = safe_float(bet.get("odds"), 1.0) or 1.0
    bet_type = bet.get("bet_type", "")
    total_xg = xg_home + xg_away

    odds_diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (odds_diff / 0.6)))

    xg_score = 0.0
    if bet_type == "H2H":
        selection = bet.get("selection", "")
        is_home = team_match_score(selection, bet.get("home", "")) >= 0.7
        xg_diff = xg_home - xg_away if is_home else xg_away - xg_home
        if xg_diff > 0.5:
            xg_score = 20.0
        elif xg_diff > 0.2:
            xg_score = 10.0
        else:
            xg_score = -5.0
    elif bet_type == "TOTALS":
        line = safe_float(bet.get("line"), 2.5)
        selection = bet.get("selection", "").lower()
        if selection == "over" and total_xg > line + 0.3:
            xg_score = 25.0
        elif selection == "under" and total_xg < line - 0.3:
            xg_score = 25.0
        elif selection == "over" and total_xg > line:
            xg_score = 15.0
        elif selection == "under" and total_xg < line:
            xg_score = 15.0
        else:
            xg_score = 0.0
    elif bet_type == "SPREADS":
        xg_diff = xg_home - xg_away
        line = safe_float(bet.get("line"), 0.0)
        selection = bet.get("selection", "")
        if selection == "HOME":
            if xg_diff + line > 0.5:
                xg_score = 20.0
            elif xg_diff + line > 0:
                xg_score = 10.0
            else:
                xg_score = -5.0
        else:
            if -xg_diff + line > 0.5:
                xg_score = 20.0
            elif -xg_diff + line > 0:
                xg_score = 10.0
            else:
                xg_score = -5.0

    social_pen = social["risk_penalty"] * 100

    weather = get_weather_basic(bet.get("home", "London").split()[-1])
    weather_pen = 0.0
    if weather.get("wind") is not None and weather["wind"] >= 12:
        weather_pen -= 8
    if isinstance(weather.get("desc"), str) and any(x in weather["desc"].lower() for x in ["eső", "zápor", "vihar"]):
        weather_pen -= 6

    raw_score = 50.0 + odds_score + xg_score - social_pen + weather_pen
    final_score = clamp(raw_score, 0.0, 100.0)

    why_parts = []
    why_parts.append(f"**Odds:** {odds:.2f} (cél: {TARGET_LEG_ODDS:.2f})")
    why_parts.append(f"**xG alap:** {xg_home:.2f} vs {xg_away:.2f} (össz: {total_xg:.2f})")
    
    if xg_score > 15:
        why_parts.append("✅ **xG erősen támogatja** ezt a fogadást.")
    elif xg_score > 0:
        why_parts.append("⚖️ **xG enyhén támogatja**.")
    else:
        why_parts.append("⚠️ **xG nem támogatja** erősen.")
    
    if social["neg_hits"] > 0:
        why_parts.append(f"⚠️ **Negatív hírek:** {social['neg_hits']} találat (-{social_pen:.0f} pont).")
    else:
        why_parts.append("✅ **Nincs negatív hír**.")
    
    if weather.get("temp") is not None:
        why_parts.append(f"🌤️ **Időjárás:** {weather['temp']:.0f}°C, {weather.get('desc','—')} (szél: {weather.get('wind','?')} m/s).")
    
    if weather_pen < 0:
        why_parts.append("🌪️ **Időjárás rizikó** (magas szél/eső).")

    reasoning = "\n".join(why_parts)
    return final_score, reasoning

def pick_best_duo(candidates: list):
    if len(candidates) < 2:
        return [], 0.0
    best = (None, None, -1e18, 0.0)
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            if a.get("match_id") == b.get("match_id"):
                continue
            total_odds = float(a.get("odds", 0.0)) * float(b.get("odds", 0.0))
            if not (TOTAL_ODDS_MIN <= total_odds <= TOTAL_ODDS_MAX):
                continue
            closeness = 1.0 - min(1.0, abs(total_odds - TARGET_TOTAL_ODDS) / 0.20)
            combined_score = float(a.get("score", 0.0)) + float(b.get("score", 0.0))
            utility = combined_score + 25.0 * closeness
            if utility > best[2]:
                best = (i, j, utility, total_odds)
    if best[0] is None:
        top2 = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:2]
        if len(top2) < 2:
            return [], 0.0
        return top2, float(top2[0]["odds"]) * float(top2[1]["odds"])
    return [candidates[best[0]], candidates[best[1]]], best[3]

# =========================
# MAIN UI
# =========================
st.title("⚽ Football Intelligence System")
st.caption("Understat xG • Odds API • Hírek • Időjárás • Magyar fordítás • Derby kizárás • Duplázó algoritmus")

status_cols = st.columns(6)
with status_cols[0]:
    st.markdown(f"**Understat** {'✅' if UNDERSTAT_AVAILABLE else '❌'}")
with status_cols[1]:
    st.markdown(f"**Odds API** {'✅' if ODDS_API_KEY else '❌'}")
with status_cols[2]:
    st.markdown(f"**Hírek** {'✅' if (NEWS_API_KEY or USE_GDELT) else '❌'}")
with status_cols[3]:
    st.markdown(f"**Időjárás** {'✅' if WEATHER_API_KEY else '❌'}")
with status_cols[4]:
    st.markdown(f"**Fordítás** {'✅' if TRANSLATE_TO_HU else '❌'}")
with status_cols[5]:
    st.markdown(f"**Feedparser** {'❌' if not FEEDPARSER_AVAILABLE else '✅'}")

st.markdown("---")

selected_leagues = st.multiselect(
    "Válassz ligákat:",
    options=list(UNDERSTAT_LEAGUES.keys()),
    default=["epl", "la_liga"],
    format_func=lambda x: UNDERSTAT_LEAGUES.get(x, x)
)

run_analysis = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)

if run_analysis and selected_leagues:
    season = season_from_today()
    all_candidates = []
    errors = []

    with st.spinner("Adatgyűjtés folyamatban (Understat, Odds API, hírek, időjárás)…"):
        for lk in selected_leagues:
            try:
                fixtures, results = understat_fetch(lk, season, DAYS_AHEAD)
            except Exception as e:
                errors.append(f"{UNDERSTAT_LEAGUES.get(lk, lk)}: {e}")
                continue

            prof = build_team_xg_profiles(results)

            for m in fixtures:
                home = clean_team(((m.get("h") or {}).get("title")))
                away = clean_team(((m.get("a") or {}).get("title")))
                kickoff = parse_dt(m.get("datetime", ""))
                if not home or not away or not kickoff:
                    continue

                if is_excluded_match(lk, home, away):
                    continue

                lh, la, n_home, n_away = expected_goals_from_profiles(home, away, prof)

                odds_league_key = None
                for ok, ov in ODDS_API_LEAGUES.items():
                    if UNDERSTAT_LEAGUES.get(lk, "").lower() in ov.lower():
                        odds_league_key = ok
                        break
                if not odds_league_key:
                    continue

                odds_data = odds_api_get(odds_league_key)
                if not odds_data["ok"]:
                    continue

                target_match = None
                for ev in odds_data["events"]:
                    ev_home = ev.get("home_team", "")
                    ev_away = ev.get("away_team", "")
                    if team_match_score(home, ev_home) >= 0.7 and team_match_score(away, ev_away) >= 0.7:
                        target_match = ev
                        break
                if not target_match:
                    continue

                best_odds = extract_best_odds(target_match, home, away)
                social = fetch_social_signals(home, away)

                for side, odds in best_odds["h2h"].items():
                    if side == "draw":
                        continue
                    selection = home if side == "home" else away
                    bet_type = "H2H"
                    all_candidates.append({
                        "match_id": f"{lk}:{home} vs {away}",
                        "home": home,
                        "away": away,
                        "league": UNDERSTAT_LEAGUES.get(lk, lk),
                        "kickoff": kickoff,
                        "bet_type": bet_type,
                        "selection": selection,
                        "line": None,
                        "odds": odds,
                        "xg_home": lh,
                        "xg_away": la,
                        "social": social,
                    })

                for (line, direction), odds in best_odds["totals"].items():
                    if line not in (2.5, 3.5, 1.5):
                        continue
                    bet_type = "TOTALS"
                    all_candidates.append({
                        "match_id": f"{lk}:{home} vs {away}",
                        "home": home,
                        "away": away,
                        "league": UNDERSTAT_LEAGUES.get(lk, lk),
                        "kickoff": kickoff,
                        "bet_type": bet_type,
                        "selection": direction.capitalize(),
                        "line": line,
                        "odds": odds,
                        "xg_home": lh,
                        "xg_away": la,
                        "social": social,
                    })

                for (line, team), odds in best_odds["spreads"].items():
                    if line not in (-1.0, -0.5, 0.5, 1.0):
                        continue
                    selection = "HOME" if team_match_score(team, home) >= team_match_score(team, away) else "AWAY"
                    bet_type = "SPREADS"
                    all_candidates.append({
                        "match_id": f"{lk}:{home} vs {away}",
                        "home": home,
                        "away": away,
                        "league": UNDERSTAT_LEAGUES.get(lk, lk),
                        "kickoff": kickoff,
                        "bet_type": bet_type,
                        "selection": selection,
                        "line": line,
                        "odds": odds,
                        "xg_home": lh,
                        "xg_away": la,
                        "social": social,
                    })

        for c in all_candidates:
            score, reasoning = score_bet_candidate(c, c["xg_home"], c["xg_away"], c["social"])
            c["score"] = score
            c["reasoning"] = reasoning

        best_duo, total_odds = pick_best_duo(all_candidates)

    st.markdown("---")
    st.subheader(f"🎯 TOP 2 AJÁNLÁS (össz-odds: {total_odds:.2f})")

    if not best_duo:
        st.warning("Nincs elegendő adat a duplázó ajánlathoz.")
    else:
        for idx, pick in enumerate(best_duo):
            with st.expander(f"**{idx+1}. {pick['home']} vs {pick['away']}** ({pick['league']})", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Típus", f"{pick['bet_type']}")
                    line_display = pick['line'] if pick['line'] is not None else "-"
                    st.metric("Választás", f"{pick['selection']} {line_display}")
                with col2:
                    st.metric("Odds", f"{pick['odds']:.2f}")
                    st.metric("Pontszám", f"{pick['score']:.1f}/100")
                with col3:
                    st.metric("xG (H/A)", f"{pick['xg_home']:.2f} / {pick['xg_away']:.2f}")
                    st.metric("Össz xG", f"{pick['xg_home'] + pick['xg_away']:.2f}")

                st.markdown("#### 📊 Indoklás")
                st.markdown(pick["reasoning"])

                st.markdown("#### 📰 Legfrissebb hírek")
                social = pick.get("social", {})
                if social.get("news") or social.get("gdelt"):
                    news_display = []
                    for item in social.get("news", [])[:3]:
                        title_hu = item.get("title_hu", item.get("title", "Nincs cím"))
                        news_display.append(f"• **{title_hu}** (NewsAPI)")
                    for item in social.get("gdelt", [])[:2]:
                        title_hu = item.get("title_hu", item.get("title", "Nincs cím"))
                        news_display.append(f"• **{title_hu}** (GDELT)")
                    if news_display:
                        st.markdown("\n".join(news_display))
                    else:
                        st.info("Nincs releváns hír.")
                else:
                    st.info("Nincs releváns hír.")

                city_guess = pick["home"].split()[-1]
                weather = get_weather_basic(city_guess)
                if weather["temp"] is not None:
                    st.markdown(f"#### 🌤️ Időjárás ({city_guess})")
                    desc = weather['desc']
                    if TRANSLATE_TO_HU and desc != "—":
                        if not any(c in desc for c in 'áéíóöőúüű'):
                            desc = translate_en_to_hu(desc)
                    st.markdown(f"**{weather['temp']:.0f}°C**, {desc}, szél: {weather.get('wind', '?')} m/s")

                if st.button(f"💾 Mentés (pick {idx+1})", key=f"save_{idx}"):
                    match_id = fd_find_match_id(pick["home"], pick["away"], pick["kickoff"])
                    con = get_db()
                    cur = con.cursor()
                    cur.execute("""
                        INSERT INTO predictions
                        (created_at, match, home, away, league, kickoff_utc,
                         bet_type, market_key, selection, line, bookmaker, odds,
                         score, reasoning, xg_home, xg_away, football_data_match_id,
                         result, opening_odds, data_quality)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        now_utc().isoformat(),
                        f"{pick['home']} vs {pick['away']}",
                        pick["home"], pick["away"],
                        pick["league"],
                        pick["kickoff"].isoformat(),
                        pick["bet_type"],
                        pick.get("market_key", ""),
                        pick["selection"],
                        pick.get("line"),
                        "best_of",
                        pick["odds"],
                        pick["score"],
                        pick["reasoning"],
                        pick["xg_home"],
                        pick["xg_away"],
                        match_id,
                        "PENDING",
                        pick["odds"],
                        "LIVE"
                    ))
                    con.commit()
                    con.close()
                    st.success("Pick mentve az adatbázisba!")

        st.markdown("---")
        st.subheader("📈 Backtest / Előzmények")
        con = get_db()
        df_history = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 15", con)
        con.close()
        if not df_history.empty:
            df_display = df_history[["created_at", "match", "bet_type", "selection", "odds", "score", "result"]].copy()
            df_display["created_at"] = pd.to_datetime(df_display["created_at"]).dt.strftime("%Y.%m.%d %H:%M")
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("Még nincs mentett tipp.")

    if errors:
        st.warning("**Figyelmeztetések:**")
        for err in errors:
            st.markdown(f"• {err}")

else:
    st.info("Válassz ligákat és kattints az **Elemzés indítása** gombra.")

st.markdown("---")
st.caption(
    f"Football Intelligence System • Understat xG • Odds API • NewsAPI & GDELT • "
    f"OpenWeather • MyMemory fordítás • Derby kizárás • Duplázó algoritmus • "
    f"Frissítve: {datetime.now().astimezone().strftime('%Y.%m.%d %H:%M')}"
)
