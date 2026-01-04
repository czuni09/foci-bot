# streamlit_app.py
# TITAN v3 – Autonóm “Mission Control” (nincs kontroll panel)
# CORE: Understat xG -> Poisson -> pick + őszinte rizikó
# Signals (kulcs nélkül): Google News RSS + GDELT -> csak penalty / RIZIKÓ jelzés
# Opcionális: Weather (OpenWeather), Odds (The Odds API), NewsAPI (ha van kulcs, de nem kötelező)
# Backtest: automatikus pick-mentés + utólagos eredmény-ellenőrzés (Understat results alapján)

import os
import re
import math
import csv
import json
import asyncio
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus
from pathlib import Path

import aiohttp
import pandas as pd
import requests
import feedparser
import streamlit as st
from understat import Understat

# =========================================================
#  CONFIG (fixen beállítva – NINCS kontroll panel)
# =========================================================
st.set_page_config(page_title="TITAN – Mission Control", page_icon="🛰️", layout="wide")

LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}
DEFAULT_DAYS_AHEAD = 4
TOP_K = 2
MAX_GOALS = 10

# Social (kulcs nélküli) – alapból ON
USE_GOOGLE_NEWS_RSS = True
USE_GDELT = True
SOCIAL_MAX_ITEMS = 12
SHOW_SOCIAL_DETAILS_IN_CARD = True  # innovatív, de rövid (top 3 cím)

# Backtest / napló
PICKS_LOG_PATH = Path("picks_log.csv")  # helyi futtatásra stabil
AUTO_LOG_TOP_PICKS = True

# Weather (opcionális)
USE_WEATHER = True
WEATHER_CITY_MODE = "team_city"  # itt most egyszerű: ha nincs mapping -> skip
WEATHER_MAX_PENALTY = 0.08

# Odds (opcionális)
USE_ODDS = True

# =========================================================
#  Secrets / ENV (NINCS kulcs a kódban)
# =========================================================
def _secret(name: str) -> str:
    return (os.getenv(name) or st.secrets.get(name, "") or "").strip()

NEWS_API_KEY = _secret("NEWS_API_KEY")              # opcionális
WEATHER_API_KEY = _secret("WEATHER_API_KEY")        # opcionális
ODDS_API_KEY = _secret("ODDS_API_KEY")              # opcionális
FOOTBALL_DATA_TOKEN = _secret("FOOTBALL_DATA_TOKEN")# opcionális (most csak placeholder)

# =========================================================
#  UI – Mission Control (innováció)
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');

:root{
  --bg0:#06070c;
  --bg1:#0b1020;
  --bg2:#0e1630;
  --card: rgba(255,255,255,0.065);
  --card2: rgba(255,255,255,0.045);
  --border: rgba(255,255,255,0.11);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.66);
  --good:#4ef0a3;
  --warn:#ffd166;
  --bad:#ff5c8a;
  --accent:#79a6ff;
  --accent2:#b387ff;
  --cyan:#62e6ff;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp{
  background:
    radial-gradient(950px 540px at 16% 10%, rgba(121,166,255,0.18), transparent 60%),
    radial-gradient(760px 430px at 86% 16%, rgba(179,135,255,0.14), transparent 55%),
    radial-gradient(680px 360px at 50% 95%, rgba(98,230,255,0.07), transparent 55%),
    linear-gradient(135deg, var(--bg0) 0%, var(--bg1) 50%, var(--bg0) 100%);
}

.hdr{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  font-size: 2.05rem;
  letter-spacing: 0.2px;
  margin: 0.15rem 0 0.1rem 0;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

.sub{ color: var(--muted); margin-bottom: 0.6rem; }

.row{
  display:flex; justify-content:space-between; align-items:center; gap:10px;
  flex-wrap:wrap;
}

.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  font-size: 0.86rem;
  color: rgba(255,255,255,0.86);
  white-space:nowrap;
}

.tag-good{ border-color: rgba(78,240,163,0.38); background: rgba(78,240,163,0.11); }
.tag-warn{ border-color: rgba(255,209,102,0.44); background: rgba(255,209,102,0.11); }
.tag-bad { border-color: rgba(255,92,138,0.44); background: rgba(255,92,138,0.12); }

.panel{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.42);
  margin: 10px 0;
}

.card{
  background: var(--card2);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  margin: 10px 0;
  box-shadow: 0 14px 45px rgba(0,0,0,0.40);
}

.grid{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}
@media (max-width: 900px){
  .grid{ grid-template-columns: 1fr; }
}

.metricbox{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 10px 12px;
}
.mtitle{ color: var(--muted); font-size: 0.82rem; margin-bottom: 4px;}
.mval{ font-weight: 800; font-size: 1.08rem; }

.small{ color: var(--muted); font-size: 0.9rem; }
hr{ border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }

.signal{
  display:inline-flex; align-items:center; gap:8px;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.035);
  font-size: 0.82rem;
  color: rgba(255,255,255,0.84);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr">🛰️ TITAN – Mission Control</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Autonóm ajánló: <b>mindig 2 legjobb pick</b> • xG→Poisson + őszinte rizikó • '
    'Hír/narratíva csak <b>RIZIKÓ jelzés</b> (penalty), nem “tuti”.</div>',
    unsafe_allow_html=True,
)

# =========================================================
#  Utils
# =========================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def season_from_today() -> int:
    # Understat season = szezon kezdő éve
    today = datetime.now().date()
    return today.year - 1 if today.month < 7 else today.year

def parse_dt(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def fmt_local(dt):
    if not dt:
        return "—"
    try:
        return dt.astimezone().strftime("%Y.%m.%d %H:%M")
    except Exception:
        return dt.strftime("%Y.%m.%d %H:%M")

def clamp(x, a, b):
    return max(a, min(b, x))

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def clean_team(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name

def stable_match_id(league_key: str, kickoff_dt: datetime, home: str, away: str) -> str:
    k = kickoff_dt.strftime("%Y%m%d%H%M") if kickoff_dt else "0000"
    return f"{league_key}:{k}:{home.lower()}__{away.lower()}"

# =========================================================
#  Poisson model
# =========================================================
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

# =========================================================
#  Social (kulcs nélküli): Google News RSS + GDELT
# =========================================================
NEG_KEYWORDS = [
    "injury", "injured", "ruled out", "out", "doubtful", "sidelined",
    "suspended", "suspension", "ban",
    "scandal", "arrest", "police", "court",
    "divorce", "wife", "girlfriend", "partner", "family",
]

def count_neg_hits(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for k in NEG_KEYWORDS if k in t)

def google_news_rss(query: str, hl="en", gl="US", ceid="US:en", limit=12):
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
    feed = feedparser.parse(url)
    out = []
    for e in (feed.entries or [])[:limit]:
        out.append({
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "published": e.get("published", ""),
            "source": (e.get("source") or {}).get("title", ""),
        })
    return out

def gdelt_doc(query: str, maxrecords=12):
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
        out.append({
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "domain": a.get("domain", ""),
            "seendate": a.get("seendate", ""),
            "tone": a.get("tone", None),
        })
    return out

def build_social_query_pack(home: str, away: str):
    # konzervatív, kevés zaj
    neg_terms = ["injury", "suspended", "scandal", "divorce", "arrest"]
    gnews_q = f'({home} OR "{away}") AND ({ " OR ".join(neg_terms) })'
    gdelt_q = f'({home} OR "{away}") ({ " OR ".join(neg_terms) })'
    return gnews_q, gdelt_q

def social_penalty(neg_hits: int) -> float:
    # kifejezetten konzervatív: csak levonás, plafon
    if neg_hits <= 0: return 0.00
    if neg_hits == 1: return 0.05
    if neg_hits == 2: return 0.08
    if 3 <= neg_hits <= 4: return 0.12
    return 0.15

# =========================================================
#  Optional Weather (OpenWeather) – csak extrém esetekben
# =========================================================
TEAM_CITY = {
    # bővíthető később (most: ha nincs város -> skip)
    "Manchester United": "Manchester,GB",
    "Manchester City": "Manchester,GB",
    "Liverpool": "Liverpool,GB",
    "Arsenal": "London,GB",
    "Chelsea": "London,GB",
    "Tottenham": "London,GB",
    "Real Madrid": "Madrid,ES",
    "Barcelona": "Barcelona,ES",
    "Bayern Munich": "Munich,DE",
    "Borussia Dortmund": "Dortmund,DE",
    "Juventus": "Turin,IT",
    "AC Milan": "Milan,IT",
    "Inter": "Milan,IT",
    "PSG": "Paris,FR",
}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather_forecast(city_q: str):
    if not WEATHER_API_KEY:
        return None
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city_q, "appid": WEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    return r.json()

def weather_penalty_for_match(home: str, kickoff_dt: datetime) -> float:
    if not USE_WEATHER or not WEATHER_API_KEY:
        return 0.0
    city = TEAM_CITY.get(home)
    if not city:
        return 0.0
    try:
        data = fetch_weather_forecast(city)
        if not data:
            return 0.0
        # Közelítsük a kickoffhoz legközelebbi 3 órás blokkot
        target = kickoff_dt
        best = None
        best_diff = None
        for item in data.get("list", [])[:80]:
            dt = datetime.fromtimestamp(int(item.get("dt", 0)), tz=timezone.utc)
            diff = abs((dt - target).total_seconds())
            if best is None or diff < best_diff:
                best = item
                best_diff = diff

        if not best:
            return 0.0

        wind = safe_float((best.get("wind") or {}).get("speed"), 0.0) or 0.0
        rain = safe_float((best.get("rain") or {}).get("3h"), 0.0) or 0.0
        snow = safe_float((best.get("snow") or {}).get("3h"), 0.0) or 0.0

        pen = 0.0
        # egyszerű, “extrém trigger” logika
        if wind >= 11: pen += 0.04
        if rain >= 3: pen += 0.03
        if snow >= 1: pen += 0.05
        return clamp(pen, 0.0, WEATHER_MAX_PENALTY)
    except Exception:
        return 0.0

# =========================================================
#  Optional Odds (The Odds API) – csak UI/value jelzésre
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_odds_snapshot():
    if not USE_ODDS or not ODDS_API_KEY:
        return None
    # Fő öt ligára külön sport kulcs kellene – ezt később finomítjuk.
    # Itt csak “keret” van, hogy a motor stabil maradjon: ha nincs adat -> skip.
    return None

# =========================================================
#  Understat async runner (Streamlit-safe)
# =========================================================
def run_async(coro):
    """Run coroutine safely in Streamlit."""
    try:
        loop = asyncio.get_running_loop()
        # ha fut loop, futtassunk új loopban
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

def build_team_xg_profiles(results: list[dict]):
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

    h_for = ph.get("home_xg_for")
    h_against = ph.get("home_xg_against")
    a_for = pa.get("away_xg_for")
    a_against = pa.get("away_xg_against")

    lh_parts = []
    if h_for is not None: lh_parts.append(h_for)
    if a_against is not None: lh_parts.append(a_against)
    lh = sum(lh_parts)/len(lh_parts) if lh_parts else base

    la_parts = []
    if a_for is not None: la_parts.append(a_for)
    if h_against is not None: la_parts.append(h_against)
    la = sum(la_parts)/len(la_parts) if la_parts else base

    lh = clamp(lh, 0.2, 3.5)
    la = clamp(la, 0.2, 3.5)

    n_home = int(ph.get("n_home", 0) or 0)
    n_away = int(pa.get("n_away", 0) or 0)
    return lh, la, n_home, n_away

def label_risk(n_home: int, n_away: int, extra_penalty: float):
    # adatminőség + jelzőpenalty együtt
    if n_home >= 8 and n_away >= 8 and extra_penalty < 0.08:
        return "MEGBÍZHATÓ", "tag-good"
    if n_home >= 4 and n_away >= 4 and extra_penalty < 0.12:
        return "RIZIKÓS", "tag-warn"
    return "NAGYON RIZIKÓS", "tag-bad"

def pick_recommendation(lh, la, p1, px, p2, pbtts, pover25):
    total_xg = lh + la

    # CORE heurisztika – stabil, őszinte
    if pbtts >= 0.58 and total_xg >= 2.55:
        return ("BTTS – IGEN", pbtts, f"Mindkét csapat gól valószínű (össz xG ~ {total_xg:.2f}).")
    if pover25 >= 0.56 and total_xg >= 2.60:
        return ("Over 2.5 gól", pover25, f"Magas gólvárakozás (össz xG ~ {total_xg:.2f}).")

    mx = max(p1, px, p2)
    if mx == p1:
        return ("Hazai győzelem (1)", p1, f"Hazai oldal valószínűbb (~{p1*100:.0f}%).")
    if mx == p2:
        return ("Vendég győzelem (2)", p2, f"Vendég oldal valószínűbb (~{p2*100:.0f}%).")
    return ("Döntetlen (X)", px, f"Döntetlen valószínűbb (~{px*100:.0f}%).")

def compute_reliability(conf: float) -> int:
    # 0–100 megbízhatósági skála
    return int(clamp(conf * 100, 0, 100))

# =========================================================
#  Backtest log + ellenőrzés (Understat results alapján)
# =========================================================
LOG_FIELDS = [
    "logged_utc",
    "league_key",
    "league",
    "season",
    "match_id",
    "kickoff_utc",
    "kickoff_local",
    "home",
    "away",
    "pick",
    "confidence",
    "risk_label",
    "social_neg_hits",
    "social_penalty",
    "weather_penalty",
    "result_home",
    "result_away",
    "outcome",  # WIN/LOSS/PUSH/UNKNOWN
]

def ensure_log_file():
    if not PICKS_LOG_PATH.exists():
        with PICKS_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()

def read_log_df() -> pd.DataFrame:
    if not PICKS_LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_FIELDS)
    return pd.read_csv(PICKS_LOG_PATH)

def append_log_rows(rows: list[dict]):
    ensure_log_file()
    # dedupe: match_id + pick
    existing = set()
    df = read_log_df()
    if not df.empty:
        for _, r in df.iterrows():
            existing.add(f"{r.get('match_id','')}|{r.get('pick','')}")
    with PICKS_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        for row in rows:
            key = f"{row.get('match_id','')}|{row.get('pick','')}"
            if key in existing:
                continue
            w.writerow(row)

def eval_pick_outcome(pick: str, gh: int, ga: int) -> str:
    pick = (pick or "").strip().lower()
    if gh is None or ga is None:
        return "UNKNOWN"
    total = gh + ga
    if "btts" in pick:
        return "WIN" if (gh >= 1 and ga >= 1) else "LOSS"
    if "over 2.5" in pick:
        return "WIN" if total >= 3 else "LOSS"
    if "under 2.5" in pick:
        return "WIN" if total <= 2 else "LOSS"
    if "(1)" in pick or "hazai" in pick:
        return "WIN" if gh > ga else ("PUSH" if gh == ga else "LOSS")
    if "(2)" in pick or "vendég" in pick or "vendeg" in pick:
        return "WIN" if ga > gh else ("PUSH" if gh == ga else "LOSS")
    if "(x)" in pick or "döntetlen" in pick or "dontetlen" in pick:
        return "WIN" if gh == ga else "LOSS"
    return "UNKNOWN"

@st.cache_data(ttl=600, show_spinner=False)
def understat_results_for_league(league_key: str, season: int):
    async def _run():
        async with aiohttp.ClientSession() as session:
            u = Understat(session)
            results = await u.get_league_results(league_key, season)
            return results or []
    return run_async(_run())

def find_result_for_match(results: list[dict], home: str, away: str, kickoff_utc: datetime):
    # Understat results-ben vannak datetime + h/a + goals
    # tolerant: +/- 36 óra, mert timezone/fixture változhat
    best = None
    best_diff = None
    for m in results or []:
        h = clean_team(((m.get("h") or {}).get("title")))
        a = clean_team(((m.get("a") or {}).get("title")))
        if h.lower() != home.lower() or a.lower() != away.lower():
            continue
        dt = parse_dt(m.get("datetime", ""))
        if not dt:
            continue
        diff = abs((dt - kickoff_utc).total_seconds())
        if diff <= 36 * 3600:
            if best is None or diff < best_diff:
                best = m
                best_diff = diff
    if not best:
        return None
    gh = safe_float(((best.get("goals") or {}).get("h")), None)
    ga = safe_float(((best.get("goals") or {}).get("a")), None)
    if gh is None or ga is None:
        return None
    return int(gh), int(ga)

def verify_log_outcomes():
    df = read_log_df()
    if df.empty:
        return df, 0

    # only unresolved
    unresolved = df[(df["outcome"].isna()) | (df["outcome"].astype(str) == "") | (df["outcome"].astype(str) == "UNKNOWN")]
    if unresolved.empty:
        return df, 0

    updated = 0
    cache_results = {}

    for idx, row in unresolved.iterrows():
        league_key = str(row.get("league_key", ""))
        season = int(row.get("season", season_from_today()))
        home = str(row.get("home", ""))
        away = str(row.get("away", ""))
        kick_str = str(row.get("kickoff_utc", ""))
        try:
            kickoff_utc = datetime.fromisoformat(kick_str)
            if kickoff_utc.tzinfo is None:
                kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if league_key not in cache_results:
            cache_results[league_key] = understat_results_for_league(league_key, season)

        res = find_result_for_match(cache_results[league_key], home, away, kickoff_utc)
        if not res:
            continue

        gh, ga = res
        outcome = eval_pick_outcome(str(row.get("pick", "")), gh, ga)
        df.at[idx, "result_home"] = gh
        df.at[idx, "result_away"] = ga
        df.at[idx, "outcome"] = outcome
        updated += 1

    if updated > 0:
        df.to_csv(PICKS_LOG_PATH, index=False)
    return df, updated

# =========================================================
#  Build match analysis (CORE + penalty)
# =========================================================
def build_match_analysis(league_key: str, league_name: str, season: int, home: str, away: str, kickoff_dt: datetime, prof: dict):
    lh, la, n_home, n_away = expected_goals_from_profiles(home, away, prof)
    pbtts = prob_btts(lh, la)
    pover25 = prob_over_25(lh, la)
    p1, px, p2 = prob_1x2(lh, la)

    pick, pval, why = pick_recommendation(lh, la, p1, px, p2, pbtts, pover25)

    # --- Social (kulcs nélküli) ---
    social = {"gnews": [], "gdelt": [], "neg_hits": 0, "risk_penalty": 0.0}
    if USE_GOOGLE_NEWS_RSS or USE_GDELT:
        gnews_q, gdelt_q = build_social_query_pack(home, away)
        try:
            if USE_GOOGLE_NEWS_RSS:
                gnews = google_news_rss(gnews_q, limit=SOCIAL_MAX_ITEMS)
                social["gnews"] = gnews
                social["neg_hits"] += sum(count_neg_hits(x.get("title", "")) for x in gnews)
            if USE_GDELT:
                garts = gdelt_doc(gdelt_q, maxrecords=SOCIAL_MAX_ITEMS)
                social["gdelt"] = garts
                for a in garts:
                    social["neg_hits"] += count_neg_hits(a.get("title", ""))
                    tone = a.get("tone")
                    if isinstance(tone, (int, float)) and tone < -4:
                        social["neg_hits"] += 1
        except Exception:
            pass

    spen = social_penalty(int(social["neg_hits"] or 0))
    social["risk_penalty"] = spen

    # --- Weather penalty (opcionális) ---
    wpen = weather_penalty_for_match(home, kickoff_dt) if USE_WEATHER else 0.0

    # --- Confidence adjust (csak levonás) ---
    conf_adj = clamp(pval - spen - wpen, 0.0, 1.0)

    # --- Risk label (őszinte) ---
    risk_label, risk_class = label_risk(n_home, n_away, extra_penalty=(spen + wpen))

    # --- Summary (rövid, “miért”) ---
    summary_lines = [
        f"**xG várható gól:** {home} `{lh:.2f}` • {away} `{la:.2f}` • össz `{(lh+la):.2f}`",
        f"**Piac esélyek:** BTTS `{pbtts*100:.0f}%` • Over2.5 `{pover25*100:.0f}%` • 1/X/2 `{p1*100:.0f}/{px*100:.0f}/{p2*100:.0f}%`",
        f"**Ajánlás:** **{pick}** • alapesély `{pval*100:.0f}%` → korrigált `{conf_adj*100:.0f}%`",
        f"**Rizikó:** **{risk_label}** • adat: (H `{n_home}` / A `{n_away}`)",
    ]
    if spen > 0:
        summary_lines.append(f"🧠 **Narratíva penalty:** −{spen*100:.0f}% (neg találat: {social['neg_hits']})")
    if wpen > 0:
        summary_lines.append(f"🌦️ **Időjárás penalty:** −{wpen*100:.0f}% (extrém trigger)")

    # signals
    signals = []
    if spen > 0: signals.append("🧠 Social")
    if wpen > 0: signals.append("🌦️ Weather")
    if ODDS_API_KEY and USE_ODDS: signals.append("💸 Odds")
    signals.append("🧱 Data")

    return {
        "league_key": league_key,
        "league": league_name,
        "season": season,
        "home": home,
        "away": away,
        "kickoff": kickoff_dt,
        "kickoff_str": fmt_local(kickoff_dt),
        "match_id": stable_match_id(league_key, kickoff_dt, home, away),
        "lh": lh,
        "la": la,
        "pbtts": pbtts,
        "pover25": pover25,
        "p1": p1, "px": px, "p2": p2,
        "pick": pick,
        "confidence_raw": pval,
        "confidence": conf_adj,
        "reliability": compute_reliability(conf_adj),
        "risk_label": risk_label,
        "risk_class": risk_class,
        "social": social,
        "social_penalty": spen,
        "weather_penalty": wpen,
        "signals": signals,
        "why": why,
        "summary": "\n".join(summary_lines),
        "quality_home": n_home,
        "quality_away": n_away,
    }

# =========================================================
#  MAIN – Autonóm futás (nincs gomb, nincs beállítás)
# =========================================================
season = season_from_today()
start = now_utc()
updated_df, updated_n = verify_log_outcomes()

status_pills = []
status_pills.append("Understat ✅")
status_pills.append("RSS ✅" if USE_GOOGLE_NEWS_RSS else "RSS —")
status_pills.append("GDELT ✅" if USE_GDELT else "GDELT —")
status_pills.append("Odds ✅" if (ODDS_API_KEY and USE_ODDS) else "Odds —")
status_pills.append("Weather ✅" if (WEATHER_API_KEY and USE_WEATHER) else "Weather —")

st.markdown(
    f"""
<div class="row">
  <div class="pill">⏱️ Frissítve: <b>{datetime.now().astimezone().strftime('%Y.%m.%d %H:%M')}</b></div>
  <div class="row">
    {"".join([f'<div class="pill">{p}</div>' for p in status_pills])}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if updated_n > 0:
    st.success(f"Utóellenőrzés: {updated_n} korábbi pick frissítve eredménnyel.")

# Fetch + analyze
rows = []
errors = []

with st.spinner("Autonóm elemzés: Understat xG + narratíva jelzés + rizikó…"):
    for lk, league_name in LEAGUES.items():
        try:
            fixtures, results = understat_fetch(lk, season, DEFAULT_DAYS_AHEAD)
        except Exception as e:
            errors.append(f"{league_name}: {e}")
            continue

        prof = build_team_xg_profiles(results)

        if not fixtures:
            continue

        for m in fixtures:
            home = clean_team(((m.get("h") or {}).get("title")))
            away = clean_team(((m.get("a") or {}).get("title")))
            kickoff = parse_dt(m.get("datetime", ""))

            if not home or not away or not kickoff:
                continue

            row = build_match_analysis(lk, league_name, season, home, away, kickoff, prof)
            rows.append(row)

if errors:
    st.warning("Néhány liga hibával tért vissza:\n\n" + "\n".join([f"• {x}" for x in errors]))

if not rows:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("Nincs meccs a közeljövőben a fix időablakban. (Autonóm mód: nincs üres oldal.)")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

df = pd.DataFrame(rows)
df = df.sort_values(by=["confidence", "kickoff"], ascending=[False, True]).reset_index(drop=True)

# Top 2 picks (autonóm)
top2 = df.head(TOP_K).copy()

# Auto-log top picks
if AUTO_LOG_TOP_PICKS and not top2.empty:
    log_rows = []
    for _, r in top2.iterrows():
        log_rows.append({
            "logged_utc": now_utc().isoformat(),
            "league_key": r["league_key"],
            "league": r["league"],
            "season": int(r["season"]),
            "match_id": r["match_id"],
            "kickoff_utc": r["kickoff"].isoformat(),
            "kickoff_local": r["kickoff_str"],
            "home": r["home"],
            "away": r["away"],
            "pick": r["pick"],
            "confidence": float(r["confidence"]),
            "risk_label": r["risk_label"],
            "social_neg_hits": int((r["social"] or {}).get("neg_hits", 0)),
            "social_penalty": float(r.get("social_penalty", 0.0)),
            "weather_penalty": float(r.get("weather_penalty", 0.0)),
            "result_home": "",
            "result_away": "",
            "outcome": "UNKNOWN",
        })
    append_log_rows(log_rows)

# =========================================================
#  Dashboard stats + Top 2 cards
# =========================================================
st.markdown("<hr/>", unsafe_allow_html=True)

# Quick stats
logdf = read_log_df()
resolved = logdf[logdf["outcome"].isin(["WIN", "LOSS", "PUSH"])] if not logdf.empty and "outcome" in logdf.columns else pd.DataFrame()
wins = int((resolved["outcome"] == "WIN").sum()) if not resolved.empty else 0
loss = int((resolved["outcome"] == "LOSS").sum()) if not resolved.empty else 0
push = int((resolved["outcome"] == "PUSH").sum()) if not resolved.empty else 0
total_res = wins + loss + push
winrate = (wins / total_res * 100) if total_res > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Közelgő meccsek", int(len(df)))
with k2:
    st.metric("TOP pickek", TOP_K)
with k3:
    st.metric("Backtest (lezárt)", total_res)
with k4:
    st.metric("Winrate", f"{winrate:.1f}%")

# TOP 2 panel
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🎯 A két legjobb választás (autonóm)")

for i, r in top2.iterrows():
    rel = int(r["reliability"])
    risk_tag = r["risk_label"]
    rclass = r["risk_class"]
    signals = " ".join([f"<span class='signal'>{s}</span>" for s in r["signals"]])

    # social headlines (max 3)
    headlines = []
    if SHOW_SOCIAL_DETAILS_IN_CARD:
        gnews = (r["social"] or {}).get("gnews", []) if isinstance(r.get("social"), dict) else []
        gd = (r["social"] or {}).get("gdelt", []) if isinstance(r.get("social"), dict) else []
        for x in (gnews or [])[:2]:
            t = x.get("title", "").strip()
            if t:
                headlines.append(f"• {t}")
        for x in (gd or [])[:1]:
            t = x.get("title", "").strip()
            if t:
                headlines.append(f"• {t}")

    extra = ""
    if headlines and (r.get("social_penalty", 0.0) > 0):
        extra = "<div class='small' style='margin-top:8px; white-space:pre-wrap;'>" + "\n".join(headlines) + "</div>"

    st.markdown(
        f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">
    <div class="pill"><b>#{i+1}</b> • <span class="small">{r['league']}</span> • Kezdés: <b>{r['kickoff_str']}</b></div>
    <div class="pill {rclass}"><b>{risk_tag}</b> • Megbízhatóság: <b>{rel}%</b></div>
  </div>

  <h3 style="margin:0.45rem 0 0.35rem 0;">{r['home']} vs {r['away']}</h3>

  <div class="grid">
    <div class="metricbox"><div class="mtitle">Ajánlás</div><div class="mval">{r['pick']}</div></div>
    <div class="metricbox"><div class="mtitle">Valószínűség</div><div class="mval">{r['confidence']*100:.0f}%</div></div>
    <div class="metricbox"><div class="mtitle">Össz xG</div><div class="mval">{(r['lh']+r['la']):.2f}</div></div>
  </div>

  <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
    {signals}
  </div>

  <details style="margin-top:10px;">
    <summary style="cursor:pointer;color:rgba(255,255,255,0.84);">Miért ezt?</summary>
    <div style="margin-top:8px; white-space:pre-wrap;">{r['summary']}</div>
    {extra}
  </details>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
#  Backtest panel (statisztika + utóellenőrzés)
# =========================================================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🧾 Backtest / Utóellenőrzés (mentett tippek)")

colA, colB = st.columns([1, 1])
with colA:
    st.write(f"Mentett tippek: **{len(logdf)}**")
    st.write(f"Lezárt: **{total_res}** • WIN: **{wins}** • LOSS: **{loss}** • PUSH: **{push}**")
with colB:
    if st.button("🔁 Eredmények újraellenőrzése most", use_container_width=True):
        df2, n2 = verify_log_outcomes()
        st.success(f"Frissítve: {n2} pick.")
        logdf = df2

if not logdf.empty:
    # utolsó 12 sor, hogy ne legyen hosszú
    view = logdf.copy()
    view = view.sort_values(by=["logged_utc"], ascending=False).head(12)
    st.dataframe(view, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Megjegyzés: a hír/narratíva jelzés (RSS+GDELT) csak rizikó-penalty. "
    "A motor nem “social alapján tippel”, hanem xG-alapú döntést korrigál őszintén."
)
