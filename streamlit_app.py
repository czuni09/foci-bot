# app.py — TITAN v2 (Masszív motor + moduláris modifiers)
# Python 3.12 | Streamlit
#
# KULCSOK (ENV vagy .streamlit/secrets.toml):
#   UNDERSTAT: nincs kulcs
#   NEWS_API_KEY         (NewsAPI.org)
#   WEATHER_API_KEY      (OpenWeather)
#   ODDS_API_KEY         (The Odds API)
#   FOOTYSTATS_API_KEY   (opcionális; ha van hivatalos hozzáférésed)
#
# Futás:
#   streamlit run app.py

import os
import re
import math
import json
import time
import asyncio
import threading
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import aiohttp
import pandas as pd
import streamlit as st
from understat import Understat


# =========================
#  KONFIG
# =========================
st.set_page_config(page_title="TITAN v2 – Match Intelligence", page_icon="🛰️", layout="wide")

NEWS_API_KEY = (os.getenv("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY", "")).strip()
WEATHER_API_KEY = (os.getenv("WEATHER_API_KEY") or st.secrets.get("WEATHER_API_KEY", "")).strip()
ODDS_API_KEY = (os.getenv("ODDS_API_KEY") or st.secrets.get("ODDS_API_KEY", "")).strip()
FOOTYSTATS_API_KEY = (os.getenv("FOOTYSTATS_API_KEY") or st.secrets.get("FOOTYSTATS_API_KEY", "")).strip()

OPENFOOTBALL_JSON_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.1.json"

LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}

DEFAULT_SEASONS = {
    "epl": 2024,
    "la_liga": 2024,
    "bundesliga": 2024,
    "serie_a": 2024,
    "ligue_1": 2024,
}

# Minimális meccsszám az "adatminőséghez"
MIN_MATCHES_OK = 8
MIN_MATCHES_WARN = 4

# Időjárás trigger küszöbök
WIND_STRONG_MS = 11.0      # 10–12 m/s környéke
RAIN_HEAVY_MM3H = 4.0      # OpenWeather 3h rain, durva küszöb
TEMP_EXTREME_C = 32.0      # meleg
TEMP_COLD_C = -2.0         # hideg

# Hír időfaktor
NEWS_STRONG_HOURS = 24
NEWS_MED_HOURS = 72
NEWS_LOOKBACK_DAYS = 7

# Módosító maximumok (biztonsági korlát)
MAX_TOTAL_PENALTY = 0.35   # -35% max
MAX_TOTAL_BOOST = 0.20     # +20% max


# =========================
#  UI – “mission control”
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

:root{
  --bg0:#06070c;
  --bg1:#0b1020;
  --bg2:#0e1630;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --good:#4ef0a3;
  --warn:#ffd166;
  --bad:#ff5c8a;
  --accent:#79a6ff;
  --accent2:#b387ff;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp{
  background:
    radial-gradient(900px 500px at 15% 10%, rgba(121,166,255,0.18), transparent 60%),
    radial-gradient(700px 400px at 85% 15%, rgba(179,135,255,0.14), transparent 55%),
    linear-gradient(135deg, var(--bg0) 0%, var(--bg1) 50%, var(--bg0) 100%);
}

.hdr{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  font-size: 2.1rem;
  letter-spacing: 0.2px;
  margin: 0.2rem 0 0.1rem 0;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

.sub{ color: var(--muted); margin-bottom: 1rem; }

.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  font-size: 0.86rem;
  color: rgba(255,255,255,0.86);
}

.panel{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.42);
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

.metricbox{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 10px 12px;
}
.mtitle{ color: var(--muted); font-size: 0.82rem; margin-bottom: 4px;}
.mval{ font-weight: 800; font-size: 1.05rem; }

.tag-good{ border-color: rgba(78,240,163,0.35); background: rgba(78,240,163,0.10); }
.tag-warn{ border-color: rgba(255,209,102,0.40); background: rgba(255,209,102,0.10); }
.tag-bad { border-color: rgba(255,92,138,0.40); background: rgba(255,92,138,0.12); }

.small{ color: var(--muted); font-size: 0.9rem; }
hr{ border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr">🛰️ TITAN v2 – Match Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Masszív xG CORE + “modifier” réteg (Hírek / Időjárás / Odds / Sérültek). '
    'Mindig ad javaslatot – ha gyenge az adat vagy negatív jel van, <b>RIZIKÓS / NEM AJÁNLOTT</b> címkével jelzi.</div>',
    unsafe_allow_html=True,
)


# =========================
#  Masszív async futtató (Streamlit kompatibilis)
# =========================
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_coro_safely(coro):
    """
    Streamlitben néha fut event loop. Ez a wrapper mindig stabilan lefuttatja:
    - ha nincs futó loop: asyncio.run
    - ha van futó loop: külön thread + új loop
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            def _in_thread():
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            fut = _EXECUTOR.submit(_in_thread)
            return fut.result(timeout=40)
    except RuntimeError:
        pass
    return asyncio.run(coro)


# =========================
#  Segédek
# =========================
def now_utc():
    return datetime.now(timezone.utc)

def parse_dt(s: str):
    # Understat datetime: "YYYY-MM-DD HH:MM:SS" (feltételezzük UTC)
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


# =========================
#  Poisson modell
# =========================
def poisson_pmf(lmb, k):
    return math.exp(-lmb) * (lmb ** k) / math.factorial(k)

def prob_over_25(lh, la, max_goals=10):
    p = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j >= 3:
                p += poisson_pmf(lh, i) * poisson_pmf(la, j)
    return clamp(p, 0.0, 1.0)

def prob_under_25(lh, la, max_goals=10):
    # P(total <= 2)
    p = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j <= 2:
                p += poisson_pmf(lh, i) * poisson_pmf(la, j)
    return clamp(p, 0.0, 1.0)

def prob_btts(lh, la):
    p = 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh + la))
    return clamp(p, 0.0, 1.0)

def prob_1x2(lh, la, max_goals=10):
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
#  Understat — async + cache
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def understat_fetch(league_key: str, season: int) -> Tuple[List[dict], List[dict], List[dict]]:
    async def _run():
        async with aiohttp.ClientSession() as session:
            u = Understat(session)
            fixtures = await u.get_league_fixtures(league_key, season)
            results = await u.get_league_results(league_key, season)
            teams = await u.get_league_teams(league_key, season)
            return fixtures or [], results or [], teams or []
    return run_coro_safely(_run())


def build_team_xg_profiles(results: List[dict]) -> Dict[str, Dict[str, Any]]:
    prof: Dict[str, Dict[str, List[float]]] = {}

    def ensure(team):
        prof.setdefault(team, {
            "home_for": [], "home_against": [],
            "away_for": [], "away_against": [],
        })

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

    out: Dict[str, Dict[str, Any]] = {}
    for team, d in prof.items():
        hf, ha = d["home_for"], d["home_against"]
        af, aa = d["away_for"], d["away_against"]
        out[team] = {
            "home_xg_for": sum(hf)/len(hf) if hf else None,
            "home_xg_against": sum(ha)/len(ha) if ha else None,
            "away_xg_for": sum(af)/len(af) if af else None,
            "away_xg_against": sum(aa)/len(aa) if aa else None,
            "n_home": len(hf),
            "n_away": len(af),
        }
    return out


def league_base_xg(results: List[dict]) -> float:
    # masszív fallback: liga átlag xG / csapat
    vals = []
    for m in results or []:
        xgh = safe_float(((m.get("xG") or {}).get("h")))
        xga = safe_float(((m.get("xG") or {}).get("a")))
        if xgh is None or xga is None:
            continue
        vals.append(xgh)
        vals.append(xga)
    if not vals:
        return 1.35
    return clamp(sum(vals)/len(vals), 0.9, 1.7)


def expected_goals_from_profiles(home: str, away: str, prof: dict, base: float) -> Tuple[float, float, int, int]:
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


def label_risk(n_home: int, n_away: int, extra_penalty: float) -> Tuple[str, str]:
    # adat + extra kockázat (hírek/időjárás stb.) együtt
    base_label = "NEM AJÁNLOTT"
    base_class = "tag-bad"

    if n_home >= MIN_MATCHES_OK and n_away >= MIN_MATCHES_OK:
        base_label, base_class = "AJÁNLOTT", "tag-good"
    elif n_home >= MIN_MATCHES_WARN and n_away >= MIN_MATCHES_WARN:
        base_label, base_class = "RIZIKÓS", "tag-warn"

    # extra_penalty rántsa le (őszinte)
    if extra_penalty >= 0.22:
        return "NEM AJÁNLOTT", "tag-bad"
    if extra_penalty >= 0.12 and base_label == "AJÁNLOTT":
        return "RIZIKÓS", "tag-warn"
    return base_label, base_class


# =========================
#  Modifiers — HÍREK (NewsAPI)
# =========================
NEWS_KEYWORDS = [
    "injury", "injured", "out", "ruled out", "doubtful", "knock",
    "suspension", "suspended", "ban",
    "scandal", "arrest", "court", "police",
    "divorce", "girlfriend", "wife", "family", "illness",
    "training ground incident", "fight",
]

def news_query(team_home: str, team_away: str) -> str:
    # szándékosan egyszerű, masszív
    t1 = team_home.replace('"', "")
    t2 = team_away.replace('"', "")
    kw = " OR ".join([f'"{k}"' for k in NEWS_KEYWORDS[:10]])
    return f'("{t1}" OR "{t2}") AND ({kw})'

@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(team_home: str, team_away: str) -> List[dict]:
    if not NEWS_API_KEY:
        return []
    q = news_query(team_home, team_away)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "from": (now_utc() - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
        "apiKey": NEWS_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json() or {}
        return data.get("articles") or []
    except Exception:
        return []

def score_news_penalty(articles: List[dict]) -> Tuple[float, List[str]]:
    """
    Vissza:
      penalty: 0..0.30
      reasons: max 3 rövid indok
    """
    if not articles:
        return 0.0, []

    now = now_utc()
    score = 0.0
    reasons = []

    for a in articles[:8]:
        dt_s = a.get("publishedAt") or ""
        try:
            dt = datetime.fromisoformat(dt_s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            dt = None

        age_h = 999.0
        if dt:
            age_h = (now - dt).total_seconds() / 3600.0

        title = (a.get("title") or "").strip()
        descr = (a.get("description") or "").strip()
        blob = f"{title} {descr}".lower()

        # időfaktor
        if age_h <= NEWS_STRONG_HOURS:
            time_w = 1.0
        elif age_h <= NEWS_MED_HOURS:
            time_w = 0.6
        elif age_h <= NEWS_LOOKBACK_DAYS * 24:
            time_w = 0.3
        else:
            time_w = 0.0

        if time_w == 0.0:
            continue

        # tartalmi jel
        hit = 0.0
        for kw in ["injury", "suspended", "arrest", "court", "illness", "divorce", "police", "scandal", "fight"]:
            if kw in blob:
                hit += 1.0
        if hit == 0:
            continue

        score += min(1.0, hit / 3.0) * time_w

        if len(reasons) < 3 and title:
            reasons.append(f"Hírjel: {title[:90]}")

    # score -> penalty (saturating)
    penalty = clamp(0.08 * score, 0.0, 0.30)
    return penalty, reasons


# =========================
#  Modifiers — IDŐJÁRÁS (OpenWeather forecast)
# =========================
TEAM_CITY_MAP = {
    # gyors példa — bővítsd. Ha nincs, modul 0-ával visszatér.
    # EPL:
    "Manchester United": "Manchester,GB",
    "Manchester City": "Manchester,GB",
    "Arsenal": "London,GB",
    "Chelsea": "London,GB",
    "Liverpool": "Liverpool,GB",
    "Tottenham": "London,GB",
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(city_query: str) -> Optional[dict]:
    if not WEATHER_API_KEY or not city_query:
        return None
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city_query, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def pick_forecast_at_kickoff(forecast_json: dict, kickoff_utc: datetime) -> Optional[dict]:
    if not forecast_json:
        return None
    lst = forecast_json.get("list") or []
    if not lst:
        return None
    best = None
    best_dt = None
    for it in lst:
        dt = it.get("dt")
        if dt is None:
            continue
        dt_utc = datetime.fromtimestamp(int(dt), tz=timezone.utc)
        if best is None:
            best = it
            best_dt = dt_utc
            continue
        if abs((dt_utc - kickoff_utc).total_seconds()) < abs((best_dt - kickoff_utc).total_seconds()):
            best = it
            best_dt = dt_utc
    return best

def score_weather_penalty(home: str, kickoff_utc: datetime) -> Tuple[float, List[str]]:
    """
    Csak extrém esetek: wind, heavy rain/snow, extrém temp.
    """
    city = TEAM_CITY_MAP.get(home, "")
    if not city:
        return 0.0, []
    wjson = fetch_weather(city)
    it = pick_forecast_at_kickoff(wjson, kickoff_utc) if wjson else None
    if not it:
        return 0.0, []

    reasons = []
    wind = safe_float(((it.get("wind") or {}).get("speed")))
    temp = safe_float(((it.get("main") or {}).get("temp")))
    rain = safe_float(((it.get("rain") or {}).get("3h")))  # mm/3h
    snow = safe_float(((it.get("snow") or {}).get("3h")))

    penalty = 0.0

    if wind is not None and wind >= WIND_STRONG_MS:
        penalty += 0.08
        reasons.append(f"Időjárás: erős szél ~{wind:.0f} m/s → Over/BTTS gyengülhet")
    if rain is not None and rain >= RAIN_HEAVY_MM3H:
        penalty += 0.06
        reasons.append(f"Időjárás: heves eső (~{rain:.1f} mm/3h) → játékminőség romolhat")
    if snow is not None and snow > 0.1:
        penalty += 0.08
        reasons.append("Időjárás: hó/latyak → kaotikusabb meccs (RIZIKÓS)")
    if temp is not None and (temp >= TEMP_EXTREME_C or temp <= TEMP_COLD_C):
        penalty += 0.04
        reasons.append(f"Időjárás: szélsőséges hőmérséklet ({temp:.0f}°C) → volatilitás nőhet")

    return clamp(penalty, 0.0, 0.15), reasons[:3]


# =========================
#  Modifiers — ODDS (The Odds API) — opcionális, csak megjelenítés/value alap
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds_stub() -> Optional[dict]:
    # Itt hagyunk egy stabil "stub" helyet.
    # Ha később engeded, bekötjük konkrét meccsre (fixture matching + market mapping).
    return None


# =========================
#  Modifiers — FootyStats (lapok/szögletek) — stub
# =========================
def fetch_footystats_stub() -> Optional[dict]:
    # FootyStats integráció endpoint-függő. Masszív motor miatt most safe: nincs adat → 0 hatás.
    return None


# =========================
#  Ajánló logika (CORE pick)
# =========================
def pick_recommendation(lh, la, ph, pd, pa, pbtts, pover25, pund25) -> Tuple[str, float, str]:
    total_xg = lh + la

    # első: gólpiacok, ha erős jel
    if pbtts >= 0.58 and total_xg >= 2.55:
        return ("BTTS – IGEN", pbtts, f"Mindkét csapat gól esélyes (össz xG ~ {total_xg:.2f}).")
    if pover25 >= 0.56 and total_xg >= 2.60:
        return ("Over 2.5 gól", pover25, f"Magas gólvárakozás (össz xG ~ {total_xg:.2f}).")

    # konzervatív: ha alacsony össz xG, Under
    if total_xg <= 2.25 and pund25 >= 0.55:
        return ("Under 2.5 gól", pund25, f"Alacsonyabb gólvárakozás (össz xG ~ {total_xg:.2f}).")

    # 1X2: legnagyobb valószínűség
    mx = max(ph, pd, pa)
    if mx == ph:
        return ("Hazai győzelem (1)", ph, f"Hazai oldal valószínűbb (~{ph*100:.0f}%).")
    if mx == pa:
        return ("Vendég győzelem (2)", pa, f"Vendég oldal valószínűbb (~{pa*100:.0f}%).")
    return ("Döntetlen (X)", pd, f"Döntetlen kiugróbb (~{pd*100:.0f}%).")


def build_match_analysis(
    home: str,
    away: str,
    kickoff_dt: datetime,
    league_name: str,
    season: int,
    prof: dict,
    base_xg: float,
    use_news: bool,
    use_weather: bool
) -> Dict[str, Any]:
    # CORE
    lh, la, n_home, n_away = expected_goals_from_profiles(home, away, prof, base_xg)
    pbtts = prob_btts(lh, la)
    pover25 = prob_over_25(lh, la)
    pund25 = prob_under_25(lh, la)
    p1, px, p2 = prob_1x2(lh, la)

    pick, base_conf, why = pick_recommendation(lh, la, p1, px, p2, pbtts, pover25, pund25)

    # MODIFIERS
    penalty = 0.0
    boost = 0.0
    mod_reasons: List[str] = []

    if use_news:
        articles = fetch_news(home, away)
        p_news, reasons = score_news_penalty(articles)
        penalty += p_news
        mod_reasons.extend(reasons)

    if use_weather:
        p_w, reasons = score_weather_penalty(home, kickoff_dt)
        penalty += p_w
        mod_reasons.extend(reasons)

    penalty = clamp(penalty, 0.0, MAX_TOTAL_PENALTY)
    boost = clamp(boost, 0.0, MAX_TOTAL_BOOST)

    final_conf = clamp(base_conf + boost - penalty, 0.05, 0.95)

    risk_label, risk_class = label_risk(n_home, n_away, penalty)

    summary_lines = [
        f"**Várható gól (xG alap):** {home} ~ `{lh:.2f}`, {away} ~ `{la:.2f}` (össz: `{(lh+la):.2f}`)",
        f"**BTTS (IGEN):** `{pbtts*100:.0f}%` | **Over 2.5:** `{pover25*100:.0f}%` | **Under 2.5:** `{pund25*100:.0f}%`",
        f"**1X2:** 1=`{p1*100:.0f}%` • X=`{px*100:.0f}%` • 2=`{p2*100:.0f}%`",
        f"**CORE ajánlás:** **{pick}** (alap bizalom: `{base_conf*100:.0f}%`) — {why}",
    ]

    if mod_reasons:
        summary_lines.append(f"**Modifiers (őszinte hatás):** −`{penalty*100:.0f}%` a bizalomból")
        for r in mod_reasons[:4]:
            summary_lines.append(f"- {r}")

    summary_lines.append(f"**Végső bizalom:** `{final_conf*100:.0f}%` • **Címke:** **{risk_label}**")

    return {
        "league": league_name,
        "season": season,
        "home": home,
        "away": away,
        "kickoff": kickoff_dt,
        "kickoff_str": fmt_local(kickoff_dt),
        "lh": lh, "la": la,
        "pbtts": pbtts,
        "pover25": pover25,
        "pund25": pund25,
        "p1": p1, "px": px, "p2": p2,
        "pick": pick,
        "confidence": final_conf,
        "base_confidence": base_conf,
        "risk_label": risk_label,
        "risk_class": risk_class,
        "quality": (n_home, n_away),
        "penalty": penalty,
        "summary": "\n".join(summary_lines),
    }


# =========================
#  Sidebar – beállítások
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    sel_leagues = st.multiselect(
        "Ligák",
        options=list(LEAGUES.keys()),
        default=list(LEAGUES.keys()),
        format_func=lambda k: LEAGUES[k],
    )
    days_ahead = st.slider("Időablak (nap)", 1, 14, 4, 1)

    min_conf = st.slider("Minimum bizalom (szűrés)", 0.40, 0.80, 0.52, 0.01)
    show_all = st.toggle("Mutasson mindent (akkor is, ha nem ajánlott)", value=True)

    st.markdown("---")
    st.markdown("### 🧠 Modifiers")
    use_news = st.toggle("Hírek (NewsAPI) – risk/confidence módosító", value=bool(NEWS_API_KEY))
    use_weather = st.toggle("Időjárás (OpenWeather) – csak trigger esetén", value=bool(WEATHER_API_KEY))

    st.markdown("---")
    st.markdown("### ℹ️ Állapot")
    st.write("Understat: ✅")
    st.write(f"NEWS_API_KEY: {'✅' if NEWS_API_KEY else '—'}")
    st.write(f"WEATHER_API_KEY: {'✅' if WEATHER_API_KEY else '—'}")
    st.write(f"ODDS_API_KEY: {'✅' if ODDS_API_KEY else '— (opcionális)'}")
    st.write(f"FOOTYSTATS_API_KEY: {'✅' if FOOTYSTATS_API_KEY else '— (opcionális)'}")


# =========================
#  Futtatás
# =========================
run = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)

if not run:
    st.info("Nyomj egy **Elemzés indítása** gombot. A rendszer ligánként betölt, és minden meccsre ad ajánlást.")
    st.stop()

all_rows: List[Dict[str, Any]] = []
errors: List[str] = []

with st.spinner("Adatok betöltése (Understat) + CORE számítás + Modifiers (Hírek/Időjárás)…"):
    for lk in sel_leagues:
        league_name = LEAGUES[lk]
        season = DEFAULT_SEASONS.get(lk, 2024)

        try:
            fixtures, results, _teams = understat_fetch(lk, season)
        except Exception as e:
            errors.append(f"{league_name}: {e}")
            continue

        # liga fallback xG
        base_xg = league_base_xg(results)
        prof = build_team_xg_profiles(results)

        # fixtures szűrés
        now = now_utc()
        limit = now + timedelta(days=int(days_ahead))
        fx = []
        for m in fixtures or []:
            dt = parse_dt(m.get("datetime", ""))
            if not dt:
                continue
            if now <= dt <= limit:
                fx.append(m)
        fx.sort(key=lambda x: x.get("datetime", ""))

        if not fx:
            all_rows.append({
                "league": league_name,
                "season": season,
                "home": "—",
                "away": "—",
                "kickoff": None,
                "kickoff_str": "—",
                "pick": "Nincs meccs az időablakban",
                "confidence": 0.0,
                "risk_label": "INFO",
                "risk_class": "tag-warn",
                "summary": f"Ebben a ligában nincs meccs a következő {days_ahead} napban.",
            })
            continue

        for m in fx:
            home = clean_team(((m.get("h") or {}).get("title")))
            away = clean_team(((m.get("a") or {}).get("title")))
            kickoff = parse_dt(m.get("datetime", ""))

            if not home or not away or not kickoff:
                continue

            row = build_match_analysis(
                home=home,
                away=away,
                kickoff_dt=kickoff,
                league_name=league_name,
                season=season,
                prof=prof,
                base_xg=base_xg,
                use_news=use_news,
                use_weather=use_weather,
            )
            all_rows.append(row)

df = pd.DataFrame(all_rows)

if errors:
    st.warning("Néhány liga hibával tért vissza (a többi működik):\n\n" + "\n".join([f"• {x}" for x in errors]))

if df.empty:
    st.error("Nem jött vissza elemzés.")
    st.stop()

# Szűrés
if not show_all:
    df = df[df["confidence"] >= min_conf].copy()

# Rendezés: kickoff idő + bizalom
df = df.sort_values(by=["kickoff_str", "confidence"], ascending=[True, False]).reset_index(drop=True)

# =========================
#  Fő nézet
# =========================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Meccsek", int((df["home"] != "—").sum()))
with k2:
    st.metric("Ligák", len(set(df["league"].tolist())))
with k3:
    st.metric("Min bizalom", f"{min_conf*100:.0f}%")
with k4:
    good = (df["risk_label"] == "AJÁNLOTT").sum()
    st.metric("AJÁNLOTT", int(good))

st.markdown("<hr/>", unsafe_allow_html=True)

# TOP PICK
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("📌 Ajánlások (magyar elemzéssel)")

top = df[df["home"] != "—"].sort_values("confidence", ascending=False).head(1)
if not top.empty:
    t = top.iloc[0].to_dict()
    st.markdown(
        f"<div class='card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;'>"
        f"<div><b>TOP PICK</b> • <span class='small'>{t['league']}</span></div>"
        f"<div class='pill {t['risk_class']}'><b>{t['risk_label']}</b></div>"
        f"</div>"
        f"<h3 style='margin:0.35rem 0 0.35rem 0;'>{t['home']} vs {t['away']}</h3>"
        f"<div class='small'>Kezdés: <b>{t['kickoff_str']}</b> • Ajánlás: <b>{t['pick']}</b> • Bizalom: <b>{t['confidence']*100:.0f}%</b></div>"
        f"<div style='margin-top:10px;white-space:pre-wrap;'>{t['summary']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("Nincs meccs a szűrés után. Kapcsold be a 'Mutasson mindent' opciót vagy csökkentsd a minimum bizalmat.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Lista
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🧾 Meccslista")

for _, r in df.iterrows():
    if r["home"] == "—":
        st.markdown(
            f"<div class='card'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div><b>{r['league']}</b> <span class='small'>(szezon: {r['season']})</span></div>"
            f"<div class='pill {r['risk_class']}'><b>INFO</b></div>"
            f"</div>"
            f"<div class='small' style='margin-top:6px;white-space:pre-wrap;'>{r['summary']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        continue

    st.markdown(
        f"<div class='card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;'>"
        f"<div class='pill'><b>{r['league']}</b> • <span class='small'>Kezdés:</span> <b>{r['kickoff_str']}</b></div>"
        f"<div class='pill {r['risk_class']}'><b>{r['risk_label']}</b></div>"
        f"</div>"
        f"<h4 style='margin:0.55rem 0 0.35rem 0;'>{r['home']} vs {r['away']}</h4>"
        f"<div class='grid'>"
        f"  <div class='metricbox'><div class='mtitle'>Ajánlás</div><div class='mval'>{r['pick']}</div></div>"
        f"  <div class='metricbox'><div class='mtitle'>Végső bizalom</div><div class='mval'>{r['confidence']*100:.0f}%</div></div>"
        f"  <div class='metricbox'><div class='mtitle'>Össz xG</div><div class='mval'>{(r['lh']+r['la']):.2f}</div></div>"
        f"</div>"
        f"<details style='margin-top:10px;'><summary style='cursor:pointer;color:rgba(255,255,255,0.82);'>Miért ezt javasolja?</summary>"
        f"<div style='margin-top:8px;white-space:pre-wrap;'>{r['summary']}</div>"
        f"</details>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Megjegyzés: Odds / FootyStats / Referee modulok helye előkészítve (stub). "
    "A motor stabil: ha nincs adat, 0 hatással tér vissza, de az ajánlás sosem üres."
)
