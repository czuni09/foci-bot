# Updated streamlit_app.py - TITAN Mission Control
# Robust, defensive, and Streamlit/Render-friendly implementation.
# See README in repo for details.

import os
import re
import math
import csv
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import requests
import feedparser

# Defensive imports for optional async libraries
try:
    import aiohttp
except Exception:
    aiohttp = None

try:
    from understat import Understat
except Exception:
    Understat = None

# Configure basic logging to file for debugging in Render/Streamlit
LOG_PATH = Path("titan_debug.log")
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("titan")

# Page config early so hard-fail pages render nicely
st.set_page_config(page_title="TITAN – Mission Control", page_icon="🛰️", layout="wide")

# Constants and defaults
LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}
DAYS_AHEAD = int(os.getenv("DAYS_AHEAD", "4"))
TOP_K = int(os.getenv("TOP_K", "2"))
MAX_GOALS = 10

# Social / news
USE_GOOGLE_NEWS_RSS = True
USE_GDELT = True
SOCIAL_MAX_ITEMS = 12
SHOW_SOCIAL_DETAILS = True
TRANSLATE_TO_HU = True

# Backtest / logging
PICKS_LOG_PATH = Path(os.getenv("PICKS_LOG_PATH", "picks_log.csv"))
AUTO_LOG_TOP_PICKS = True

# Secrets helper
def _secret(name: str) -> str:
    # Prefer environment variables (Render), fall back to Streamlit secrets
    return (os.getenv(name) or (st.secrets.get(name) if hasattr(st, "secrets") else None) or "")

NEWS_API_KEY = _secret("NEWS_API_KEY")
WEATHER_API_KEY = _secret("WEATHER_API_KEY")
ODDS_API_KEY = _secret("ODDS_API_KEY")
FOOTBALL_DATA_TOKEN = _secret("FOOTBALL_DATA_TOKEN")

# Early dependency checks that render helpful instructions
missing = []
if aiohttp is None:
    missing.append("aiohttp")
if Understat is None:
    missing.append("understat")

if missing:
    st.title("TITAN – Missing dependencies")
    st.error(f"Hiányzó csomag(ok): {', '.join(missing)}")
    st.code("pip install " + " ".join(missing) + "\n\npip install -r requirements.txt", language="bash")
    st.stop()

# UI styling (kept compact but readable)
st.markdown("""
<style>
/* Minimal style that is Render/Streamlit friendly */
body { color: #eaeef8; background: #071226; }
.hdr{ font-family: 'Inter', Arial; font-size: 1.9rem; font-weight:700; color: #79a6ff; }
.panel{ background: rgba(255,255,255,0.03); padding:12px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hdr">🛰️ TITAN – Mission Control (stabilized)</div>', unsafe_allow_html=True)

# ------------------ Utilities ------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def season_from_today() -> int:
    t = datetime.now().date()
    return t.year - 1 if t.month < 7 else t.year


def parse_dt(s: str) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    # Understat historically returns 'YYYY-MM-DD HH:MM:SS' but some endpoints may use ISO
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            # fallback: try to extract numbers
            m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", s)
            if m:
                try:
                    dt = datetime.fromisoformat(m.group(1))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return None
    return None


def fmt_local(dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    try:
        return dt.astimezone().strftime("%Y.%m.%d %H:%M")
    except Exception:
        return dt.strftime("%Y.%m.%d %H:%M")


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def clean_team(name: Optional[str]) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def stable_match_id(league_key: str, kickoff_dt: Optional[datetime], home: str, away: str) -> str:
    k = kickoff_dt.strftime("%Y%m%d%H%M") if kickoff_dt else "0000"
    return f"{league_key}:{k}:{home.lower()}__{away.lower()}"

# ------------------ Derby / Rival Exclusions ------------------
EXCLUDED_MATCHUPS = set([
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
])

EPL_BIG6 = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham"}


def is_excluded_match(league_key: str, home: str, away: str) -> bool:
    try:
        if (home, away) in EXCLUDED_MATCHUPS:
            return True
        if league_key == "epl" and home in EPL_BIG6 and away in EPL_BIG6:
            return True
    except Exception:
        logger.exception("Error evaluating exclusion")
    return False

# ------------------ Poisson model / probabilities ------------------

def poisson_pmf(lmb: float, k: int) -> float:
    # protective: if lmb is 0, return 1 at k==0
    try:
        if lmb <= 0 and k == 0:
            return 1.0
        if lmb <= 0:
            return 0.0
        return math.exp(-lmb) * (lmb ** k) / math.factorial(k)
    except Exception:
        return 0.0


def prob_over_25(lh: float, la: float, max_goals: int = MAX_GOALS) -> float:
    p = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j >= 3:
                p += poisson_pmf(lh, i) * poisson_pmf(la, j)
    return clamp(p, 0.0, 1.0)


def prob_btts(lh: float, la: float) -> float:
    try:
        p = 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh + la))
        return clamp(p, 0.0, 1.0)
    except Exception:
        return 0.0


def prob_1x2(lh: float, la: float, max_goals: int = MAX_GOALS) -> Tuple[float, float, float]:
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

# ------------------ Social: Google News RSS + GDELT + translation ------------------
NEG_KEYWORDS = [
    "injury", "injured", "ruled out", "out", "doubtful", "sidelined",
    "suspended", "suspension", "ban",
    "scandal", "arrest", "police", "court",
    "divorce", "wife", "girlfriend", "partner", "family",
]


def count_neg_hits(text: Optional[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in NEG_KEYWORDS if k in t)


def google_news_rss(query: str, hl: str = "en", gl: str = "US", ceid: str = "US:en", limit: int = 12) -> List[Dict[str, Any]]:
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
    try:
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
    except Exception:
        logger.exception("google_news_rss failed")
        return []


def gdelt_doc(query: str, maxrecords: int = 12) -> List[Dict[str, Any]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "sort": "HybridRel",
    }
    try:
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
    except Exception:
        logger.exception("gdelt_doc failed")
        return []


def build_social_query_pack(home: str, away: str) -> Tuple[str, str]:
    neg_terms = ["injury", "suspended", "scandal", "divorce", "arrest"]
    gnews_q = f'({home} OR "{away}") AND ({" OR ".join(neg_terms)})'
    gdelt_q = f'({home} OR "{away}") ({" OR ".join(neg_terms)})'
    return gnews_q, gdelt_q


def social_penalty(neg_hits: int) -> float:
    if neg_hits <= 0:
        return 0.0
    if neg_hits == 1:
        return 0.05
    if neg_hits == 2:
        return 0.08
    if 3 <= neg_hits <= 4:
        return 0.12
    return 0.15

@st.cache_data(ttl=86400, show_spinner=False)
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
    except Exception:
        logger.exception("translate_en_to_hu failed")
        return t


def maybe_hu(text: str) -> str:
    if not TRANSLATE_TO_HU:
        return text
    # Heuristic: if text looks too short or contains non-english tokens, skip translation
    if not text or len(text) < 10:
        return text
    return translate_en_to_hu(text)

# ------------------ Async runner for Streamlit-safe environment ------------------

def run_async(coro):
    """
    Run the given coroutine safely:
    - If no running loop: use asyncio.run
    - If a loop is running (Streamlit), start a new event loop in a separate thread and run the coroutine there

    Returns the result of the coroutine or raises exceptions from it.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        # no running loop
        return asyncio.run(coro)

    # running loop exists -> create a new loop in separate thread
    def _run_in_new_loop(c):
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(c)
        finally:
            try:
                new_loop.close()
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run_in_new_loop, coro)
        return fut.result()

@st.cache_data(ttl=600, show_spinner=False)
def understat_fetch(league_key: str, season: int, days_ahead: int):
    async def _run():
        if aiohttp is None or Understat is None:
            return [], []
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

def build_team_xg_profiles(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    prof: Dict[str, Dict[str, Any]] = {}

    def ensure(team: str):
        prof.setdefault(team, {"home_for": [], "home_against": [], "away_for": [], "away_against": []})

    for m in results or []:
        h = clean_team(((m.get("h") or {}).get("title")))
        a = clean_team(((m.get("a") or {}).get("title")))
        xgh = safe_float(((m.get("xG") or {}).get("h")))
        xga = safe_float(((m.get("xG") or {}).get("a")))
        if not h or not a:
            continue
        # Understat sometimes provides strings or None
        if xgh is None or xga is None:
            continue
        ensure(h); ensure(a)
        prof[h]["home_for"].append(xgh)
        prof[h]["home_against"].append(xga)
        prof[a]["away_for"].append(xga)
        prof[a]["away_against"].append(xgh)

    out: Dict[str, Dict[str, Any]] = {}
    for team, d in prof.items():
        hf = d["home_for"]; ha = d["home_against"]
        af = d["away_for"]; aa = d["away_against"]
        out[team] = {
            "home_xg_for": sum(hf) / len(hf) if hf else None,
            "home_xg_against": sum(ha) / len(ha) if ha else None,
            "away_xg_for": sum(af) / len(af) if af else None,
            "away_xg_against": sum(aa) / len(aa) if aa else None,
            "n_home": len(hf),
            "n_away": len(af),
        }
    return out

def expected_goals_from_profiles(home: str, away: str, prof: Dict[str, Dict[str, Any]], base: float = 1.35) -> Tuple[float, float, int, int]:
    ph = prof.get(home, {})
    pa = prof.get(away, {})

    h_for = ph.get("home_xg_for")
    h_against = ph.get("home_xg_against")
    a_for = pa.get("away_xg_for")
    a_against = pa.get("away_xg_against")

    lh_parts: List[float] = []
    if h_for is not None:
        lh_parts.append(h_for)
    if a_against is not None:
        lh_parts.append(a_against)
    lh = sum(lh_parts) / len(lh_parts) if lh_parts else base

    la_parts: List[float] = []
    if a_for is not None:
        la_parts.append(a_for)
    if h_against is not None:
        la_parts.append(h_against)
    la = sum(la_parts) / len(la_parts) if la_parts else base

    lh = clamp(lh, 0.2, 3.5)
    la = clamp(la, 0.2, 3.5)

    n_home = int(ph.get("n_home", 0) or 0)
    n_away = int(pa.get("n_away", 0) or 0)
    return lh, la, n_home, n_away

def label_risk(n_home: int, n_away: int, extra_penalty: float) -> Tuple[str, str]:
    if n_home >= 8 and n_away >= 8 and extra_penalty < 0.08:
        return "MEGBÍZHATÓ", "tag-good"
    if n_home >= 4 and n_away >= 4 and extra_penalty < 0.12:
        return "RIZIKÓS", "tag-warn"
    return "NAGYON RIZIKÓS", "tag-bad"

def pick_recommendation(lh: float, la: float, p1: float, px: float, p2: float, pbtts: float, pover25: float) -> Tuple[str, float, str]:
    total_xg = lh + la
    if pbtts >= 0.58 and total_xg >= 2.55:
        return ("BTTS – IGEN", pbtts, f"Mindkét csapat gól esélyes (össz xG ~ {total_xg:.2f}).")
    if pover25 >= 0.56 and total_xg >= 2.60:
        return ("Over 2.5 gól", pover25, f"Magas gólvárakozás (össz xG ~ {total_xg:.2f}).")

    mx = max(p1, px, p2)
    if mx == p1:
        return ("Hazai győzelem (1)", p1, f"Hazai oldal valószínűbb (~{p1*100:.0f}%).")
    if mx == p2:
        return ("Vendég győzelem (2)", p2, f"Vendég oldal valószínűbb (~{p2*100:.0f}%).")
    return ("Döntetlen (X)", px, f"Döntetlen valószínűbb (~{px*100:.0f}%).")


def compute_reliability(conf: float) -> int:
    return int(clamp(conf * 100.0, 0, 100))

# ------------------ Backtest log helpers ------------------
LOG_FIELDS = [
    "logged_utc", "league_key", "league", "season", "match_id",
    "kickoff_utc", "kickoff_local", "home", "away",
    "pick", "confidence", "risk_label",
    "social_neg_hits", "social_penalty",
    "result_home", "result_away", "outcome",
]


def ensure_log_file() -> None:
    if not PICKS_LOG_PATH.exists():
        with PICKS_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()


def read_log_df() -> pd.DataFrame:
    if not PICKS_LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_FIELDS)
    try:
        return pd.read_csv(PICKS_LOG_PATH)
    except Exception:
        logger.exception("Failed to read picks log file")
        return pd.DataFrame(columns=LOG_FIELDS)


def append_log_rows(rows: List[Dict[str, Any]]) -> None:
    ensure_log_file()
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


def eval_pick_outcome(pick: str, gh: Optional[int], ga: Optional[int]) -> str:
    pick = (pick or "").strip().lower()
    if gh is None or ga is None:
        return "UNKNOWN"
    total = gh + ga
    if "btts" in pick:
        return "WIN" if (gh >= 1 and ga >= 1) else "LOSS"
    if "over 2.5" in pick:
        return "WIN" if total >= 3 else "LOSS"
    if "(1)" in pick or "hazai" in pick:
        return "WIN" if gh > ga else ("PUSH" if gh == ga else "LOSS")
    if "(2)" in pick or "vend" in pick:
        return "WIN" if ga > gh else ("PUSH" if gh == ga else "LOSS")
    if "(x)" in pick or "döntetlen" in pick or "dontetlen" in pick:
        return "WIN" if gh == ga else "LOSS"
    return "UNKNOWN"

@st.cache_data(ttl=600, show_spinner=False)
def understat_results_for_league(league_key: str, season: int):
    async def _run():
        if aiohttp is None or Understat is None:
            return []
        async with aiohttp.ClientSession() as session:
            u = Understat(session)
            results = await u.get_league_results(league_key, season)
            return results or []
    return run_async(_run())


def find_result_for_match(results: List[Dict[str, Any]], home: str, away: str, kickoff_utc: datetime):
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


def verify_log_outcomes() -> Tuple[pd.DataFrame, int]:
    df = read_log_df()
    if df.empty:
        return df, 0
    unresolved = df[(df["outcome"].isna()) | (df["outcome"].astype(str) == "") | (df["outcome"].astype(str) == "UNKNOWN")]
    if unresolved.empty:
        return df, 0

    updated = 0
    cache_results: Dict[str, List[Dict[str, Any]]] = {}
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

# ------------------ Build match analysis ------------------

def build_match_analysis(league_key: str, league_name: str, season: int, home: str, away: str, kickoff_dt: datetime, prof: Dict[str, Any]) -> Dict[str, Any]:
    lh, la, n_home, n_away = expected_goals_from_profiles(home, away, prof)
    pbtts = prob_btts(lh, la)
    pover25 = prob_over_25(lh, la)
    p1, px, p2 = prob_1x2(lh, la)

    pick, pval, why = pick_recommendation(lh, la, p1, px, p2, pbtts, pover25)

    social = {"gnews": [], "gdelt": [], "neg_hits": 0, "risk_penalty": 0.0}
    if USE_GOOGLE_NEWS_RSS or USE_GDELT:
        gnews_q, gdelt_q = build_social_query_pack(home, away)
        try:
            if USE_GOOGLE_NEWS_RSS:
                gnews = google_news_rss(gnews_q, limit=SOCIAL_MAX_ITEMS)
                if TRANSLATE_TO_HU:
                    for x in gnews:
                        x["title_hu"] = maybe_hu(x.get("title", ""))
                social["gnews"] = gnews
                social["neg_hits"] += sum(count_neg_hits(x.get("title", "")) for x in gnews)

            if USE_GDELT:
                garts = gdelt_doc(gdelt_q, maxrecords=SOCIAL_MAX_ITEMS)
                if TRANSLATE_TO_HU:
                    for x in garts:
                        x["title_hu"] = maybe_hu(x.get("title", ""))
                social["gdelt"] = garts
                for a in garts:
                    social["neg_hits"] += count_neg_hits(a.get("title", ""))
                    tone = a.get("tone")
                    if isinstance(tone, (int, float)) and tone < -4:
                        social["neg_hits"] += 1
        except Exception:
            logger.exception("Social fetch failed for %s vs %s", home, away)

    spen = social_penalty(int(social["neg_hits"] or 0))
    social["risk_penalty"] = spen

    conf_adj = clamp(pval - spen, 0.0, 1.0)
    risk_label, risk_class = label_risk(n_home, n_away, extra_penalty=spen)

    summary_lines = [
        f"**xG várható gól:** {home} `{lh:.2f}` • {away} `{la:.2f}` • össz `{(lh+la):.2f}`",
        f"**Piac esélyek:** BTTS `{pbtts*100:.0f}%` • Over2.5 `{pover25*100:.0f}%` • 1/X/2 `{p1*100:.0f}/{px*100:.0f}/{p2*100:.0f}%`",
        f"**Ajánlás:** **{pick}** • alapesély `{pval*100:.0f}%` → korrigált `{conf_adj*100:.0f}%`",
        f"**Rizikó:** **{risk_label}** • adat: (H `{n_home}` / A `{n_away}`)",
    ]
    if spen > 0:
        summary_lines.append(f"🧠 **Hír/narratíva penalty:** −{spen*100:.0f}% (neg találat: {social['neg_hits']})")

    signals = []
    if spen > 0:
        signals.append("🧠 News/Sentiment")
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
        "lh": lh, "la": la,
        "pick": pick,
        "confidence_raw": pval,
        "confidence": conf_adj,
        "reliability": compute_reliability(conf_adj),
        "risk_label": risk_label,
        "risk_class": risk_class,
        "social": social,
        "social_penalty": spen,
        "signals": signals,
        "why": why,
        "summary": "\n".join(summary_lines),
        "quality_home": n_home,
        "quality_away": n_away,
    }

# ------------------ MAIN UI & Flow ------------------
season = season_from_today()

status_pills = [
    "Understat ✅",
    "RSS ✅" if USE_GOOGLE_NEWS_RSS else "RSS —",
    "GDELT ✅" if USE_GDELT else "GDELT —",
    "HU fordítás ✅" if TRANSLATE_TO_HU else "HU fordítás —",
]

st.markdown(f"""
<div style='display:flex;gap:12px;align-items:center'>
  <div style='padding:6px 10px;border-radius:8px;background:rgba(255,255,255,0.03)'>⏱️ Frissítve: <b>{datetime.now().astimezone().strftime('%Y.%m.%d %H:%M')}</b></div>
  {''.join([f"<div style='padding:6px 10px;border-radius:8px;background:rgba(255,255,255,0.02)'>{p}</div>" for p in status_pills])}
</div>
""", unsafe_allow_html=True)

logdf, updated_n = verify_log_outcomes()
if updated_n > 0:
    st.success(f"Utóellenőrzés: {updated_n} mentett tipp frissítve eredménnyel.")

rows: List[Dict[str, Any]] = []
errors: List[str] = []

with st.spinner("Autonóm elemzés: Understat xG + rangadó/derby kizárás + hírek fordítása…"):
    for lk, league_name in LEAGUES.items():
        try:
            fixtures, results = understat_fetch(lk, season, DAYS_AHEAD)
        except Exception as e:
            errors.append(f"{league_name}: {e}")
            logger.exception("understat_fetch failed for %s", lk)
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

            rows.append(build_match_analysis(lk, league_name, season, home, away, kickoff, prof))

if errors:
    st.warning("Néhány liga hibával tért vissza:\n\n" + "\n".join([f"• {x}" for x in errors]))

if not rows:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("Nincs ajánlható meccs az időablakban (derby/rangadók kizárva).")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Build dataframe and take top picks

df = pd.DataFrame(rows).sort_values(by=["confidence", "kickoff"], ascending=[False, True]).reset_index(drop=True)
top2 = df.head(TOP_K).copy()

# Auto-log top picks
if AUTO_LOG_TOP_PICKS and not top2.empty:
    log_rows: List[Dict[str, Any]] = []
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
            "result_home": "",
            "result_away": "",
            "outcome": "UNKNOWN",
        })
    append_log_rows(log_rows)

# Dashboard
st.markdown("<hr/>", unsafe_allow_html=True)
resolved = pd.DataFrame()
if not logdf.empty and "outcome" in logdf.columns:
    resolved = logdf[logdf["outcome"].isin(["WIN", "LOSS", "PUSH"]) ]

wins = int((resolved["outcome"] == "WIN").sum()) if not resolved.empty else 0
loss = int((resolved["outcome"] == "LOSS").sum()) if not resolved.empty else 0
push = int((resolved["outcome"] == "PUSH").sum()) if not resolved.empty else 0
total_res = wins + loss + push
winrate = (wins / total_res * 100) if total_res > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Közelgő meccsek", int(len(df)))
with c2:
    st.metric("TOP pickek", TOP_K)
with c3:
    st.metric("Lezárt tippek", total_res)
with c4:
    st.metric("Winrate", f"{winrate:.1f}%")

# Top picks cards
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🎯 A két legjobb választás (autonóm)")

for idx, r in top2.iterrows():
    rel = int(r["reliability"])
    signals = " ".join([f"<span style='background:rgba(255,255,255,0.02);padding:4px;border-radius:8px;margin-right:6px'>{s}</span>" for s in r["signals"]])

    extra_lines: List[str] = []
    if SHOW_SOCIAL_DETAILS and isinstance(r.get("social"), dict):
        gnews = (r["social"] or {}).get("gnews", []) or []
        gd = (r["social"] or {}).get("gdelt", []) or []
        for x in gnews[:2]:
            t = x.get("title_hu") if TRANSLATE_TO_HU else x.get("title")
            if t:
                extra_lines.append(f"• {t}")
        for x in gd[:1]:
            t = x.get("title_hu") if TRANSLATE_TO_HU else x.get("title")
            if t:
                extra_lines.append(f"• {t}")

    extra_html = ""
    if extra_lines and r.get("social_penalty", 0.0) > 0:
        extra_html = "<div style='margin-top:8px; white-space:pre-wrap; color:#cbd5e1'>" + "\n".join(extra_lines) + "</div>"

    st.markdown(f"""
<div style='padding:12px;background:rgba(255,255,255,0.02);border-radius:12px;margin-bottom:10px'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <div style='font-size:0.9rem'><b>#{idx+1}</b> • <span style='opacity:0.85'>{r['league']}</span> • Kezdés: <b>{r['kickoff_str']}</b></div>
    <div style='padding:8px;border-radius:10px;background:rgba(255,255,255,0.03)'><b>{r['risk_label']}</b> • {rel}%</div>
  </div>
  <h3 style='margin:8px 0 6px 0'>{r['home']} vs {r['away']}</h3>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px'>
    <div style='padding:8px;background:rgba(255,255,255,0.01);border-radius:8px'>
      <div style='opacity:0.85'>Ajánlás</div><div style='font-weight:700'>{r['pick']}</div></div>
    <div style='padding:8px;background:rgba(255,255,255,0.01);border-radius:8px'>
      <div style='opacity:0.85'>Valószínűség</div><div style='font-weight:700'>{r['confidence']*100:.0f}%</div></div>
    <div style='padding:8px;background:rgba(255,255,255,0.01);border-radius:8px'>
      <div style='opacity:0.85'>Össz xG</div><div style='font-weight:700'>{(r['lh']+r['la']):.2f}</div></div>
  </div>
  <div style='margin-top:8px'>
    {signals}
  </div>
  <details style='margin-top:8px;color:#cbd5e1'>
    <summary>Miért ezt?</summary>
    <div style='margin-top:8px; white-space:pre-wrap;'>{r['summary']}</div>
    {extra_html}
  </details>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Backtest panel
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🧾 Backtest / Utóellenőrzés (mentett tippek)")
left, right = st.columns([1, 1])
with left:
    st.write(f"Mentett tippek: **{len(logdf)}**")
    st.write(f"Lezárt: **{total_res}** • WIN: **{wins}** • LOSS: **{loss}** • PUSH: **{push}**")
with right:
    if st.button("🔁 Eredmények újraellenőrzése", use_container_width=True):
        df2, n2 = verify_log_outcomes()
        st.success(f"Frissítve: {n2} tipp.")
        logdf = df2

if not logdf.empty:
    view = logdf.sort_values(by=["logged_utc"], ascending=False).head(15)
    st.dataframe(view, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)
st.caption("Derby/rangadó kizárás aktív. Ha még több párosítást akarsz tiltani, az EXCLUDED_MATCHUPS listát bővítsd.")

# End of file
