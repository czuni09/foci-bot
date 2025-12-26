import os
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter, Retry

# =========================
# KONFIG
# =========================
APP_TITLE = "Match Intelligence (Robust)"
DB_PATH = "match_intel.db"
TIMEOUT = (3.05, 12)          # connect/read (ne fagyjon)  :contentReference[oaicite:2]{index=2}
TTL_API = 900                 # 15 perc cache (rate limit + gyors) :contentReference[oaicite:3]{index=3}
MAX_EVENTS = 200              # védelem túl sok adat ellen
UTC_NOW = lambda: datetime.now(timezone.utc)

try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except KeyError:
    st.error("Hiányzik: ODDS_API_KEY a Streamlit Secrets-ben.")
    st.stop()

# Választható: valós meccs/forma API kulcs (ajánlott)
# pl. FOOTBALL_DATA_KEY / API_FOOTBALL_KEY / SPORTMONKS_KEY stb.
FORM_API_KEY = st.secrets.get("FORM_API_KEY", None)

LEAGUES = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
]

# =========================
# HTTP (RETRY + 429)
# =========================
@st.cache_resource
def http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    a = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("https://", a)
    s.mount("http://", a)
    return s

S = http_session()

def _get_json(url: str, *, params: Optional[dict] = None) -> Dict[str, Any]:
    """Robusztus GET JSON: timeout + HTTP hiba + 429 tisztességes kezelése."""
    r = S.get(url, params=params, timeout=TIMEOUT)
    # Ha a szerver 429-et ad, retry mechanizmus már próbálkozott.
    # Itt még egyszer, rövid várakozással lehet kulturáltan visszamenni:
    if r.status_code == 429:
        time.sleep(2)  # Odds API javaslat: pár mp várakozás :contentReference[oaicite:4]{index=4}
        r = S.get(url, params=params, timeout=TIMEOUT)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}") from e

    try:
        return r.json()
    except Exception as e:
        raise RuntimeError("Nem JSON válasz érkezett.") from e

# =========================
# DB
# =========================
@st.cache_resource
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            league TEXT NOT NULL,
            event_id TEXT NOT NULL,
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            kickoff_utc TEXT NOT NULL,
            features_json TEXT NOT NULL,
            score REAL NOT NULL
        )
    """)
    conn.commit()
    return conn

DB = db()

# =========================
# ADATMODELL
# =========================
@dataclass(frozen=True)
class Weights:
    form: float = 0.45
    availability: float = 0.25
    market_stability: float = 0.15
    home_away_split: float = 0.15

# =========================
# ODDS API: események + alap market info (csak monitor jelleggel)
# =========================
@st.cache_data(ttl=TTL_API)
def fetch_odds_events(league: str) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/{league}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}
    data = _get_json(url, params=params)
    return data if isinstance(data, list) else []

def within_24h(kickoff_iso: str) -> bool:
    now = UTC_NOW()
    ko = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
    return now <= ko <= now + timedelta(hours=24)

def pick_best_book(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    books = m.get("bookmakers", [])
    if not books:
        return None
    # preferált: bet365, különben első
    for b in books:
        if b.get("key") == "bet365":
            return b
    return books[0]

def best_h2h_outcome(book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    markets = book.get("markets", [])
    h2h = next((x for x in markets if x.get("key") == "h2h"), None)
    if not h2h:
        return None
    outs = h2h.get("outcomes", [])
    if not outs:
        return None
    # legkisebb odds = favorit (csak megfigyelés)
    return min(outs, key=lambda x: x.get("price", 9999))

# =========================
# FORMA / HIÁNYZÓK: itt jön a "valóság"
# - Ez a rész szolgáltató-függő: TE kötsz be konkrét focis adat API-t.
# - Ha nincs, a score NEM lesz "értelmes", csak piaci meta.
# =========================
def safe_default_form() -> Dict[str, Any]:
    return {
        "last5_points": None,
        "last5_wins": None,
        "last5_goals_for": None,
        "last5_goals_against": None,
        "home_strength": None,
        "away_strength": None,
        "availability_flag": None,  # pl. kulcshiány
    }

@st.cache_data(ttl=TTL_API)
def fetch_team_context(team_name: str) -> Dict[str, Any]:
    # TODO: ide jön a valós providered (API-Football / SportMonks / stb.)
    # Ha nincs bekötve, visszaadunk üres/None értékeket.
    if not FORM_API_KEY:
        return safe_default_form()

    # Példa: itt te fogod lecserélni a saját endpointokra.
    # Biztonság kedvéért nem feltételezek endpointot.
    return safe_default_form()

# =========================
# SCORE: súlyozott, determinisztikus (NINCS random)
# =========================
def normalize_points(x: Optional[float], lo: float, hi: float) -> float:
    if x is None:
        return 0.0
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def compute_score(ctx_home: Dict[str, Any], ctx_away: Dict[str, Any], market: Dict[str, Any], w: Weights) -> Tuple[float, Dict[str, float]]:
    # FORM (pl. last5_points 0..15)
    form_home = normalize_points(ctx_home.get("last5_points"), 0, 15)
    form_away = normalize_points(ctx_away.get("last5_points"), 0, 15)
    form_component = (form_home + (1 - form_away)) / 2

    # AVAILABILITY (0/1 vagy None)
    # 1 = rendben, 0 = kulcshiány
    av_home = ctx_home.get("availability_flag")
    av_away = ctx_away.get("availability_flag")
    av_component = 0.5
    if av_home is not None and av_away is not None:
        av_component = (float(av_home) + float(1 - av_away)) / 2

    # MARKET_STABILITY: ha sok bookmaker van és nem extrém az odds, stabilabb
    book_count = market.get("bookmaker_count", 0)
    fav_price = market.get("fav_price")
    st_component = 0.0
    if fav_price is not None:
        st_component = 0.6 * normalize_points(book_count, 1, 12) + 0.4 * (1 - normalize_points(abs(fav_price - 2.2), 0, 1.2))

    # HOME/AWAY SPLIT (ha van adat)
    hs = normalize_points(ctx_home.get("home_strength"), 0, 1)
    aw = normalize_points(ctx_away.get("away_strength"), 0, 1)
    ha_component = (hs + (1 - aw)) / 2 if (ctx_home.get("home_strength") is not None and ctx_away.get("away_strength") is not None) else 0.0

    breakdown = {
        "form": form_component,
        "availability": av_component,
        "market_stability": st_component,
        "home_away_split": ha_component,
    }
    score = (
        w.form * breakdown["form"]
        + w.availability * breakdown["availability"]
        + w.market_stability * breakdown["market_stability"]
        + w.home_away_split * breakdown["home_away_split"]
    )
    return float(score), breakdown

# =========================
# PIPELINE
# =========================
def build_ranked_events(weights: Weights) -> pd.DataFrame:
    rows = []
    now = UTC_NOW()

    for lg in LEAGUES:
        try:
            events = fetch_odds_events(lg)
        except Exception:
            continue

        for m in events[:MAX_EVENTS]:
            try:
                if not within_24h(m["commence_time"]):
                    continue

                book = pick_best_book(m)
                if not book:
                    continue

                fav = best_h2h_outcome(book)
                if not fav:
                    continue

                market_meta = {
                    "bookmaker_count": len(m.get("bookmakers", [])),
                    "fav_price": float(fav.get("price")) if fav.get("price") is not None else None,
                    "fav_name": fav.get("name"),
                }

                home = m.get("home_team", "")
                away = m.get("away_team", "")
                kickoff = m["commence_time"]

                ctx_home = fetch_team_context(home)
                ctx_away = fetch_team_context(away)

                score, breakdown = compute_score(ctx_home, ctx_away, market_meta, weights)

                rows.append({
                    "league": lg,
                    "home": home,
                    "away": away,
                    "kickoff_utc": kickoff,
                    "market_fav": market_meta["fav_name"],
                    "market_fav_price": market_meta["fav_price"],
                    "bookmakers": market_meta["bookmaker_count"],
                    "score": round(score, 4),
                    **{f"w_{k}": round(v, 4) for k, v in breakdown.items()}
                })

            except Exception:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)

# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

weights = Weights(
    form=st.sidebar.slider("Form súly", 0.0, 1.0, 0.45, 0.05),
    availability=st.sidebar.slider("Hiányzók súly", 0.0, 1.0, 0.25, 0.05),
    market_stability=st.sidebar.slider("Piaci stabilitás súly", 0.0, 1.0, 0.15, 0.05),
    home_away_split=st.sidebar.slider("H/A split súly", 0.0, 1.0, 0.15, 0.05),
)

if st.button("Futtatás"):
    df = build_ranked_events(weights)
    if df.empty:
        st.warning("Nincs adat a következő 24 órában (vagy API hiba / rate limit).")
    else:
        st.subheader("Top események (elemzési pontszám szerint)")
        st.dataframe(df.head(25), use_container_width=True)

st.subheader("Korábbi mentések")
try:
    hist = pd.read_sql_query("SELECT ts_utc, league, home, away, kickoff_utc, score FROM runs ORDER BY id DESC LIMIT 200", DB)
    st.dataframe(hist, use_container_width=True)
except Exception:
    st.info("Nincs még mentett futás.")
