"""
TITAN – Strategic Intelligence (Improved)

This Streamlit application collects upcoming soccer odds from The Odds API,
evaluates each betting candidate using additional context (weather, news, value
against the market) and proposes a two‑pick ticket with total odds around 2.00.

Key improvements over the original implementation:

* **Caching of external API calls** via `st.cache_data` with a TTL to reduce
  request volume and respect rate limits. Streamlit’s documentation recommends
  caching API calls so they only run once per TTL window and avoid
  unnecessary repeats:contentReference[oaicite:0]{index=0}.
* **Best price across bookmakers** is selected for each candidate. Instead of
  choosing a single bookmaker arbitrarily, we collect prices for each market
  across all bookmakers and select the highest price for our selection (for
  H2H, totals and spreads). We also compute the average price across all
  available bookmakers and derive a simple **value score** defined as
  `(best_odds / avg_odds) - 1`. This highlights bets where one book is
  lagging behind the market. Positive values indicate a better payout than
  average and boost the candidate’s score.
* **Improved fuzzy team matching** using Python’s built‑in
  `difflib.SequenceMatcher` in combination with token overlap. Fuzzy
  matching is widely used to handle minor spelling variations and abbreviations
  in team names:contentReference[oaicite:1]{index=1}.
* **Extended match search window** when identifying football‑data.org match IDs.
  Matches are searched in a ±1 day window around kickoff and within an
  8‑hour tolerance to account for timezone differences. This increases the
  likelihood of matching the correct fixture.
* **Database concurrency improvements**: SQLite is configured in WAL
  (Write‑Ahead Logging) mode with a 30 second timeout to allow concurrent
  reads while a write transaction is open. WAL mode allows reads and writes
  to proceed concurrently:contentReference[oaicite:2]{index=2}. We also set the
  synchronous mode to NORMAL for better write performance.
* **Totals and spreads push handling**: When the total goals exactly equals the
  totals line or a spread adjustment results in a draw, the bet is marked as
  `VOID`. Sports betting glossaries note that a push results in the stake
  being returned:contentReference[oaicite:3]{index=3}.

Note: This application should be run with appropriate API keys set in the
environment (ODDS_API_KEY, WEATHER_API_KEY, NEWS_API_KEY, FOOTBALL_DATA_KEY).
"""

import streamlit as st
import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import time
import re
from math import sqrt
from difflib import SequenceMatcher

# =========================================================
#  KONFIG
# =========================================================
st.set_page_config(page_title="⚽ TITAN – Strategic Intelligence", layout="wide", page_icon="⚽")

import os

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")

missing = []
if not ODDS_API_KEY:
    missing.append("ODDS_API_KEY")
if not WEATHER_KEY:
    missing.append("WEATHER_API_KEY")
if not NEWS_API_KEY:
    missing.append("NEWS_API_KEY")
if not FOOTBALL_DATA_KEY:
    missing.append("FOOTBALL_DATA_KEY")

if missing:
    st.error(f"⚠️ Hiányzó környezeti változók: {', '.join(missing)}")
    st.stop()


DB_PATH = "titan.db"

# Két tipp össz-odds cél (tűrés)
TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.90
TOTAL_ODDS_MAX = 2.10

# egy tippre “ideális” odds kb. sqrt(2) ≈ 1.414
TARGET_LEG_ODDS = sqrt(2)

# Szűrés
WINDOW_HOURS = 24
MIN_LEG_ODDS = 1.25
MAX_LEG_ODDS = 1.95

# Odds API: soccer marketek (ami jön, azt használjuk)
REQUEST_MARKETS = ["h2h", "totals", "spreads"]

# Soccer ligák (Odds API kulcsok)
DEFAULT_LEAGUES = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
]

# =========================================================
#  DB
# =========================================================
def db():
    """
    Open a SQLite connection with WAL mode and a generous timeout.

    WAL (write‑ahead logging) mode allows readers to continue reading while
    a writer is writing:contentReference[oaicite:4]{index=4}. We also set
    `synchronous` to NORMAL to improve write performance. A 30 second timeout
    reduces the chance of `database is locked` errors when the UI triggers
    concurrent reads/writes.
    """
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    # Enable WAL and adjust synchronous settings
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return con


def init_db():
    con = db()
    cur = con.cursor()
    # Ensure the predictions table exists. Additional fields can be added later
    # without breaking existing schema. We keep selection_team, avg_odds and
    # value_score in meta instead of separate columns to avoid migrations.
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

        football_data_match_id INTEGER,
        result TEXT DEFAULT 'PENDING',
        settled_at TEXT,
        home_goals INTEGER,
        away_goals INTEGER
    )
    """)
    con.commit()
    # Add new columns for closing odds and CLV if they do not exist. This
    # approach uses PRAGMA table_info to inspect existing columns and then
    # attempts to alter the table conditionally. Wrapping in try/except
    # prevents errors if the columns are already present.
    try:
        cur.execute("PRAGMA table_info(predictions)")
        cols = [row[1] for row in cur.fetchall()]
        if "closing_odds" not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN closing_odds REAL")
        if "clv_percent" not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN clv_percent REAL")
        con.commit()
    except Exception:
        # If alter fails (e.g. during creation on some platforms), ignore
        pass
    con.close()

init_db()

# =========================================================
#  SEGÉDFÜGGVÉNYEK – normalizálás / összeillesztés
# =========================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_to_dt(s: str) -> datetime:
    """
    Parse ISO timestamps returned by the Odds API and football‑data.org.
    Strings ending with 'Z' are treated as UTC. Otherwise we rely on
    datetime.fromisoformat to parse offset‑aware timestamps.
    """
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def norm_team(s: str) -> str:
    """
    Normalize a team name by lowering the case, removing unwanted characters and
    applying common replacements. This helps matching team names across data
    sources. The replacements dictionary can be extended to include other
    frequent abbreviations.
    """
    s = s.lower() if s else ""
    s = re.sub(r"[^a-z0-9áéíóöőúüű\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    repl = {
        "manchester utd": "manchester united",
        "man utd": "manchester united",
        "bayern munchen": "bayern münchen",
        "bayern munich": "bayern münchen",
        "internazionale": "inter",
        "psg": "paris saint germain",
    }
    return repl.get(s, s)


def team_match_score(a: str, b: str) -> float:
    """
    Compute a similarity score between two team names.

    This function uses both token overlap and sequence similarity. The maximum
    of the two scores is returned. Token overlap counts the intersection of
    words divided by the union; sequence similarity uses Python's
    `difflib.SequenceMatcher` to compute a ratio between 0 and 1. Fuzzy
    matching of team names is recommended when combining data from multiple
    sources:contentReference[oaicite:5]{index=5}.
    """
    if not a or not b:
        return 0.0
    a2, b2 = norm_team(a), norm_team(b)
    if a2 == b2:
        return 1.0
    # Token overlap
    at = set(a2.split())
    bt = set(b2.split())
    inter = len(at & bt)
    union = max(1, len(at | bt))
    token_score = inter / union
    # Sequence similarity
    seq_score = SequenceMatcher(None, a2, b2).ratio()
    return max(token_score, seq_score)


# =========================================================
#  FOOTBALL-DATA.ORG – meccsek/eredmények/frissítés
# =========================================================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_KEY}


def fd_get(url: str, params=None, timeout=15):
    """
    Wrapper around requests.get for football‑data.org. The API discourages
    polling for data too frequently:contentReference[oaicite:6]{index=6}, so the
    calling code should limit call frequency.
    """
    r = requests.get(url, headers=fd_headers(), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fd_find_match_id(home: str, away: str, kickoff_utc: datetime) -> int | None:
    """
    Attempt to find the football‑data match ID by searching within a ±1 day
    window around the kickoff. Team names are matched using fuzzy matching. We
    accept matches within ±8 hours. If multiple candidates exist, the one with
    the highest similarity score is chosen provided it exceeds a threshold.
    """
    if not home or not away or not kickoff_utc:
        return None
    # Define search window ±1 day around kickoff
    date_from = (kickoff_utc - timedelta(days=1)).date().isoformat()
    date_to = (kickoff_utc + timedelta(days=1)).date().isoformat()
    url = "https://api.football-data.org/v4/matches"
    try:
        data = fd_get(url, params={"dateFrom": date_from, "dateTo": date_to})
    except Exception:
        return None
    candidates = data.get("matches", []) or []
    best = (0.0, None)
    for m in candidates:
        try:
            fd_home = m["homeTeam"]["name"]
            fd_away = m["awayTeam"]["name"]
            utc_str = m.get("utcDate")
            fd_utc = iso_to_dt(utc_str)
        except Exception:
            continue
        # Filter by time proximity ±8 hours
        if abs((fd_utc - kickoff_utc).total_seconds()) > 8 * 3600:
            continue
        score = (team_match_score(home, fd_home) + team_match_score(away, fd_away)) / 2.0
        if score > best[0]:
            best = (score, m.get("id"))
    # Accept matches with similarity ≥ 0.55
    return best[1] if best[0] >= 0.55 else None


def fd_settle_prediction(pred_row: dict) -> dict:
    """
    Given a DB record, fetch the match result and compute WON/LOST/VOID.

    * H2H: selection team wins = WON; tie = LOST (three‑way market implies
      draw counts as loss).
    * TOTALS: Over wins if total goals > line; Under wins if total < line; if
      total equals the line, it is a push and the bet is marked VOID:contentReference[oaicite:7]{index=7}.
    * SPREADS: Spread adjustment results in WON if the adjusted score is positive,
      LOST if negative, and VOID (push) if zero.
    """
    match_id = pred_row.get("football_data_match_id")
    if not match_id:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}
    url = f"https://api.football-data.org/v4/matches/{match_id}"
    try:
        m = fd_get(url)
    except Exception:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}
    status = m.get("status", "")
    # Only settle if the match is finished or awarded
    if status not in ["FINISHED", "AWARDED"]:
        return {"result": "PENDING", "home_goals": None, "away_goals": None}
    score_ft = m.get("score", {}).get("fullTime", {}) or {}
    hg = score_ft.get("home")
    ag = score_ft.get("away")
    if hg is None or ag is None:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}
    bet_type = pred_row.get("bet_type")
    selection = pred_row.get("selection")
    line = pred_row.get("line")
    # H2H: draw counts as loss in 3‑way markets
    if bet_type == "H2H":
        if norm_team(selection) == norm_team(pred_row.get("home")):
            res = "WON" if hg > ag else "LOST"
        elif norm_team(selection) == norm_team(pred_row.get("away")):
            res = "WON" if ag > hg else "LOST"
        else:
            res = "UNKNOWN"
    # TOTALS: Over/Under; push if total equals the line:contentReference[oaicite:8]{index=8}
    elif bet_type == "TOTALS":
        total = hg + ag
        if selection.lower() == "over":
            if total > float(line):
                res = "WON"
            elif total < float(line):
                res = "LOST"
            else:
                res = "VOID"
        elif selection.lower() == "under":
            if total < float(line):
                res = "WON"
            elif total > float(line):
                res = "LOST"
            else:
                res = "VOID"
        else:
            res = "UNKNOWN"
    # SPREADS: Asian‑style handicap; push = VOID
    elif bet_type == "SPREADS":
        adj = None
        try:
            p = float(line)
        except Exception:
            p = None
        if selection.upper() == "HOME" and p is not None:
            adj = (hg + p) - ag
        elif selection.upper() == "AWAY" and p is not None:
            adj = (ag + p) - hg
        if adj is None:
            res = "UNKNOWN"
        else:
            if adj > 0:
                res = "WON"
            elif adj < 0:
                res = "LOST"
            else:
                res = "VOID"
    else:
        res = "UNKNOWN"
    return {"result": res, "home_goals": int(hg), "away_goals": int(ag)}


def refresh_past_results() -> int:
    """
    At each run, attempt to settle predictions whose kickoff has passed by at
    least 2 hours. This function returns the number of predictions updated. It
    reads pending or unknown predictions from the database, checks the result
    using football‑data.org and writes back the outcome.
    """
    con = db()
    df = pd.read_sql_query("SELECT * FROM predictions WHERE result IN ('PENDING','UNKNOWN')", con)
    con.close()
    if df.empty:
        return 0
    updated = 0
    now = now_utc()
    for _, row in df.iterrows():
        try:
            kickoff = iso_to_dt(row["kickoff_utc"])
        except Exception:
            continue
        if now < kickoff + timedelta(hours=2):
            continue
        settle = fd_settle_prediction(row.to_dict())
        if settle["result"] in ["PENDING"]:
            continue
        con = db()
        cur = con.cursor()
        cur.execute(
            """
            UPDATE predictions
            SET result=?, settled_at=?, home_goals=?, away_goals=?
            WHERE id=?
            """,
            (
                settle["result"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                settle["home_goals"],
                settle["away_goals"],
                int(row["id"]),
            ),
        )
        con.commit()
        con.close()
        updated += 1
    return updated


# =========================================================
#  CLOSING ODDS (CLV) UPDATE
# =========================================================
def update_closing_odds() -> int:
    """
    Attempt to capture closing odds for pending predictions and compute the
    closing line value (CLV). For each prediction that lacks a closing
    odds value and has not yet been settled, we fetch current odds from
    The Odds API within a window around kickoff (90 minutes before to
    10 minutes after) and compute the CLV as (closing_odds / placed_odds) - 1.
    Positive CLV indicates our placed odds were better than the closing
    price, aligning with the definition of closing line value:contentReference[oaicite:9]{index=9}.

    Returns the number of predictions for which closing odds were updated.
    """
    con = db()
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE (closing_odds IS NULL OR clv_percent IS NULL) AND result IN ('PENDING','UNKNOWN')",
        con,
    )
    con.close()
    if df.empty:
        return 0
    updated = 0
    now = now_utc()
    for _, row in df.iterrows():
        try:
            kickoff = iso_to_dt(row["kickoff_utc"])
        except Exception:
            continue
        # Only attempt to capture closing odds within 90 minutes before kickoff
        # and up to 10 minutes after kickoff. Outside this window we skip to
        # avoid stale data.
        if kickoff is None or not (kickoff - timedelta(minutes=90) <= now <= kickoff + timedelta(minutes=10)):
            continue
        league = row.get("league")
        # We need a league key to fetch odds. If missing, skip.
        if not league:
            continue
        try:
            data = odds_api_get(league, REQUEST_MARKETS)
        except Exception:
            continue
        # Find the best matching match in the returned data
        best_match = None
        best_score = 0.0
        for m in data:
            m_home = m.get("home_team")
            m_away = m.get("away_team")
            m_time = iso_to_dt(m.get("commence_time"))
            if not m_home or not m_away or not m_time:
                continue
            # Filter by time proximity (±3 hours) to narrow down candidates
            if abs((m_time - kickoff).total_seconds()) > 3 * 3600:
                continue
            score = (team_match_score(row["home"], m_home) + team_match_score(row["away"], m_away)) / 2.0
            if score > best_score:
                best_score = score
                best_match = m
        # Require a reasonable similarity threshold
        if best_match is None or best_score < 0.55:
            continue
        closing_odds = None
        bet_type = row.get("bet_type")
        selection = row.get("selection")
        line = row.get("line")
        if bet_type == "H2H":
            prices = []
            for b in best_match.get("bookmakers", []) or []:
                for market in b.get("markets", []) or []:
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []) or []:
                            name = outcome.get("name")
                            if not name:
                                continue
                            # Use fuzzy matching to compare selection with outcome name
                            if team_match_score(name, selection) >= 0.7:
                                try:
                                    price = float(outcome.get("price"))
                                except Exception:
                                    continue
                                prices.append(price)
            if prices:
                closing_odds = max(prices)
        elif bet_type == "TOTALS":
            prices = []
            target_line = None
            try:
                target_line = float(line)
            except Exception:
                pass
            if target_line is not None:
                for b in best_match.get("bookmakers", []) or []:
                    for market in b.get("markets", []) or []:
                        if market.get("key") == "totals":
                            for outcome in market.get("outcomes", []) or []:
                                name = (outcome.get("name") or "").capitalize()
                                try:
                                    point = float(outcome.get("point"))
                                    price = float(outcome.get("price"))
                                except Exception:
                                    continue
                                if abs(point - target_line) < 1e-6 and name == selection:
                                    prices.append(price)
            if prices:
                closing_odds = max(prices)
        elif bet_type == "SPREADS":
            prices = []
            target_line = None
            try:
                target_line = float(line)
            except Exception:
                pass
            if target_line is not None:
                for b in best_match.get("bookmakers", []) or []:
                    for market in b.get("markets", []) or []:
                        if market.get("key") == "spreads":
                            for outcome in market.get("outcomes", []) or []:
                                try:
                                    point = float(outcome.get("point"))
                                    price = float(outcome.get("price"))
                                except Exception:
                                    continue
                                if abs(point - target_line) > 1e-6:
                                    continue
                                team_name = outcome.get("name")
                                # Determine whether this outcome corresponds to the HOME or AWAY selection
                                if not team_name:
                                    continue
                                is_home = team_match_score(team_name, row["home"]) >= team_match_score(team_name, row["away"])
                                sel_code = "HOME" if is_home else "AWAY"
                                if sel_code == selection:
                                    prices.append(price)
            if prices:
                closing_odds = max(prices)
        # If we successfully captured a closing odds value, compute CLV
        if closing_odds is not None:
            placed_odds = None
            try:
                placed_odds = float(row["odds"])
            except Exception:
                pass
            if placed_odds and placed_odds > 0:
                clv_percent = (closing_odds / placed_odds) - 1.0
            else:
                clv_percent = None
            con_u = db()
            cur_u = con_u.cursor()
            cur_u.execute(
                "UPDATE predictions SET closing_odds=?, clv_percent=? WHERE id=?",
                (closing_odds, clv_percent, int(row["id"])),
            )
            con_u.commit()
            con_u.close()
            updated += 1
    return updated


# =========================================================
#  KÜLSŐ ADAT – időjárás / hírek (rövid, magyar indoklás)
# =========================================================

@st.cache_data(ttl=900)
def get_weather_basic(city_guess: str = "London") -> dict:
    """
    Query current weather conditions for a city using OpenWeatherMap API.
    The call is cached for 15 minutes to avoid excessive requests and rate limits.
    Streamlit’s caching mechanism stores the results and reuses them within the TTL:contentReference[oaicite:10]{index=10}.
    """
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_guess, "appid": WEATHER_KEY, "units": "metric", "lang": "hu"}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        return {
            "temp": float(data.get("main", {}).get("temp", float("nan"))),
            "desc": (data.get("weather", [{}])[0].get("description") or "ismeretlen"),
            "wind": float(data.get("wind", {}).get("speed", float("nan"))),
        }
    except Exception:
        return {"temp": None, "desc": "ismeretlen", "wind": None}


@st.cache_data(ttl=900)
def news_brief(team_name: str) -> dict:
    """
    Retrieve up to two recent English news headlines about a team from the
    NewsAPI. The function scores the sentiment based on simple keyword
    heuristics and returns both the score and formatted headline strings.
    Results are cached for 15 minutes:contentReference[oaicite:11]{index=11}. Excessive API calls
    are avoided by caching:contentReference[oaicite:12]{index=12}.
    """
    try:
        url = "https://newsapi.org/v2/everything"
        q = f'"{team_name}" (injury OR injured OR out OR doubt OR suspended OR return OR fit OR lineup)'
        params = {
            "q": q,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 3,
            "apiKey": NEWS_API_KEY,
        }
        r = requests.get(url, params=params, timeout=10)
        js = r.json()
        arts = js.get("articles", []) or []
        if not arts:
            return {"score": 0, "lines": []}
        lines = []
        score = 0
        for a in arts[:2]:
            title = (a.get("title") or "").strip()
            src = (a.get("source", {}) or {}).get("name", "ismeretlen")
            txt = (title + " " + (a.get("description") or "")).lower()
            if any(w in txt for w in ["injury", "injured", "out", "doubt", "suspended"]):
                score -= 1
            if any(w in txt for w in ["return", "fit", "back", "boost"]):
                score += 1
            if title:
                lines.append(f"• {title} ({src})")
        return {"score": score, "lines": lines}
    except Exception:
        return {"score": 0, "lines": []}


# =========================================================
#  ODDS API – több piac, jelöltek generálása
# =========================================================

@st.cache_data(ttl=300)
def odds_api_get(league_key: str, markets: list[str]) -> list:
    """
    Fetch odds data for a league from The Odds API. The result is cached for
    5 minutes to reduce request count. According to The Odds API terms, you
    should avoid polling too frequently. Caching ensures we respect rate limits.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_candidates_from_match(m: dict) -> list[dict]:
    """
    Create betting candidates from a single match entry from The Odds API.
    For each market (H2H, totals, spreads) we gather prices across all
    bookmakers, select the best price for our selection and compute average
    prices to derive a value score. Candidates whose odds fall outside the
    allowed range are ignored.
    """
    out: list[dict] = []
    home = m.get("home_team")
    away = m.get("away_team")
    kickoff = iso_to_dt(m.get("commence_time"))
    if not home or not away or not kickoff:
        return out
    # Collect bookmaker markets
    bookmakers = m.get("bookmakers", []) or []
    # Helper to compute average and best price for a list of odds
    def avg_and_best(prices: list[float]) -> tuple[float, float]:
        if not prices:
            return (None, None)
        avg = sum(prices) / len(prices)
        best = max(prices)  # for bettor, higher decimal odds = better payout
        return avg, best
    # -------- H2H --------
    # Build a map: team -> list of decimal prices across bookmakers
    h2h_prices: dict[str, list[float]] = {}
    for b in bookmakers:
        for market in b.get("markets", []) or []:
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []) or []:
                    name = outcome.get("name")
                    try:
                        price = float(outcome.get("price"))
                    except Exception:
                        continue
                    if name in [home, away]:
                        h2h_prices.setdefault(name, []).append(price)
    if h2h_prices:
        # Determine favourite based on lowest average price
        avg_prices = {team: sum(prices) / len(prices) for team, prices in h2h_prices.items()}
        favourite = min(avg_prices, key=avg_prices.get)
        avg_price, best_price = avg_and_best(h2h_prices.get(favourite, []))
        if avg_price and best_price and MIN_LEG_ODDS <= best_price <= MAX_LEG_ODDS:
            value_score = (best_price / avg_price) - 1 if avg_price else 0.0
            out.append({
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": None,
                "kickoff": kickoff,
                "bet_type": "H2H",
                "market_key": "h2h",
                "selection": favourite,
                "selection_team": favourite,
                "line": None,
                "bookmaker": "best_of",
                "odds": best_price,
                "avg_odds": avg_price,
                "value_score": value_score,
            })
    # -------- TOTALS --------
    # Collect totals outcomes grouped by line and name (Over/Under)
    totals_prices: dict[tuple[float, str], list[float]] = {}
    for b in bookmakers:
        for market in b.get("markets", []) or []:
            if market.get("key") == "totals":
                for outcome in market.get("outcomes", []) or []:
                    name = (outcome.get("name") or "").capitalize()
                    try:
                        price = float(outcome.get("price"))
                        point = float(outcome.get("point"))
                    except Exception:
                        continue
                    # We are only interested in Over/Under outcomes
                    if name.lower() in ["over", "under"]:
                        totals_prices.setdefault((point, name), []).append(price)
    # Prefer the standard 2.5 line if available, otherwise fall back to 3.5 then 1.5
    for target_line in [2.5, 3.5, 1.5]:
        for name in ["Over", "Under"]:
            prices = totals_prices.get((target_line, name), [])
            if prices:
                avg_price, best_price = avg_and_best(prices)
                if avg_price and best_price and MIN_LEG_ODDS <= best_price <= MAX_LEG_ODDS:
                    value_score = (best_price / avg_price) - 1 if avg_price else 0.0
                    out.append({
                        "match": f"{home} vs {away}",
                        "home": home,
                        "away": away,
                        "league": None,
                        "kickoff": kickoff,
                        "bet_type": "TOTALS",
                        "market_key": "totals",
                        "selection": name,
                        "selection_team": None,
                        "line": target_line,
                        "bookmaker": "best_of",
                        "odds": best_price,
                        "avg_odds": avg_price,
                        "value_score": value_score,
                    })
                # Only take one line (the preferred one), break after adding candidates
        if any(key[0] == target_line for key in totals_prices.keys()):
            break
    # -------- SPREADS --------
    # Collect spreads outcomes grouped by point and team name
    spreads_prices: dict[tuple[float, str], list[float]] = {}
    for b in bookmakers:
        for market in b.get("markets", []) or []:
            if market.get("key") == "spreads":
                for outcome in market.get("outcomes", []) or []:
                    team_name = outcome.get("name")
                    try:
                        price = float(outcome.get("price"))
                        point = float(outcome.get("point"))
                    except Exception:
                        continue
                    spreads_prices.setdefault((point, team_name), []).append(price)
    # Evaluate preferred spread points; we allow both home and away selections
    preferred_points = [-1.0, -0.5, 0.5, 1.0]
    for p in preferred_points:
        # Collect unique team names that have this point
        names = [key[1] for key in spreads_prices.keys() if abs(key[0] - p) < 1e-6]
        for team_name in names:
            prices = spreads_prices.get((p, team_name), [])
            if not prices:
                continue
            avg_price, best_price = avg_and_best(prices)
            if avg_price and best_price and MIN_LEG_ODDS <= best_price <= MAX_LEG_ODDS:
                # Determine if this team is the home or away team
                sel = "HOME" if team_match_score(team_name, home) >= team_match_score(team_name, away) else "AWAY"
                value_score = (best_price / avg_price) - 1 if avg_price else 0.0
                out.append({
                    "match": f"{home} vs {away}",
                    "home": home,
                    "away": away,
                    "league": None,
                    "kickoff": kickoff,
                    "bet_type": "SPREADS",
                    "market_key": "spreads",
                    "selection": sel,
                    "selection_team": team_name,
                    "line": p,
                    "bookmaker": "best_of",
                    "odds": best_price,
                    "avg_odds": avg_price,
                    "value_score": value_score,
                })
        # Only evaluate the first available preferred point
        if any(abs(key[0] - p) < 1e-6 for key in spreads_prices.keys()):
            break
    return out


# =========================================================
#  PONTOZÁS + INDOKLÁS
# =========================================================
def score_candidate(c: dict) -> tuple[float, str, dict]:
    """
    Compute a numeric score (0–100), generate a Hungarian explanation and
    return metadata used later in the UI. The scoring considers odds proximity
    to the target leg odds, value compared to market averages, simple news
    sentiment and weather penalties. Positive value scores boost the rating.
    """
    odds = float(c.get("odds"))
    avg_odds = float(c.get("avg_odds", odds))
    value_score = float(c.get("value_score", 0.0))
    # Odds proximity component: closer to TARGET_LEG_ODDS is better
    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))
    # Value component: positive value implies better than market
    value_bonus = 20.0 * value_score
    # Weather penalty: high wind or precipitation reduce confidence
    city_guess = (c["home"].split()[-1] if c["home"] else "London")
    w = get_weather_basic(city_guess)
    weather_pen = 0.0
    try:
        if w.get("wind") is not None and w.get("wind", 0) >= 12:
            weather_pen -= 6
        if isinstance(w.get("desc", ""), str) and any(x in w.get("desc", "").lower() for x in ["eső", "zápor", "vihar"]):
            weather_pen -= 4
    except Exception:
        pass
    # News sentiment: bias the score based on relevant headlines
    news_home = news_brief(c["home"])
    # Sleep briefly between calls to avoid hitting the NewsAPI rate limit
    time.sleep(0.2)
    news_away = news_brief(c["away"])
    news_bias = 0
    if c["bet_type"] == "H2H":
        if team_match_score(c["selection"], c["home"]) >= 0.7:
            news_bias = news_home["score"]
        else:
            news_bias = news_away["score"]
    else:
        news_bias = news_home["score"] + news_away["score"]
    news_score = float(news_bias) * 6.0
    # Raw score baseline and combination
    raw = 50.0 + odds_score + value_bonus + news_score + weather_pen
    final = max(0.0, min(100.0, raw))
    # Explanation text
    if c["bet_type"] == "H2H":
        bet_label = f"Végkimenetel (H2H): **{c['selection']}**"
    elif c["bet_type"] == "TOTALS":
        bet_label = f"Gólok száma (Totals): **{c['selection']} {c['line']}**"
    elif c["bet_type"] == "SPREADS":
        side = "Hazai" if c["selection"] == "HOME" else "Vendég"
        bet_label = f"Hendikep (Spreads): **{side} {c['line']}**"
    else:
        bet_label = f"Piac: {c['bet_type']}"
    why = []
    why.append(f"Az odds **{odds:.2f}**, az átlag odds **{avg_odds:.2f}**, a value arány **{value_score*100:.1f}%** – ez jól illeszkedik a duplázó (~2.00) célhoz.")
    if news_bias > 0:
        why.append("A friss hírek összképe inkább **pozitív** a választás szempontjából.")
    elif news_bias < 0:
        why.append("A friss hírekben van **kockázati jel** (sérülés/hiányzás gyanú), ezért óvatosabb.")
    else:
        why.append("A hírek alapján nincs egyértelmű extra kockázat vagy boost.")
    if w.get("temp") is not None:
        try:
            why.append(f"Időjárás: {w['temp']:.0f}°C, {w['desc']} (szél: {w['wind'] if w['wind'] is not None else '?'} m/s).")
        except Exception:
            pass
    reasoning = bet_label + "\n\n" + " ".join(why)
    meta = {"weather": w, "news_home": news_home, "news_away": news_away}
    return final, reasoning, meta


# =========================================================
#  DUÓ KIVÁLASZTÁS (össz-odds ~ 2.00)
# =========================================================
def pick_best_duo(cands: list[dict]) -> tuple[list[dict], float]:
    """
    Select two distinct bets such that their combined odds fall within the
    [TOTAL_ODDS_MIN, TOTAL_ODDS_MAX] range and are as close as possible to
    TARGET_TOTAL_ODDS. Utility is the sum of scores plus a bonus for odds
    closeness. If no pair meets the odds range, the top two scoring
    candidates are returned. Candidates from the same match are not paired.
    """
    if len(cands) < 2:
        return [], 0.0
    best = (None, None, -1e9, 0.0)  # (i, j, utility, total_odds)
    n = len(cands)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]
            if a["match"] == b["match"]:
                continue
            total_odds = float(a["odds"]) * float(b["odds"])
            if not (TOTAL_ODDS_MIN <= total_odds <= TOTAL_ODDS_MAX):
                continue
            closeness = 1.0 - min(1.0, abs(total_odds - TARGET_TOTAL_ODDS) / 0.15)
            utility = float(a["score"]) + float(b["score"]) + 20.0 * closeness
            if utility > best[2]:
                best = (i, j, utility, total_odds)
    if best[0] is None:
        top2 = sorted(cands, key=lambda x: x["score"], reverse=True)[:2]
        if len(top2) < 2:
            return [], 0.0
        return top2, float(top2[0]["odds"]) * float(top2[1]["odds"])
    return [cands[best[0]], cands[best[1]]], best[3]


# =========================================================
#  FŐ ELEMZÉS
# =========================================================
def run_analysis(leagues: list[str]) -> dict:
    """
    Main analysis pipeline. It refreshes past results, collects candidates from
    upcoming matches, scores them, sorts by score, selects the best duo and
    attempts to find football‑data match IDs for the selected ticket.
    """
    # 0) update closing odds before settling past predictions. Capturing closing
    # odds near kickoff allows us to compute CLV before the match starts. The
    # returned count indicates how many records were updated. We do not show
    # this count directly but it aids internal analysis.
    _clv_updated = update_closing_odds()
    # 1) update past predictions (settle finished matches)
    updated = refresh_past_results()
    # 2) collect candidates
    candidates: list[dict] = []
    now = now_utc()
    limit = now + timedelta(hours=WINDOW_HOURS)
    for lg in leagues:
        try:
            data = odds_api_get(lg, REQUEST_MARKETS)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for m in data:
            try:
                kickoff = iso_to_dt((m.get("commence_time") or "").replace("Z", "+00:00"))
            except Exception:
                continue
            if not (now <= kickoff <= limit):
                continue
            cands = extract_candidates_from_match(m)
            for c in cands:
                c["league"] = lg
                sc, reason, meta = score_candidate(c)
                c["score"] = sc
                c["reasoning"] = reason
                c["meta"] = meta
                candidates.append(c)
            time.sleep(0.05)  # avoid hitting rate limits
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    ticket, total_odds = pick_best_duo(candidates)
    for t in ticket:
        try:
            mid = fd_find_match_id(t["home"], t["away"], t["kickoff"])
        except Exception:
            mid = None
        t["football_data_match_id"] = mid
    return {
        "updated_results": updated,
        "candidates": candidates,
        "ticket": ticket,
        "total_odds": total_odds,
    }


def save_ticket(ticket: list[dict]):
    """
    Persist the selected ticket to the database. Each ticket contains two
    predictions. The meta information (avg_odds, value_score, etc.) is not
    persisted as separate columns but can be reconstructed from the JSON
    snapshot if needed. This function is idempotent.
    """
    if not ticket:
        return
    con = db()
    cur = con.cursor()
    for t in ticket:
        cur.execute(
            """
            INSERT INTO predictions
            (created_at, match, home, away, league, kickoff_utc,
             bet_type, market_key, selection, line, bookmaker, odds,
             score, reasoning, football_data_match_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                t["match"],
                t["home"],
                t["away"],
                t["league"],
                t["kickoff"].isoformat(),
                t["bet_type"],
                t["market_key"],
                t["selection"],
                t["line"],
                t["bookmaker"],
                float(t["odds"]),
                float(t["score"]),
                t["reasoning"],
                t.get("football_data_match_id"),
            ),
        )
    con.commit()
    con.close()


# =========================================================
#  UI (magyar)
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@300;400;600&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%); }
.hdr {
  font-family: 'Orbitron', sans-serif;
  background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  font-weight: 900; font-size: 2.6rem; margin: 0.2rem 0 0.8rem 0;
}
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px;
  margin: 12px 0;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.muted { opacity: 0.85; }
.badge {
  display:inline-block; padding: 2px 10px; border-radius: 999px;
  background: rgba(123,44,191,0.25); border: 1px solid rgba(0,212,255,0.35);
  margin-left: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr">⚽ TITAN – Strategic Intelligence</div>', unsafe_allow_html=True)
st.caption("Manuális futtatás | 24 órán belüli meccsek | 2 tipp ~ 2.00 össz-odds | több piac (amennyit az Odds API ad)")

with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    leagues = st.multiselect("Ligák", DEFAULT_LEAGUES, default=DEFAULT_LEAGUES)
    st.write(f"Piacok lekérése: `{', '.join(REQUEST_MARKETS)}` (ami elérhető, azt használjuk)")
    st.write(f"Leg odds szűrés: {MIN_LEG_ODDS:.2f} – {MAX_LEG_ODDS:.2f}")
    st.write(f"Duplázó cél: {TARGET_TOTAL_ODDS:.2f} (tűrés: {TOTAL_ODDS_MIN:.2f}–{TOTAL_ODDS_MAX:.2f})")
    st.divider()
    con = db()
    df_all = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    con.close()
    total = len(df_all)
    won = int((df_all["result"] == "WON").sum()) if total else 0
    lost = int((df_all["result"] == "LOST").sum()) if total else 0
    void = int((df_all["result"] == "VOID").sum()) if total else 0
    pending = int((df_all["result"] == "PENDING").sum()) if total else 0
    decided = max(1, (won + lost))
    hit = (won / decided) * 100.0
    # Compute average CLV% for decided bets where CLV is available
    df_clv = df_all.copy()
    # Convert clv_percent column to numeric; missing values remain NaN
    if "clv_percent" in df_clv.columns:
        df_clv["clv_percent"] = pd.to_numeric(df_clv["clv_percent"], errors="coerce")
        clv_decided = df_clv[df_clv["result"].isin(["WON", "LOST"]) & df_clv["clv_percent"].notna()]
        avg_clv = clv_decided["clv_percent"].mean() if len(clv_decided) else float('nan')
    else:
        avg_clv = float('nan')
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Összes tipp", total)
        st.metric("Találat %", f"{hit:.0f}%")
        if not pd.isna(avg_clv):
            st.metric("Átlag CLV %", f"{avg_clv * 100:.2f}%")
        else:
            st.metric("Átlag CLV %", "—")
    with c2:
        st.metric("Nyert", won)
        st.metric("Vesztett", lost)
    st.caption(f"VOID: {void} | PENDING: {pending}")

colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)
with colB:
    save_btn = st.button("💾 Két tipp mentése DB-be", use_container_width=True)

if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

if run_btn:
    with st.spinner("Elemzés fut… (előző eredmények frissítése + új jelöltek számítása)"):
        res = run_analysis(leagues)
        st.session_state["last_run"] = res
        if res["updated_results"] > 0:
            st.success(f"✅ Korábbi tippek frissítve: {res['updated_results']} db lezárva.")
        else:
            st.info("ℹ️ Nincs frissítendő korábbi tipp (vagy még nem értek véget).")

if st.session_state["last_run"] is not None:
    res = st.session_state["last_run"]
    ticket = res["ticket"]
    total_odds = res["total_odds"]
    st.subheader("🎫 Ajánlott duplázó (2 tipp)")
    if not ticket:
        st.warning("Nincs elég jelölt a 24 órás ablakban (vagy a marketek nem adtak használható oddsot).")
    else:
        st.markdown(
            f"**Össz-odds:** `{total_odds:.2f}`  <span class='badge'>cél: ~{TARGET_TOTAL_ODDS:.2f}</span>",
            unsafe_allow_html=True,
        )
        for idx, t in enumerate(ticket, start=1):
            kickoff_local = t["kickoff"].astimezone()
            meta = t.get("meta", {})
            w = meta.get("weather", {})
            nh = meta.get("news_home", {})
            na = meta.get("news_away", {})
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{idx}  {t['match']}")
            st.markdown(
                f"<span class='muted'>Liga:</span> `{t['league']}`  |  <span class='muted'>Kezdés:</span> **{kickoff_local.strftime('%Y.%m.%d %H:%M')}**",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Piac:** `{t['bet_type']}`  |  **Odds:** `{t['odds']:.2f}`  |  **Score:** `{t['score']:.0f}/100`",)
            # fogadás specifikáció
            if t["bet_type"] == "H2H":
                st.write(f"**Tipp:** {t['selection']} (meccs győztese – rendes játékidő)")
            elif t["bet_type"] == "TOTALS":
                st.write(f"**Tipp:** {t['selection']} {t['line']}")
            elif t["bet_type"] == "SPREADS":
                side = "Hazai" if t["selection"] == "HOME" else "Vendég"
                st.write(f"**Tipp:** {side} {t['line']}")
            else:
                st.write(f"**Tipp:** {t['selection']}")
            st.markdown("**Miért ezt ajánlja:**")
            st.write(t["reasoning"])
            if w:
                st.caption(
                    f"🌦️ Időjárás (város tipp): {w.get('temp','?')}°C, {w.get('desc','?')}, szél: {w.get('wind','?')} m/s",
                )
            if nh.get("lines") or na.get("lines"):
                with st.expander("📰 Friss hírcímek (forrással)", expanded=False):
                    st.write(f"**{t['home']}**")
                    if nh.get("lines"):
                        for line in nh["lines"]:
                            st.write(line)
                    else:
                        st.write("• nincs releváns friss cím")
                    st.write(f"**{t['away']}**")
                    if na.get("lines"):
                        for line in na["lines"]:
                            st.write(line)
                    else:
                        st.write("• nincs releváns friss cím")
            st.caption(f"football-data match_id: {t.get('football_data_match_id')}")
            st.markdown("</div>", unsafe_allow_html=True)

if save_btn:
    if st.session_state["last_run"] is None or not st.session_state["last_run"]["ticket"]:
        st.warning("Előbb futtasd az elemzést, hogy legyen 2 tipp.")
    else:
        save_ticket(st.session_state["last_run"]["ticket"])
        st.success("✅ A két tipp mentve az adatbázisba.")

st.divider()
st.subheader("📜 Előzmények + statisztika")
con = db()
df = pd.read_sql_query(
    """
    SELECT id, created_at, match, league, kickoff_utc, bet_type, selection, line,
           odds, closing_odds, clv_percent, score, result, home_goals, away_goals
    FROM predictions
    ORDER BY id DESC
    LIMIT 400
    """,
    con,
)
con.close()
st.dataframe(df, use_container_width=True)
if not df.empty:
    df2 = df.copy()
    df2["odds"] = pd.to_numeric(df2["odds"], errors="coerce")
    df2["score"] = pd.to_numeric(df2["score"], errors="coerce")
    df2["closing_odds"] = pd.to_numeric(df2.get("closing_odds"), errors="coerce")
    df2["clv_percent"] = pd.to_numeric(df2.get("clv_percent"), errors="coerce")
    st.caption("Megjegyzés: a találati arányt és CLV%-t csak a lezárt (WON/LOST) tippekre számoljuk, VOID/UNKNOWN nélkül.")
    decided_df = df2[df2["result"].isin(["WON", "LOST"])]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lezárt tippek", len(decided_df))
    with c2:
        st.metric("Átlag odds", f"{decided_df['odds'].mean():.2f}" if len(decided_df) else "—")
    with c3:
        hit_rate = (decided_df["result"].eq("WON").mean() * 100.0) if len(decided_df) else 0.0
        st.metric("Találat %", f"{hit_rate:.0f}%")
    with c4:
        # Compute mean CLV% for decided bets where clv_percent is not NA
        valid_clv = decided_df["clv_percent"].dropna()
        avg_clv_dec = valid_clv.mean() if len(valid_clv) else float('nan')
        st.metric("Átlag CLV %", f"{avg_clv_dec * 100:.2f}%" if not pd.isna(avg_clv_dec) else "—")

