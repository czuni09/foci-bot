import os
import re
import time
import json
import sqlite3
from math import sqrt
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st


# =========================================================
#  TITAN – Strategic Intelligence (LIVE odds + fallback MODEL)
#  - LIVE: TheOddsAPI (ha van kvóta/kulcs)
#  - FALLBACK: football-data fixtures + egyszerű modell (nem bukméker odds!)
#  - NINCS "kamu meccs": csak football-data vagy DB cache fixtures
# =========================================================

st.set_page_config(page_title="⚽ TITAN – Strategic Intelligence", layout="wide", page_icon="⚽")

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")

DB_PATH = "titan.db"

# Ticket cél
TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.90
TOTAL_ODDS_MAX = 2.10
TARGET_LEG_ODDS = sqrt(2)  # ~1.414

# Odds markets
REQUEST_MARKETS = ["h2h", "totals", "spreads"]

# Odds API ligák
DEFAULT_LEAGUES = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
]

# Football-data versenyek (fallback fixtures / standings modellhez)
# (Ezek stabil ID-k a football-data API-ban.)
FD_COMPETITIONS = {
    "soccer_epl": 2021,                 # Premier League
    "soccer_spain_la_liga": 2014,       # La Liga
    "soccer_germany_bundesliga": 2002,  # Bundesliga
    "soccer_italy_serie_a": 2019,       # Serie A
    "soccer_france_ligue_one": 2015,    # Ligue 1
    "soccer_uefa_champs_league": 2001,  # UCL
    "soccer_uefa_europa_league": 2146,  # UEL (football-data-nál változhat, de általában ez)
}


# =========================================================
#  DB
# =========================================================
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return con


def init_db():
    con = db()
    cur = con.cursor()

    cur.execute(
        """
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
            away_goals INTEGER,

            closing_odds REAL,
            clv_percent REAL,

            data_quality TEXT
        )
        """
    )

    # fixtures cache: valódi meccsek eltárolása UI-hoz akkor is, ha épp nincs API
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fixtures_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT,
            league TEXT,
            fd_match_id INTEGER,
            kickoff_utc TEXT,
            home TEXT,
            away TEXT,
            status TEXT,
            payload_json TEXT
        )
        """
    )

    con.commit()
    con.close()


init_db()


# =========================================================
#  SEGÉD
# =========================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_to_dt(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def norm_team(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9áéíóöőúüű\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    repl = {
        "manchester utd": "manchester united",
        "man utd": "manchester united",
        "bayern munchen": "bayern münchen",
        "bayern munich": "bayern münchen",
        "internazionale": "inter",
        "psg": "paris saint germain",
        "athletic bilbao": "athletic club",
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


def sigmoid(x: float) -> float:
    # stabil
    if x >= 0:
        z = pow(2.718281828, -x)
        return 1 / (1 + z)
    z = pow(2.718281828, x)
    return z / (1 + z)


# =========================================================
#  football-data.org (fixtures + standings modell)
# =========================================================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_KEY} if FOOTBALL_DATA_KEY else {}


@st.cache_data(ttl=900)
def fd_get(url, params=None, timeout=20):
    if not FOOTBALL_DATA_KEY:
        raise RuntimeError("Missing FOOTBALL_DATA_KEY")
    r = requests.get(url, headers=fd_headers(), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def cache_fixtures(league_key: str, matches: list[dict]):
    con = db()
    cur = con.cursor()
    fetched_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for m in matches:
        mid = (m.get("id") or None)
        kickoff = (m.get("utcDate") or None)
        home = ((m.get("homeTeam") or {}).get("name") or "")
        away = ((m.get("awayTeam") or {}).get("name") or "")
        status = (m.get("status") or "")

        cur.execute(
            """
            INSERT INTO fixtures_cache (fetched_at, league, fd_match_id, kickoff_utc, home, away, status, payload_json)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                fetched_at,
                league_key,
                int(mid) if isinstance(mid, int) else None,
                kickoff,
                home,
                away,
                status,
                json.dumps(m, ensure_ascii=False),
            )
        )
    con.commit()
    con.close()


def load_cached_fixtures(leagues: list[str], date_from: str, date_to: str) -> list[dict]:
    # fallback ha nincs FD kulcs / időszakos hiba: a DB-ből töltjük a legutóbbiakat
    con = db()
    q = """
        SELECT payload_json
        FROM fixtures_cache
        WHERE league IN ({})
          AND kickoff_utc >= ?
          AND kickoff_utc <= ?
        ORDER BY kickoff_utc ASC
        LIMIT 300
    """.format(",".join(["?"] * len(leagues)))
    params = list(leagues) + [date_from, date_to]
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    out = []
    for s in df["payload_json"].tolist():
        try:
            out.append(json.loads(s))
        except Exception:
            pass
    return out


@st.cache_data(ttl=900)
def fd_upcoming_matches_for_comp(comp_id: int, date_from: str, date_to: str) -> list[dict]:
    url = f"https://api.football-data.org/v4/competitions/{comp_id}/matches"
    js = fd_get(url, params={"dateFrom": date_from, "dateTo": date_to})
    matches = js.get("matches", []) or []
    # csak scheduled/timed
    out = []
    for m in matches:
        stt = (m.get("status") or "")
        if stt in ("SCHEDULED", "TIMED"):
            out.append(m)
    return out


@st.cache_data(ttl=6 * 3600)
def fd_strengths_for_comp(comp_id: int) -> dict:
    """
    Egyszerű, gyors "erő" modell standingsből.
    Nem fogadásra "garancia", csak fallback minőség (MODEL).
    """
    url = f"https://api.football-data.org/v4/competitions/{comp_id}/standings"
    js = fd_get(url)
    standings = js.get("standings", []) or []
    table = None
    for s in standings:
        if (s.get("type") or "") == "TOTAL":
            table = s.get("table", [])
            break
    if not table:
        # ha nincs TOTAL, vegyük az elsőt
        table = (standings[0].get("table") if standings else []) or []

    strengths = {}
    for row in table:
        team = ((row.get("team") or {}).get("name") or "")
        played = safe_float(row.get("playedGames"), 0) or 0
        points = safe_float(row.get("points"), 0) or 0
        gd = safe_float(row.get("goalDifference"), 0) or 0

        if played <= 0:
            continue

        ppg = points / played
        gdpg = gd / played
        # egyszerű rating (skálázott): ppg dominál, gd finomít
        rating = (ppg * 1.0) + (gdpg * 0.15)
        strengths[norm_team(team)] = float(rating)

    return strengths


def build_real_fixtures(leagues: list[str], window_hours: int, debug_rows: list) -> list[dict]:
    """
    Valódi upcoming meccsek:
    - elsődlegesen football-data API-ból
    - ha nincs kulcs / hiba: DB fixtures_cache-ből
    """
    now = now_utc()
    end = now + timedelta(hours=int(window_hours))
    date_from = now.date().isoformat()
    date_to = end.date().isoformat()

    fixtures = []
    used_cache = False

    if FOOTBALL_DATA_KEY:
        for lg in leagues:
            comp_id = FD_COMPETITIONS.get(lg)
            if not comp_id:
                debug_rows.append((lg, "FD: nincs comp mapping", 0, ""))
                continue
            try:
                ms = fd_upcoming_matches_for_comp(comp_id, date_from, date_to)
                debug_rows.append((lg, "FD: OK", len(ms), f"comp_id={comp_id}"))
                # cacheljük DB-be (hogy később kulcs nélkül is legyen valós match UI-hoz)
                try:
                    cache_fixtures(lg, ms)
                except Exception:
                    pass
                for m in ms:
                    kickoff = iso_to_dt(m.get("utcDate"))
                    if not kickoff:
                        continue
                    if now <= kickoff <= end:
                        fixtures.append({"league": lg, "comp_id": comp_id, "fd": m})
            except Exception as e:
                debug_rows.append((lg, "FD: HIBA", 0, str(e)[:200]))
    else:
        debug_rows.append(("_GLOBAL_", "FD: nincs FOOTBALL_DATA_KEY", 0, ""))

    if not fixtures:
        # DB cache fallback
        used_cache = True
        cached = load_cached_fixtures(leagues, f"{date_from}T00:00:00Z", f"{date_to}T23:59:59Z")
        for m in cached:
            # próbáljuk league kulcsot a cache rekordból? payloadban nincs.
            # Itt csak "valós meccs" kell UI-hoz: a league-t a kiválasztott ligák közül körbeosztjuk.
            kickoff = iso_to_dt(m.get("utcDate"))
            if not kickoff:
                continue
            if now <= kickoff <= end:
                fixtures.append({"league": leagues[0] if leagues else "soccer_epl", "comp_id": None, "fd": m})

    return fixtures, used_cache


# =========================================================
#  Weather + News (opcionális)
# =========================================================
@st.cache_data(ttl=900)
def get_weather_basic(city_guess="London"):
    if not WEATHER_KEY:
        return {"temp": None, "desc": "nincs adat", "wind": None}
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_guess, "appid": WEATHER_KEY, "units": "metric", "lang": "hu"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "temp": safe_float((data.get("main") or {}).get("temp")),
            "desc": ((data.get("weather") or [{}])[0] or {}).get("description", "ismeretlen"),
            "wind": safe_float((data.get("wind") or {}).get("speed")),
        }
    except Exception:
        return {"temp": None, "desc": "ismeretlen", "wind": None}


@st.cache_data(ttl=900)
def news_brief(team_name: str):
    if not NEWS_API_KEY:
        return {"score": 0, "lines": []}
    try:
        url = "https://newsapi.org/v2/everything"
        q = f'"{team_name}" (injury OR injured OR out OR doubt OR suspended OR return OR fit OR lineup)'
        params = {"q": q, "language": "en", "sortBy": "publishedAt", "pageSize": 3, "apiKey": NEWS_API_KEY}
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return {"score": 0, "lines": []}
        js = r.json()
        arts = js.get("articles", []) or []
        if not arts:
            return {"score": 0, "lines": []}

        lines, score = [], 0
        for a in arts[:2]:
            title = (a.get("title") or "").strip()
            src = (a.get("source") or {}).get("name", "ismeretlen")
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
#  Odds API (LIVE)
# =========================================================
@st.cache_data(ttl=180)
def odds_api_get(league_key: str, markets: list[str], regions: str = "eu"):
    if not ODDS_API_KEY:
        raise requests.exceptions.HTTPError("Missing ODDS_API_KEY")
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def best_price_across_books(event: dict, market_key: str, selection_name: str, line: float | None = None):
    """
    Kiszedi a best price-t az összes bookmakerből egy adott selectionre.
    Visszaad: best_odds, avg_odds
    """
    prices = []
    for b in event.get("bookmakers", []) or []:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != market_key:
                continue
            for o in mk.get("outcomes", []) or []:
                nm = (o.get("name") or "")
                pt = safe_float(o.get("point"))
                pr = safe_float(o.get("price"))
                if pr is None:
                    continue
                if market_key == "h2h":
                    if team_match_score(nm, selection_name) >= 0.92:
                        prices.append(pr)
                elif market_key == "totals":
                    if nm.lower() == selection_name.lower() and line is not None and pt is not None and abs(pt - float(line)) < 1e-6:
                        prices.append(pr)
                elif market_key == "spreads":
                    if line is not None and pt is not None and abs(pt - float(line)) < 1e-6:
                        # spreadsnél nm = team név
                        if team_match_score(nm, selection_name) >= 0.92:
                            prices.append(pr)
    if not prices:
        return None, None
    avg_o = sum(prices) / len(prices)
    best_o = max(prices)
    return float(best_o), float(avg_o)


def find_matching_odds_event(events: list[dict], home: str, away: str):
    """
    Odds API event matching a football-data fixturehez (csapatnév fuzzy).
    """
    best = (0.0, None)
    for ev in events:
        h = ev.get("home_team") or ""
        a = ev.get("away_team") or ""
        sc = (team_match_score(home, h) + team_match_score(away, a)) / 2.0
        if sc > best[0]:
            best = (sc, ev)
    return best[1] if best[0] >= 0.65 else None


# =========================================================
#  Candidate építés: LIVE vagy MODEL
# =========================================================
def model_candidates_from_fixture(league_key: str, comp_id: int | None, fd_match: dict, min_odds: float, max_odds: float):
    """
    Fallback MODEL jelöltek (nem bukméker odds):
    - home win "fair" odds standings-alapú erőből
    - over 1.5 / under 4.5 pseudo-odds várható gól alapján (egyszerű)
    """
    out = []
    kickoff = iso_to_dt(fd_match.get("utcDate"))
    home = ((fd_match.get("homeTeam") or {}).get("name") or "")
    away = ((fd_match.get("awayTeam") or {}).get("name") or "")
    mid = fd_match.get("id")

    if not kickoff or not home or not away:
        return out

    strengths = {}
    if FOOTBALL_DATA_KEY and comp_id:
        try:
            strengths = fd_strengths_for_comp(comp_id)
        except Exception:
            strengths = {}

    sh = strengths.get(norm_team(home), 1.4)
    sa = strengths.get(norm_team(away), 1.4)

    # home win p
    home_adv = 0.18
    p_home = sigmoid((sh - sa) + home_adv)  # ~0.35-0.70 tipikusan
    fair_home_odds = max(1.20, min(3.50, 1.0 / max(0.05, p_home)))

    # expected goals (nagyon egyszerű: erők + baseline)
    # több erő különbség -> több gól a favorittól
    eg = 2.55 + (sh - sa) * 0.35
    eg = max(1.6, min(3.6, eg))

    # pseudo odds: minél magasabb eg, annál inkább over 1.5
    # (nem tudományos, de UI-hoz és jelzéshez ok)
    over15_odds = max(1.20, min(1.85, 1.85 - (eg - 2.0) * 0.35))
    under45_odds = max(1.15, min(1.70, 1.45 + (eg - 2.5) * 0.10))

    # szűrés
    def add_if(ok_odds, bet_type, market_key, selection, line, note):
        if ok_odds is None:
            return
        if not (min_odds <= ok_odds <= max_odds):
            return
        out.append({
            "match": f"{home} vs {away}",
            "home": home,
            "away": away,
            "league": league_key,
            "kickoff": kickoff,
            "bet_type": bet_type,
            "market_key": market_key,
            "selection": selection,
            "line": line,
            "bookmaker": "MODEL",
            "odds": float(ok_odds),
            "avg_odds": float(ok_odds),
            "value_score": 0.0,
            "data_quality": f"MODEL (NEM ODDS) – {note}",
            "football_data_match_id": int(mid) if isinstance(mid, int) else None
        })

    add_if(fair_home_odds, "H2H", "h2h", home, None, "standings-alapú fair odds")
    add_if(over15_odds, "TOTALS", "totals", "Over", 1.5, "egyszerű várható gól modell")
    add_if(under45_odds, "TOTALS", "totals", "Under", 4.5, "egyszerű várható gól modell")

    return out


def live_candidates_from_odds(league_key: str, odds_events: list[dict], fd_match: dict, min_odds: float, max_odds: float):
    """
    LIVE jelöltek Odds API-ból a football-data fixturehez illesztve.
    - H2H: favorit (legalacsonyabb best odds a két csapat között) -> valóságban inkább min avg, de itt ok
    - Totals: 2.5 preferálva, majd 3.5, 1.5
    - Spreads: -0.5 / +0.5 / -1 / +1
    """
    out = []
    kickoff = iso_to_dt(fd_match.get("utcDate"))
    home = ((fd_match.get("homeTeam") or {}).get("name") or "")
    away = ((fd_match.get("awayTeam") or {}).get("name") or "")
    mid = fd_match.get("id")

    if not kickoff or not home or not away:
        return out

    ev = find_matching_odds_event(odds_events, home, away)
    if not ev:
        return out

    # H2H: best price csapatra, és választjuk azt, ami "közelebb" van a target leg odds-hoz
    bh, ah = best_price_across_books(ev, "h2h", home), best_price_across_books(ev, "h2h", away)
    if bh[0] and ah[0]:
        # válasszunk olyat, ami odds tartományban van és értelmes
        candidates = [(home, bh[0], bh[1]), (away, ah[0], ah[1])]
        candidates = [c for c in candidates if min_odds <= c[1] <= max_odds]
        if candidates:
            # prefer: closer to TARGET_LEG_ODDS
            pick = min(candidates, key=lambda x: abs(x[1] - TARGET_LEG_ODDS))
            sel, best_o, avg_o = pick
            value = (best_o / avg_o) - 1.0 if (avg_o and avg_o > 0) else 0.0
            out.append({
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": league_key,
                "kickoff": kickoff,
                "bet_type": "H2H",
                "market_key": "h2h",
                "selection": sel,
                "line": None,
                "bookmaker": "best_of",
                "odds": float(best_o),
                "avg_odds": float(avg_o) if avg_o else float(best_o),
                "value_score": float(value),
                "data_quality": "LIVE",
                "football_data_match_id": int(mid) if isinstance(mid, int) else None
            })

    # Totals: preferált line-ok
    for line in (2.5, 3.5, 1.5):
        bo_over, ao_over = best_price_across_books(ev, "totals", "Over", line=line)
        bo_under, ao_under = best_price_across_books(ev, "totals", "Under", line=line)
        added_any = False
        if bo_over and min_odds <= bo_over <= max_odds:
            value = (bo_over / ao_over) - 1.0 if ao_over else 0.0
            out.append({
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": league_key,
                "kickoff": kickoff,
                "bet_type": "TOTALS",
                "market_key": "totals",
                "selection": "Over",
                "line": float(line),
                "bookmaker": "best_of",
                "odds": float(bo_over),
                "avg_odds": float(ao_over) if ao_over else float(bo_over),
                "value_score": float(value),
                "data_quality": "LIVE",
                "football_data_match_id": int(mid) if isinstance(mid, int) else None
            })
            added_any = True
        if bo_under and min_odds <= bo_under <= max_odds:
            value = (bo_under / ao_under) - 1.0 if ao_under else 0.0
            out.append({
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": league_key,
                "kickoff": kickoff,
                "bet_type": "TOTALS",
                "market_key": "totals",
                "selection": "Under",
                "line": float(line),
                "bookmaker": "best_of",
                "odds": float(bo_under),
                "avg_odds": float(ao_under) if ao_under else float(bo_under),
                "value_score": float(value),
                "data_quality": "LIVE",
                "football_data_match_id": int(mid) if isinstance(mid, int) else None
            })
            added_any = True
        if added_any:
            break

    # Spreads: ha elérhető, próbáljuk -0.5 / +0.5 / -1 / +1
    for hcap in (-0.5, 0.5, -1.0, 1.0):
        # spreads marketben selection_name = team név
        bo_h, ao_h = best_price_across_books(ev, "spreads", home, line=hcap)
        bo_a, ao_a = best_price_across_books(ev, "spreads", away, line=hcap)
        any_added = False

        if bo_h and min_odds <= bo_h <= max_odds:
            value = (bo_h / ao_h) - 1.0 if ao_h else 0.0
            out.append({
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": league_key,
                "kickoff": kickoff,
                "bet_type": "SPREADS",
                "market_key": "spreads",
                "selection": "HOME",
                "line": float(hcap),
                "bookmaker": "best_of",
                "odds": float(bo_h),
                "avg_odds": float(ao_h) if ao_h else float(bo_h),
                "value_score": float(value),
                "data_quality": "LIVE",
                "football_data_match_id": int(mid) if isinstance(mid, int) else None
            })
            any_added = True

        if bo_a and min_odds <= bo_a <= max_odds:
            value = (bo_a / ao_a) - 1.0 if ao_a else 0.0
            out.append({
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": league_key,
                "kickoff": kickoff,
                "bet_type": "SPREADS",
                "market_key": "spreads",
                "selection": "AWAY",
                "line": float(hcap),
                "bookmaker": "best_of",
                "odds": float(bo_a),
                "avg_odds": float(ao_a) if ao_a else float(bo_a),
                "value_score": float(value),
                "data_quality": "LIVE",
                "football_data_match_id": int(mid) if isinstance(mid, int) else None
            })
            any_added = True

        if any_added:
            break

    return out


# =========================================================
#  PONTOZÁS + INDOKLÁS
# =========================================================
def score_candidate(c: dict) -> tuple[float, str, dict]:
    odds = safe_float(c.get("odds"), 0.0) or 0.0
    avg_odds = safe_float(c.get("avg_odds"), odds) or odds
    value_score = safe_float(c.get("value_score"), 0.0) or 0.0
    dq = str(c.get("data_quality", "LIVE"))

    # odds closeness
    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))

    # value bonus (LIVE-nál)
    value_bonus = 20.0 * value_score if "LIVE" in dq else 0.0

    # MODEL büntetés (nem bukméker)
    model_pen = -18.0 if dq.startswith("MODEL") else 0.0

    # weather/news
    city_guess = (c.get("home", "London").split()[-1] if c.get("home") else "London")
    w = get_weather_basic(city_guess)
    weather_pen = 0.0
    if w.get("wind") is not None and w["wind"] >= 12:
        weather_pen -= 5
    if isinstance(w.get("desc"), str) and any(x in w["desc"].lower() for x in ["eső", "zápor", "vihar"]):
        weather_pen -= 3

    news_home = news_brief(c.get("home", ""))
    time.sleep(0.05)
    news_away = news_brief(c.get("away", ""))

    news_bias = 0
    if c.get("bet_type") == "H2H":
        if team_match_score(c.get("selection", ""), c.get("home", "")) >= 0.7:
            news_bias = news_home.get("score", 0)
        else:
            news_bias = news_away.get("score", 0)
    else:
        news_bias = (news_home.get("score", 0) + news_away.get("score", 0))

    news_score = float(news_bias) * 6.0

    raw = 52.0 + odds_score + value_bonus + news_score + weather_pen + model_pen
    final = max(0.0, min(100.0, raw))

    # label
    bet_type = c.get("bet_type")
    if bet_type == "H2H":
        bet_label = f"Végkimenetel: **{c.get('selection')}**"
    elif bet_type == "TOTALS":
        bet_label = f"Gólok száma: **{c.get('selection')} {c.get('line')}**"
    elif bet_type == "SPREADS":
        side = "Hazai" if c.get("selection") == "HOME" else "Vendég"
        bet_label = f"Hendikep: **{side} {c.get('line')}**"
    else:
        bet_label = f"Piac: {bet_type}"

    why = []
    if dq.startswith("MODEL"):
        why.append("⚠️ **MODEL tipp (nem bukméker odds)** – akkor fut, ha nincs élő Odds API adat. Óvatosan kezeld.")
    why.append(f"Odds: **{odds:.2f}** (átlag: {avg_odds:.2f}, value: {value_score*100:.1f}%).")
    if news_bias > 0:
        why.append("Hírek: inkább **pozitív**.")
    elif news_bias < 0:
        why.append("Hírek: van **kockázati jel** (sérülés/hiányzás).")
    else:
        why.append("Hírek: nincs erős extra jel.")
    if w.get("temp") is not None:
        why.append(f"Időjárás: {w['temp']:.0f}°C, {w.get('desc','?')} (szél: {w.get('wind','?')} m/s).")

    reasoning = bet_label + "\n\n" + " ".join(why)
    meta = {"weather": w, "news_home": news_home, "news_away": news_away}
    return final, reasoning, meta


# =========================================================
#  DUÓ KIVÁLASZTÁS
# =========================================================
def pick_best_duo(cands: list[dict]) -> tuple[list[dict], float, str]:
    """
    1) Első kör: total_odds a [min,max] sávban
    2) Ha nincs: második kör: a két legjobb score külön meccsből (nem garantált 2.00)
    """
    if len(cands) < 2:
        return [], 0.0, "NINCS_ELEG"

    best = (None, None, -1e18, 0.0)  # i, j, utility, total_odds
    n = len(cands)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]
            if a.get("match") == b.get("match"):
                continue
            total_odds = float(a.get("odds", 0.0)) * float(b.get("odds", 0.0))
            if not (TOTAL_ODDS_MIN <= total_odds <= TOTAL_ODDS_MAX):
                continue
            closeness = 1.0 - min(1.0, abs(total_odds - TARGET_TOTAL_ODDS) / 0.15)
            utility = float(a.get("score", 0.0)) + float(b.get("score", 0.0)) + 22.0 * closeness
            if utility > best[2]:
                best = (i, j, utility, total_odds)

    if best[0] is not None:
        return [cands[best[0]], cands[best[1]]], best[3], "OK_2_00"

    # fallback: top2 külön meccsből
    sorted_c = sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)
    pick = []
    seen = set()
    for c in sorted_c:
        if c.get("match") in seen:
            continue
        pick.append(c)
        seen.add(c.get("match"))
        if len(pick) == 2:
            break
    if len(pick) < 2:
        return [], 0.0, "NINCS_ELEG"

    total_odds = float(pick[0].get("odds", 0.0)) * float(pick[1].get("odds", 0.0))
    return pick, total_odds, "FALLBACK_TOP2"


# =========================================================
#  Mentés
# =========================================================
def save_ticket(ticket: list[dict]):
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
             score, reasoning, football_data_match_id, closing_odds, clv_percent, data_quality)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                t.get("match"),
                t.get("home"),
                t.get("away"),
                t.get("league"),
                (t.get("kickoff").isoformat() if t.get("kickoff") else None),
                t.get("bet_type"),
                t.get("market_key"),
                t.get("selection"),
                t.get("line"),
                t.get("bookmaker"),
                float(t.get("odds", 0.0)),
                float(t.get("score", 0.0)),
                t.get("reasoning"),
                t.get("football_data_match_id"),
                None,
                None,
                t.get("data_quality", "LIVE"),
            ),
        )
    con.commit()
    con.close()


# =========================================================
#  RUN ANALYSIS
# =========================================================
def run_analysis(leagues: list[str], window_hours: int, min_odds: float, max_odds: float, regions: str, debug: bool):
    candidates = []
    debug_rows = []

    # 1) valós fixtures (football-data vagy DB cache)
    fixtures, used_cache = build_real_fixtures(leagues, window_hours, debug_rows)

    # 2) Odds API lekérés ligánként (ha tudjuk)
    odds_by_league = {}
    odds_ok = True
    for lg in leagues:
        try:
            data = odds_api_get(lg, REQUEST_MARKETS, regions=regions)
            if isinstance(data, list):
                odds_by_league[lg] = data
                debug_rows.append((lg, "ODDS: OK", len(data), ""))
            else:
                odds_by_league[lg] = []
                debug_rows.append((lg, "ODDS: NEM LISTA", 0, ""))
        except Exception as e:
            odds_ok = False
            odds_by_league[lg] = []
            debug_rows.append((lg, "ODDS: HIBA", 0, str(e)[:220]))

    # 3) jelöltek építése fixture-enként:
    #    - ha van matching odds event: LIVE
    #    - különben: MODEL
    for fx in fixtures:
        lg = fx["league"]
        comp_id = fx.get("comp_id")
        m = fx["fd"]

        live_events = odds_by_league.get(lg, []) or []
        live_c = []
        if live_events:
            live_c = live_candidates_from_odds(lg, live_events, m, min_odds, max_odds)

        if live_c:
            candidates.extend(live_c)
        else:
            # fallback modell (valós meccsre)
            model_c = model_candidates_from_fixture(lg, comp_id, m, min_odds, max_odds)
            candidates.extend(model_c)

        time.sleep(0.01)

    # 4) pontozás
    scored = []
    for c in candidates:
        sc, reason, meta = score_candidate(c)
        c["score"] = sc
        c["reasoning"] = reason
        c["meta"] = meta
        scored.append(c)

    scored = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)

    # 5) ticket
    ticket, total_odds, ticket_mode = pick_best_duo(scored)

    # minőség flag
    used_model = any(str(t.get("data_quality", "")).startswith("MODEL") for t in ticket) if ticket else False

    return {
        "fixtures_count": len(fixtures),
        "used_fixtures_cache": used_cache,
        "odds_ok": odds_ok,
        "candidates": scored,
        "ticket": ticket,
        "total_odds": total_odds,
        "ticket_mode": ticket_mode,
        "used_model": used_model,
        "debug_rows": debug_rows,
    }


# =========================================================
#  MODERN UI (tabs + compact cards)
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 20% 10%, rgba(0,212,255,0.10), transparent 60%),
                  radial-gradient(1200px 600px at 80% 20%, rgba(255,0,110,0.10), transparent 60%),
                  linear-gradient(135deg, #050716 0%, #0b1030 55%, #050716 100%); }

.hdr {
  font-family: 'Orbitron', sans-serif;
  font-weight: 900;
  font-size: 2.2rem;
  margin: 0.1rem 0 0.25rem 0;
  letter-spacing: 0.5px;
  background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.subhdr { opacity: 0.85; margin-bottom: 0.8rem; }

.kpi {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  margin: 10px 0;
  box-shadow: 0 10px 30px rgba(0,0,0,0.30);
}

.badge {
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  font-size: 0.78rem;
  margin-left: 8px;
}

.badge_live { border-color: rgba(0,212,255,0.45); background: rgba(0,212,255,0.12); }
.badge_model{ border-color: rgba(255,193,7,0.55); background: rgba(255,193,7,0.14); }
.badge_warn { border-color: rgba(255,0,110,0.55); background: rgba(255,0,110,0.14); }

.small { font-size: 0.90rem; opacity: 0.92; }
.muted { opacity: 0.75; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr">⚽ TITAN – Strategic Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subhdr">LIVE odds, ha van kvóta • ha nincs, valódi meccsekre MODEL fallback • nincs kamu mérkőzés</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    DEBUG = st.toggle("🔎 Debug (státuszok)", value=True)

    leagues = st.multiselect("Ligák", DEFAULT_LEAGUES, default=DEFAULT_LEAGUES)
    window_hours = st.slider("Időablak (óra)", 12, 168, 24, 12)

    st.markdown("#### Odds szűrés")
    min_odds = st.number_input("Min odds / tipp", value=1.25, step=0.01, format="%.2f")
    max_odds = st.number_input("Max odds / tipp", value=1.95, step=0.01, format="%.2f")
    regions = st.selectbox("Odds API régió", options=["eu", "uk", "eu,uk"], index=0)

    st.divider()
    st.markdown("#### Kulcs státusz")
    st.write(f"ODDS_API_KEY: {'✅' if ODDS_API_KEY else '❌'}")
    st.write(f"FOOTBALL_DATA_KEY: {'✅' if FOOTBALL_DATA_KEY else '❌'}")
    st.write(f"WEATHER_API_KEY: {'✅' if WEATHER_KEY else '❌'}")
    st.write(f"NEWS_API_KEY: {'✅' if NEWS_API_KEY else '❌'}")


c1, c2 = st.columns([1, 1])
with c1:
    run_btn = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)
with c2:
    save_btn = st.button("💾 Két tipp mentése DB-be", use_container_width=True)

if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

if run_btn:
    with st.spinner("Elemzés fut… (fixtures + odds/model + pontozás)"):
        res = run_analysis(leagues, int(window_hours), float(min_odds), float(max_odds), regions, DEBUG)
        st.session_state["last_run"] = res

tabs = st.tabs(["🎫 Ticket", "📌 Jelöltek", "📜 Előzmények", "🧪 Debug"])

# ---------------- Ticket tab ----------------
with tabs[0]:
    res = st.session_state.get("last_run")
    if not res:
        st.info("Nyomd meg az **Elemzés indítása** gombot.")
    else:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='kpi'><div class='muted'>Valódi meccsek</div><div style='font-size:1.35rem;font-weight:700'>{res['fixtures_count']}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi'><div class='muted'>Össz jelölt</div><div style='font-size:1.35rem;font-weight:700'>{len(res['candidates'])}</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi'><div class='muted'>Ticket mód</div><div style='font-size:1.10rem;font-weight:700'>{res['ticket_mode']}</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='kpi'><div class='muted'>Össz-odds</div><div style='font-size:1.35rem;font-weight:700'>{res['total_odds']:.2f}</div></div>", unsafe_allow_html=True)

        if res["used_fixtures_cache"]:
            st.warning("⚠️ football-data nem volt elérhető: DB-ből töltöttem a legutóbbi **valódi** fixture cache-t (ha volt).")

        if not res["odds_ok"]:
            st.warning("⚠️ Odds API hiba/401/kvóta → LIVE odds helyett **MODEL** tippek futnak (valódi meccsekre).")

        ticket = res["ticket"]
        if not ticket:
            st.error("Nincs ticket. (Valószínűleg nincs football-data fixture és cache sem.)")
        else:
            badge = "<span class='badge badge_live'>LIVE</span>"
            if res["used_model"]:
                badge = "<span class='badge badge_model'>MODEL (NEM ODDS)</span> <span class='badge badge_warn'>ÓVATOSAN</span>"

            st.markdown(f"### 🎫 Ajánlott duplázó {badge}", unsafe_allow_html=True)

            for idx, t in enumerate(ticket, start=1):
                kickoff_local = t["kickoff"].astimezone() if t.get("kickoff") else None
                meta = t.get("meta", {})
                w = meta.get("weather", {})
                nh = meta.get("news_home", {})
                na = meta.get("news_away", {})

                st.markdown("<div class='card'>", unsafe_allow_html=True)

                title = f"#{idx}  {t['match']}"
                st.markdown(f"#### {title}")

                sub = f"<span class='muted'>Liga:</span> <code>{t.get('league')}</code>"
                if kickoff_local:
                    sub += f" &nbsp;•&nbsp; <span class='muted'>Kezdés:</span> <b>{kickoff_local.strftime('%Y.%m.%d %H:%M')}</b>"
                st.markdown(sub, unsafe_allow_html=True)

                st.write(
                    f"**Minőség:** `{t.get('data_quality','LIVE')}`  |  **Piac:** `{t['bet_type']}`  |  **Odds:** `{t['odds']:.2f}`  |  **Score:** `{t['score']:.0f}/100`"
                )

                if t["bet_type"] == "H2H":
                    st.write(f"**Tipp:** {t['selection']} (rendes játékidő)")
                elif t["bet_type"] == "TOTALS":
                    st.write(f"**Tipp:** {t['selection']} {t['line']}")
                elif t["bet_type"] == "SPREADS":
                    side = "Hazai" if t["selection"] == "HOME" else "Vendég"
                    st.write(f"**Tipp:** {side} {t['line']}")
                else:
                    st.write(f"**Tipp:** {t['selection']}")

                st.markdown("**Miért:**")
                st.write(t["reasoning"])

                if w:
                    st.caption(f"🌦️ Időjárás: {w.get('temp','?')}°C, {w.get('desc','?')}, szél: {w.get('wind','?')} m/s")

                if (nh.get("lines") or na.get("lines")):
                    with st.expander("📰 Friss hírcímek (forrással)", expanded=False):
                        st.write(f"**{t['home']}**")
                        for line in (nh.get("lines") or ["• nincs releváns friss cím"]):
                            st.write(line)
                        st.write(f"**{t['away']}**")
                        for line in (na.get("lines") or ["• nincs releváns friss cím"]):
                            st.write(line)

                st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Candidates tab ----------------
with tabs[1]:
    res = st.session_state.get("last_run")
    if not res:
        st.info("Előbb futtasd az elemzést.")
    else:
        dfc = pd.DataFrame(res["candidates"])
        if dfc.empty:
            st.warning("Nincs jelölt.")
        else:
            show_cols = ["league", "match", "kickoff", "bet_type", "selection", "line", "odds", "score", "data_quality"]
            for c in show_cols:
                if c not in dfc.columns:
                    dfc[c] = None
            dfc["kickoff"] = dfc["kickoff"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
            st.dataframe(dfc[show_cols].head(250), use_container_width=True)

# ---------------- History tab ----------------
with tabs[2]:
    con = db()
    df = pd.read_sql_query(
        """
        SELECT id, created_at, match, league, kickoff_utc,
               bet_type, selection, line, odds,
               score, result, data_quality
        FROM predictions
        ORDER BY id DESC
        LIMIT 400
        """,
        con,
    )
    con.close()
    st.dataframe(df, use_container_width=True)

# ---------------- Debug tab ----------------
with tabs[3]:
    res = st.session_state.get("last_run")
    if not res:
        st.info("Előbb futtasd az elemzést.")
    else:
        dbg = pd.DataFrame(res["debug_rows"], columns=["league", "status", "count", "details"])
        st.dataframe(dbg, use_container_width=True)

# Save button action (global)
if save_btn:
    res = st.session_state.get("last_run")
    if not res or not res.get("ticket"):
        st.warning("Előbb futtasd az elemzést, hogy legyen ticket.")
    else:
        save_ticket(res["ticket"])
        st.success("✅ Ticket elmentve az adatbázisba.")
