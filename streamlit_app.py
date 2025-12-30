import os
import re
import time
import sqlite3
from math import sqrt
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st


# =========================================================
#  KONFIG
# =========================================================
st.set_page_config(page_title="⚽ TITAN – Strategic Intelligence", layout="wide", page_icon="⚽")

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

# Duplázó cél (2 tipp szorzata)
TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.90
TOTAL_ODDS_MAX = 2.10
TARGET_LEG_ODDS = sqrt(2)  # ~1.414

# Szűrés
WINDOW_HOURS_DEFAULT = 24
MIN_LEG_ODDS_DEFAULT = 1.25
MAX_LEG_ODDS_DEFAULT = 1.95

REQUEST_MARKETS = ["h2h", "totals", "spreads"]

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
            clv_percent REAL
        )
        """
    )
    # Backward compatible: add columns if older db
    try:
        cur.execute("PRAGMA table_info(predictions)")
        cols = [r[1] for r in cur.fetchall()]
        if "closing_odds" not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN closing_odds REAL")
        if "clv_percent" not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN clv_percent REAL")
    except Exception:
        pass

    con.commit()
    con.close()


init_db()


# =========================================================
#  SEGÉDFÜGGVÉNYEK
# =========================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_to_dt(s: str) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


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


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


# =========================================================
#  FOOTBALL-DATA.ORG
# =========================================================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_KEY}


def fd_get(url, params=None, timeout=15):
    r = requests.get(url, headers=fd_headers(), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fd_find_match_id(home: str, away: str, kickoff_utc: datetime) -> int | None:
    if not home or not away or not kickoff_utc:
        return None

    date_from = (kickoff_utc - timedelta(days=1)).date().isoformat()
    date_to = (kickoff_utc + timedelta(days=1)).date().isoformat()

    try:
        data = fd_get("https://api.football-data.org/v4/matches", params={"dateFrom": date_from, "dateTo": date_to})
    except Exception:
        return None

    best_score = 0.0
    best_id = None

    for m in data.get("matches", []) or []:
        try:
            fd_home = m["homeTeam"]["name"]
            fd_away = m["awayTeam"]["name"]
            fd_utc = iso_to_dt(m.get("utcDate"))
        except Exception:
            continue

        if not fd_utc:
            continue
        if abs((fd_utc - kickoff_utc).total_seconds()) > 8 * 3600:
            continue

        s = (team_match_score(home, fd_home) + team_match_score(away, fd_away)) / 2.0
        if s > best_score:
            best_score = s
            best_id = m.get("id")

    return best_id if best_score >= 0.55 else None


def fd_settle_prediction(pred_row: dict) -> dict:
    match_id = pred_row.get("football_data_match_id")
    if not match_id:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    try:
        m = fd_get(f"https://api.football-data.org/v4/matches/{match_id}")
    except Exception:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    status = m.get("status", "")
    if status not in ["FINISHED", "AWARDED"]:
        return {"result": "PENDING", "home_goals": None, "away_goals": None}

    ft = (m.get("score", {}) or {}).get("fullTime", {}) or {}
    hg = ft.get("home")
    ag = ft.get("away")
    if hg is None or ag is None:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    bet_type = pred_row.get("bet_type")
    selection = pred_row.get("selection") or ""
    line = pred_row.get("line")

    if bet_type == "H2H":
        if norm_team(selection) == norm_team(pred_row.get("home")):
            res = "WON" if hg > ag else "LOST"
        elif norm_team(selection) == norm_team(pred_row.get("away")):
            res = "WON" if ag > hg else "LOST"
        else:
            res = "UNKNOWN"

    elif bet_type == "TOTALS":
        ln = safe_float(line)
        if ln is None:
            res = "UNKNOWN"
        else:
            total = hg + ag
            sel = selection.lower().strip()
            if sel == "over":
                res = "WON" if total > ln else ("VOID" if total == ln else "LOST")
            elif sel == "under":
                res = "WON" if total < ln else ("VOID" if total == ln else "LOST")
            else:
                res = "UNKNOWN"

    elif bet_type == "SPREADS":
        ln = safe_float(line)
        sel = selection.upper().strip()
        if ln is None or sel not in ("HOME", "AWAY"):
            res = "UNKNOWN"
        else:
            if sel == "HOME":
                adj = (hg + ln) - ag
            else:
                adj = (ag + ln) - hg
            res = "WON" if adj > 0 else ("VOID" if adj == 0 else "LOST")
    else:
        res = "UNKNOWN"

    return {"result": res, "home_goals": int(hg), "away_goals": int(ag)}


def refresh_past_results() -> int:
    con = db()
    df = pd.read_sql_query("SELECT * FROM predictions WHERE result IN ('PENDING','UNKNOWN')", con)
    con.close()

    if df.empty:
        return 0

    updated = 0
    now = now_utc()

    for _, row in df.iterrows():
        kickoff = iso_to_dt(row.get("kickoff_utc"))
        if not kickoff:
            continue
        if now < kickoff + timedelta(hours=2):
            continue

        settle = fd_settle_prediction(row.to_dict())
        if settle["result"] == "PENDING":
            continue

        con2 = db()
        cur2 = con2.cursor()
        cur2.execute(
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
        con2.commit()
        con2.close()
        updated += 1

    return updated


# =========================================================
#  CLOSING ODDS / CLV (meccs közelében)
# =========================================================
def update_closing_odds(league_key: str, kickoff: datetime, bet_type: str, selection: str, line: float | None, home: str, away: str) -> float | None:
    """
    Visszaadja a 'closing' oddsot (best_of), ha megtalálja a meccset és a piacot.
    Megjegyzés: a CLV-t a DB frissítés résznél számoljuk a mentett placed odds alapján.
    """
    try:
        data = odds_api_get(league_key, REQUEST_MARKETS, regions="eu")
    except Exception:
        return None
    if not isinstance(data, list):
        return None

    best_match = None
    best_s = 0.0
    for m in data:
        mh = m.get("home_team")
        ma = m.get("away_team")
        mt = iso_to_dt(m.get("commence_time"))
        if not mh or not ma or not mt:
            continue
        if abs((mt - kickoff).total_seconds()) > 3 * 3600:
            continue
        s = (team_match_score(home, mh) + team_match_score(away, ma)) / 2.0
        if s > best_s:
            best_s = s
            best_match = m

    if not best_match or best_s < 0.55:
        return None

    prices = []

    bet_type = (bet_type or "").upper().strip()
    sel = (selection or "").strip()

    if bet_type == "H2H":
        for b in best_match.get("bookmakers", []) or []:
            for mk in b.get("markets", []) or []:
                if mk.get("key") != "h2h":
                    continue
                for o in mk.get("outcomes", []) or []:
                    nm = o.get("name") or ""
                    if team_match_score(nm, sel) >= 0.7:
                        pr = safe_float(o.get("price"))
                        if pr:
                            prices.append(pr)

    elif bet_type == "TOTALS":
        ln = safe_float(line)
        if ln is None:
            return None
        for b in best_match.get("bookmakers", []) or []:
            for mk in b.get("markets", []) or []:
                if mk.get("key") != "totals":
                    continue
                for o in mk.get("outcomes", []) or []:
                    nm = (o.get("name") or "").strip().capitalize()
                    pt = safe_float(o.get("point"))
                    pr = safe_float(o.get("price"))
                    if pt is None or pr is None:
                        continue
                    if abs(pt - ln) < 1e-6 and nm == sel.capitalize():
                        prices.append(pr)

    elif bet_type == "SPREADS":
        ln = safe_float(line)
        if ln is None:
            return None
        sel_code = sel.upper()
        if sel_code not in ("HOME", "AWAY"):
            return None
        for b in best_match.get("bookmakers", []) or []:
            for mk in b.get("markets", []) or []:
                if mk.get("key") != "spreads":
                    continue
                for o in mk.get("outcomes", []) or []:
                    team_nm = o.get("name") or ""
                    pt = safe_float(o.get("point"))
                    pr = safe_float(o.get("price"))
                    if pt is None or pr is None:
                        continue
                    if abs(pt - ln) > 1e-6:
                        continue
                    is_home = team_match_score(team_nm, home) >= team_match_score(team_nm, away)
                    code = "HOME" if is_home else "AWAY"
                    if code == sel_code:
                        prices.append(pr)

    return max(prices) if prices else None


def refresh_clv_for_pending(debug=False) -> int:
    """
    Meccs közelében (kickoff-90p ... kickoff+10p) megpróbálja kitölteni closing_odds + clv_percent mezőt.
    """
    con = db()
    df = pd.read_sql_query(
        """
        SELECT * FROM predictions
        WHERE (closing_odds IS NULL OR clv_percent IS NULL)
          AND result IN ('PENDING','UNKNOWN')
        """,
        con,
    )
    con.close()

    if df.empty:
        return 0

    now = now_utc()
    updated = 0

    for _, row in df.iterrows():
        kickoff = iso_to_dt(row.get("kickoff_utc"))
        if not kickoff:
            continue

        if not (kickoff - timedelta(minutes=90) <= now <= kickoff + timedelta(minutes=10)):
            continue

        league_key = row.get("league")
        if not league_key:
            continue

        closing = update_closing_odds(
            league_key=league_key,
            kickoff=kickoff,
            bet_type=row.get("bet_type"),
            selection=row.get("selection"),
            line=row.get("line"),
            home=row.get("home"),
            away=row.get("away"),
        )

        if closing is None:
            continue

        placed = safe_float(row.get("odds"))
        clv = (closing / placed) - 1.0 if placed and placed > 0 else None

        con2 = db()
        cur2 = con2.cursor()
        cur2.execute(
            "UPDATE predictions SET closing_odds=?, clv_percent=? WHERE id=?",
            (closing, clv, int(row["id"])),
        )
        con2.commit()
        con2.close()
        updated += 1

        if debug:
            st.info(f"CLV frissítve (id={int(row['id'])}) closing={closing:.2f} placed={placed:.2f} clv={(clv*100):.2f}%")

    return updated


# =========================================================
#  Külső adatok (cache)
# =========================================================
@st.cache_data(ttl=600)
def get_weather_basic(city_guess="London"):
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


@st.cache_data(ttl=600)
def news_brief(team_name: str):
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
        r = requests.get(url, params=params, timeout=12)
        # NewsAPI gyakran 401/429 -> ne dobjunk itt streamlit app-crasht, csak adjunk 0-t
        if r.status_code != 200:
            return {"score": 0, "lines": [], "err": f"HTTP {r.status_code}"}
        js = r.json()
        arts = js.get("articles", []) or []
        if not arts:
            return {"score": 0, "lines": []}

        lines = []
        score = 0
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
#  ODDS API (NEM nyeljük el a hibát: a hívó kezeli)
# =========================================================
@st.cache_data(ttl=120)
def odds_api_get(league_key: str, markets: list[str], regions: str = "eu"):
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


# =========================================================
#  Jelöltek generálása (best_of + avg_odds + value_score)
# =========================================================
def extract_candidates_from_match(m: dict, min_odds: float, max_odds: float) -> list[dict]:
    out = []
    home = m.get("home_team")
    away = m.get("away_team")
    kickoff = iso_to_dt(m.get("commence_time"))

    if not home or not away or not kickoff:
        return out

    bookmakers = m.get("bookmakers", []) or []

    def avg_best(prices: list[float]):
        if not prices:
            return None, None
        avg = sum(prices) / len(prices)
        best = max(prices)  # bettor: higher is better
        return avg, best

    # -------- H2H --------
    h2h_prices = {}  # team -> prices
    for b in bookmakers:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "h2h":
                continue
            for o in mk.get("outcomes", []) or []:
                nm = o.get("name")
                pr = safe_float(o.get("price"))
                if nm in (home, away) and pr:
                    h2h_prices.setdefault(nm, []).append(pr)

    if h2h_prices:
        avg_map = {tm: sum(ps) / len(ps) for tm, ps in h2h_prices.items() if ps}
        if avg_map:
            fav = min(avg_map, key=avg_map.get)  # favourite = lowest avg
            avg_o, best_o = avg_best(h2h_prices.get(fav, []))
            if avg_o and best_o and min_odds <= best_o <= max_odds:
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append(
                    {
                        "match": f"{home} vs {away}",
                        "home": home,
                        "away": away,
                        "league": None,
                        "kickoff": kickoff,
                        "bet_type": "H2H",
                        "market_key": "h2h",
                        "selection": fav,
                        "line": None,
                        "bookmaker": "best_of",
                        "odds": best_o,
                        "avg_odds": avg_o,
                        "value_score": value,
                    }
                )

    # -------- TOTALS --------
    totals = {}  # (line, Over/Under) -> prices
    for b in bookmakers:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "totals":
                continue
            for o in mk.get("outcomes", []) or []:
                nm = (o.get("name") or "").strip().capitalize()
                if nm.lower() not in ("over", "under"):
                    continue
                pt = safe_float(o.get("point"))
                pr = safe_float(o.get("price"))
                if pt is None or pr is None:
                    continue
                totals.setdefault((pt, nm), []).append(pr)

    for target_line in (2.5, 3.5, 1.5):
        hit_any = False
        for nm in ("Over", "Under"):
            ps = totals.get((float(target_line), nm), [])
            if not ps:
                continue
            avg_o, best_o = avg_best(ps)
            if avg_o and best_o and min_odds <= best_o <= max_odds:
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append(
                    {
                        "match": f"{home} vs {away}",
                        "home": home,
                        "away": away,
                        "league": None,
                        "kickoff": kickoff,
                        "bet_type": "TOTALS",
                        "market_key": "totals",
                        "selection": nm,
                        "line": float(target_line),
                        "bookmaker": "best_of",
                        "odds": best_o,
                        "avg_odds": avg_o,
                        "value_score": value,
                    }
                )
                hit_any = True
        if hit_any:
            break

    # -------- SPREADS --------
    spreads = {}  # (handicap, team_name) -> prices
    for b in bookmakers:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "spreads":
                continue
            for o in mk.get("outcomes", []) or []:
                team_nm = o.get("name")
                pt = safe_float(o.get("point"))
                pr = safe_float(o.get("price"))
                if team_nm and pt is not None and pr is not None:
                    spreads.setdefault((pt, team_nm), []).append(pr)

    preferred_points = (-1.0, -0.5, 0.5, 1.0)
    for p in preferred_points:
        keys = [k for k in spreads.keys() if abs(k[0] - float(p)) < 1e-6]
        if not keys:
            continue
        for (pt, team_nm) in keys:
            ps = spreads.get((pt, team_nm), [])
            avg_o, best_o = avg_best(ps)
            if avg_o and best_o and min_odds <= best_o <= max_odds:
                sel = "HOME" if team_match_score(team_nm, home) >= team_match_score(team_nm, away) else "AWAY"
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append(
                    {
                        "match": f"{home} vs {away}",
                        "home": home,
                        "away": away,
                        "league": None,
                        "kickoff": kickoff,
                        "bet_type": "SPREADS",
                        "market_key": "spreads",
                        "selection": sel,
                        "line": float(p),
                        "bookmaker": "best_of",
                        "odds": best_o,
                        "avg_odds": avg_o,
                        "value_score": value,
                    }
                )
        break

    return out


# =========================================================
#  PONTOZÁS + INDOKLÁS
# =========================================================
def score_candidate(c: dict) -> tuple[float, str, dict]:
    odds = safe_float(c.get("odds"), 0.0)
    avg_odds = safe_float(c.get("avg_odds"), odds) or odds
    value_score = safe_float(c.get("value_score"), 0.0) or 0.0

    # odds closeness
    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))

    # value bonus (market average vs best_of)
    value_bonus = 20.0 * value_score

    # weather
    city_guess = (c.get("home", "London").split()[-1] if c.get("home") else "London")
    w = get_weather_basic(city_guess)
    weather_pen = 0.0
    if w.get("wind") is not None and w["wind"] >= 12:
        weather_pen -= 6
    if isinstance(w.get("desc"), str) and any(x in w["desc"].lower() for x in ["eső", "zápor", "vihar"]):
        weather_pen -= 4

    # news
    news_home = news_brief(c.get("home", ""))
    time.sleep(0.15)
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

    raw = 50.0 + odds_score + value_bonus + news_score + weather_pen
    final = max(0.0, min(100.0, raw))

    bet_type = c.get("bet_type")
    if bet_type == "H2H":
        bet_label = f"Végkimenetel (H2H): **{c.get('selection')}**"
    elif bet_type == "TOTALS":
        bet_label = f"Gólok száma (Totals): **{c.get('selection')} {c.get('line')}**"
    elif bet_type == "SPREADS":
        side = "Hazai" if c.get("selection") == "HOME" else "Vendég"
        bet_label = f"Hendikep (Spreads): **{side} {c.get('line')}**"
    else:
        bet_label = f"Piac: {bet_type}"

    why = []
    why.append(f"Odds: **{odds:.2f}** (átlag: {avg_odds:.2f}, value: {value_score*100:.1f}%).")
    if news_bias > 0:
        why.append("Hírek összképe: **pozitív**.")
    elif news_bias < 0:
        why.append("Hírekben van **kockázati jel** (sérülés/hiányzás).")
    else:
        why.append("Hírek alapján nincs egyértelmű extra jel.")
    if w.get("temp") is not None:
        why.append(f"Időjárás (város tipp): {w['temp']:.0f}°C, {w.get('desc','?')} (szél: {w.get('wind','?')} m/s).")

    reasoning = bet_label + "\n\n" + " ".join(why)
    meta = {"weather": w, "news_home": news_home, "news_away": news_away}
    return final, reasoning, meta


# =========================================================
#  DUÓ KIVÁLASZTÁS (2.00 körül)
# =========================================================
def pick_best_duo(cands: list[dict]) -> tuple[list[dict], float]:
    if len(cands) < 2:
        return [], 0.0

    best_i, best_j = None, None
    best_util = -1e18
    best_total = 0.0

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
            util = float(a.get("score", 0.0)) + float(b.get("score", 0.0)) + 20.0 * closeness
            if util > best_util:
                best_util = util
                best_i, best_j = i, j
                best_total = total_odds

    if best_i is None:
        top2 = sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)[:2]
        if len(top2) < 2:
            return [], 0.0
        return top2, float(top2[0]["odds"]) * float(top2[1]["odds"])

    return [cands[best_i], cands[best_j]], best_total


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
             score, reasoning, football_data_match_id, closing_odds, clv_percent)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            ),
        )
    con.commit()
    con.close()


# =========================================================
#  FŐ ELEMZÉS (NEM nyel el hibát: debug táblában kiírja)
# =========================================================
def run_analysis(leagues: list[str], window_hours: int, min_odds: float, max_odds: float, regions: str, debug: bool) -> dict:
    # 0) CLV frissítés (meccs közelében)
    clv_updated = refresh_clv_for_pending(debug=debug)

    # 1) múlt lezárása
    updated_results = refresh_past_results()

    # 2) odds jelöltek
    candidates = []
    now = now_utc()
    limit = now + timedelta(hours=int(window_hours))

    debug_rows = []
    errors = 0

    for lg in leagues:
        # Odds API call: itt direkt külön kezeljük a hibát, hogy lásd az okot
        try:
            data = odds_api_get(lg, REQUEST_MARKETS, regions=regions)
            if not isinstance(data, list):
                debug_rows.append((lg, "NEM LISTA válasz", 0, ""))
                continue
            debug_rows.append((lg, "OK", len(data), ""))
        except requests.exceptions.HTTPError as e:
            errors += 1
            resp_txt = ""
            try:
                resp_txt = (e.response.text or "")[:300] if e.response is not None else ""
            except Exception:
                resp_txt = ""
            debug_rows.append((lg, f"HTTPError {getattr(e.response,'status_code', '')}", 0, resp_txt))
            continue
        except Exception as e:
            errors += 1
            debug_rows.append((lg, "EXCEPTION", 0, str(e)[:300]))
            continue

        for m in data:
            kickoff = iso_to_dt(m.get("commence_time"))
            if not kickoff:
                continue
            if not (now <= kickoff <= limit):
                continue

            cands = extract_candidates_from_match(m, min_odds=min_odds, max_odds=max_odds)
            for c in cands:
                c["league"] = lg
                sc, reason, meta = score_candidate(c)
                c["score"] = sc
                c["reasoning"] = reason
                c["meta"] = meta
                candidates.append(c)

            time.sleep(0.03)

    candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    ticket, total_odds = pick_best_duo(candidates)

    for t in ticket:
        try:
            t["football_data_match_id"] = fd_find_match_id(t["home"], t["away"], t["kickoff"])
        except Exception:
            t["football_data_match_id"] = None

    return {
        "updated_results": updated_results,
        "clv_updated": clv_updated,
        "candidates": candidates,
        "ticket": ticket,
        "total_odds": total_odds,
        "debug_rows": debug_rows,
        "errors": errors,
    }


# =========================================================
#  UI
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
st.caption("Manuális futtatás | meccsek X órán belül | 2 tipp ~2.00 össz-odds | h2h/totals/spreads | debug: valós okok kiírása")

with st.sidebar:
    st.markdown("### ⚙️ Beállítások")

    DEBUG = st.toggle("🔎 Debug mód (ligánkénti státusz + hibák)", value=True)

    leagues = st.multiselect("Ligák", DEFAULT_LEAGUES, default=DEFAULT_LEAGUES)
    window_hours = st.slider("Időablak (óra)", min_value=12, max_value=168, value=WINDOW_HOURS_DEFAULT, step=12)

    min_odds = st.number_input("Min odds / tipp", value=float(MIN_LEG_ODDS_DEFAULT), step=0.01, format="%.2f")
    max_odds = st.number_input("Max odds / tipp", value=float(MAX_LEG_ODDS_DEFAULT), step=0.01, format="%.2f")

    regions = st.selectbox("Odds API régió", options=["eu", "uk", "eu,uk"], index=0)

    st.divider()

    con = db()
    df_all = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    con.close()

    total = len(df_all)
    won = int((df_all["result"] == "WON").sum()) if total else 0
    lost = int((df_all["result"] == "LOST").sum()) if total else 0
    void = int((df_all["result"] == "VOID").sum()) if total else 0
    pending = int((df_all["result"] == "PENDING").sum()) if total else 0

    decided = df_all[df_all["result"].isin(["WON", "LOST"])].copy() if total else pd.DataFrame()
    hit = (decided["result"].eq("WON").mean() * 100.0) if len(decided) else 0.0

    clv_mean = None
    if total and "clv_percent" in df_all.columns:
        tmp = df_all[df_all["result"].isin(["WON", "LOST"])].copy()
        tmp["clv_percent"] = pd.to_numeric(tmp["clv_percent"], errors="coerce")
        tmp = tmp[tmp["clv_percent"].notna()]
        clv_mean = tmp["clv_percent"].mean() if len(tmp) else None

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Összes tipp", total)
        st.metric("Találat %", f"{hit:.0f}%")
        st.metric("Átlag CLV %", f"{clv_mean*100:.2f}%" if clv_mean is not None else "—")
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
    with st.spinner("Elemzés fut… (CLV frissítés + múlt lezárás + új jelöltek)"):
        res = run_analysis(leagues, window_hours, float(min_odds), float(max_odds), regions, DEBUG)
        st.session_state["last_run"] = res

        if res["clv_updated"] > 0:
            st.success(f"✅ Closing odds/CLV frissítve: {res['clv_updated']} db.")
        if res["updated_results"] > 0:
            st.success(f"✅ Korábbi tippek lezárva: {res['updated_results']} db.")
        if res["clv_updated"] == 0 and res["updated_results"] == 0:
            st.info("ℹ️ Nincs frissítendő korábbi tipp (vagy még nem értek véget / nem meccs közelében futtattad).")

        if DEBUG:
            with st.expander("🔎 Debug: Odds API státusz ligánként", expanded=True):
                dbg = pd.DataFrame(res["debug_rows"], columns=["league", "status", "events", "details"])
                st.dataframe(dbg, use_container_width=True)
                st.caption("Ha HTTPError 401/429: kulcs/kvóta. Ha events=0: nincs meccs/region/market gond.")

if st.session_state["last_run"] is not None:
    res = st.session_state["last_run"]
    ticket = res["ticket"]
    total_odds = res["total_odds"]

    st.subheader("🎫 Ajánlott duplázó (2 tipp)")
    if not ticket:
        st.warning("Nincs elég jelölt az időablakban (VAGY Odds API hiba / 0 event / nincs market). Kapcsold be a Debug módot.")
    else:
        st.markdown(f"**Össz-odds:** `{total_odds:.2f}`  <span class='badge'>cél: ~{TARGET_TOTAL_ODDS:.2f}</span>", unsafe_allow_html=True)

        for idx, t in enumerate(ticket, start=1):
            kickoff_local = t["kickoff"].astimezone() if t.get("kickoff") else None
            meta = t.get("meta", {})
            w = meta.get("weather", {})
            nh = meta.get("news_home", {})
            na = meta.get("news_away", {})

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{idx}  {t['match']}")
            st.markdown(
                f"<span class='muted'>Liga:</span> `{t['league']}`  |  <span class='muted'>Kezdés:</span> **{kickoff_local.strftime('%Y.%m.%d %H:%M')}**"
                if kickoff_local else
                f"<span class='muted'>Liga:</span> `{t['league']}`",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Piac:** `{t['bet_type']}`  |  **Odds:** `{t['odds']:.2f}`  |  **Score:** `{t['score']:.0f}/100`")

            if t["bet_type"] == "H2H":
                st.write(f"**Tipp:** {t['selection']} (rendes játékidő)")
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
                st.caption(f"🌦️ Időjárás (város tipp): {w.get('temp','?')}°C, {w.get('desc','?')}, szél: {w.get('wind','?')} m/s")

            if nh.get("lines") or na.get("lines"):
                with st.expander("📰 Friss hírcímek (forrással)", expanded=False):
                    st.write(f"**{t['home']}**")
                    for line in (nh.get("lines") or ["• nincs releváns friss cím"]):
                        st.write(line)
                    st.write(f"**{t['away']}**")
                    for line in (na.get("lines") or ["• nincs releváns friss cím"]):
                        st.write(line)

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
    SELECT id, created_at, match, league, kickoff_utc,
           bet_type, selection, line, odds,
           closing_odds, clv_percent,
           score, result, home_goals, away_goals
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
    df2["closing_odds"] = pd.to_numeric(df2["closing_odds"], errors="coerce")
    df2["clv_percent"] = pd.to_numeric(df2["clv_percent"], errors="coerce")

    st.caption("Megjegyzés: találat% és CLV% csak a lezárt (WON/LOST) tippekre, VOID/UNKNOWN nélkül.")
    decided = df2[df2["result"].isin(["WON", "LOST"])]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lezárt tippek", len(decided))
    with c2:
        st.metric("Átlag odds", f"{decided['odds'].mean():.2f}" if len(decided) else "—")
    with c3:
        hit = (decided["result"].eq("WON").mean() * 100.0) if len(decided) else 0.0
        st.metric("Találat %", f"{hit:.0f}%")
    with c4:
        v = decided["clv_percent"].dropna()
        st.metric("Átlag CLV %", f"{(v.mean()*100):.2f}%" if len(v) else "—")


