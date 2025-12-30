from __future__ import annotations

import os
import re
import time
import sqlite3
from math import sqrt
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
import streamlit as st


# =========================================================
#  KONFIG
# =========================================================
st.set_page_config(page_title="⚽ TITAN X – Risk-Aware Intelligence", layout="wide", page_icon="⚽")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
WEATHER_KEY = os.getenv("WEATHER_API_KEY", "").strip()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "").strip()

DB_PATH = "titan.db"

TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.90
TOTAL_ODDS_MAX = 2.10
TARGET_LEG_ODDS = sqrt(2)

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

            opening_odds REAL,
            closing_odds REAL,
            clv_percent REAL,
            data_quality TEXT
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


def iso_to_dt(s: str) -> Optional[datetime]:
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


def fmt_dt_local(dt_utc: Optional[datetime]) -> str:
    if not dt_utc:
        return "—"
    try:
        return dt_utc.astimezone().strftime("%Y.%m.%d %H:%M")
    except Exception:
        return dt_utc.strftime("%Y.%m.%d %H:%M")


def short(s: str, n=220) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


# =========================================================
#  FOOTBALL-DATA.ORG
# =========================================================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_KEY} if FOOTBALL_DATA_KEY else {}


@st.cache_data(ttl=300)
def fd_get(url: str, params=None, timeout=15) -> Dict[str, Any]:
    if not FOOTBALL_DATA_KEY:
        return {"_error": "Nincs FOOTBALL_DATA_KEY"}
    r = requests.get(url, headers=fd_headers(), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fd_fixtures_window(hours_ahead: int = 24):
    """
    Valós meccsek listája a football-data.org-ról (időablak: most -> +hours_ahead).
    """
    if not FOOTBALL_DATA_KEY:
        return [], "Nincs FOOTBALL_DATA_KEY (football-data.org)."

    start = now_utc().date().isoformat()
    end = (now_utc() + timedelta(hours=hours_ahead)).date().isoformat()

    try:
        data = fd_get("https://api.football-data.org/v4/matches", params={"dateFrom": start, "dateTo": end})
        matches = data.get("matches", []) or []
        out = []
        for m in matches:
            utc = iso_to_dt(m.get("utcDate"))
            if not utc:
                continue
            if not (now_utc() <= utc <= now_utc() + timedelta(hours=hours_ahead)):
                continue

            ht = (m.get("homeTeam") or {}).get("name")
            at = (m.get("awayTeam") or {}).get("name")
            comp = (m.get("competition") or {}).get("name")
            comp_code = (m.get("competition") or {}).get("code")
            comp_id = (m.get("competition") or {}).get("id")
            mid = m.get("id")

            if ht and at:
                out.append(
                    {
                        "match_id": mid,
                        "competition": comp or comp_code or "ismeretlen",
                        "competition_code": comp_code,
                        "competition_id": comp_id,
                        "home": ht,
                        "away": at,
                        "kickoff_utc": utc,
                        "status": m.get("status", ""),
                    }
                )
        out.sort(key=lambda x: x["kickoff_utc"])
        return out, ""
    except Exception as e:
        return [], f"football-data hiba: {e}"


@st.cache_data(ttl=900)
def fd_standings_by_competition_code(code: str) -> Dict[str, Any]:
    """
    Állás lekérés (ha elérhető). Nem minden versenynél engedi a free tier.
    """
    if not FOOTBALL_DATA_KEY or not code:
        return {"_error": "Nincs FOOTBALL_DATA_KEY vagy nincs code."}
    url = f"https://api.football-data.org/v4/competitions/{code}/standings"
    try:
        return fd_get(url, timeout=20)
    except Exception as e:
        return {"_error": str(e)}


def rank_from_standings(standings_json: Dict[str, Any], team_name: str) -> Optional[int]:
    """
    Kiveszi a csapat helyezését az állásból (ha megtalálható).
    """
    if not standings_json or standings_json.get("_error"):
        return None
    tables = standings_json.get("standings", []) or []
    # jellemzően "TOTAL" vagy "REGULAR_SEASON"
    for block in tables:
        table = block.get("table", []) or []
        for row in table:
            t = (row.get("team") or {}).get("name", "")
            pos = row.get("position")
            if t and pos is not None:
                if team_match_score(t, team_name) >= 0.85:
                    return int(pos)
    return None


# =========================================================
#  SETTLEMENT (eredmények zárása)
# =========================================================
def fd_settle_prediction(pred_row: dict) -> dict:
    match_id = pred_row.get("football_data_match_id")
    if not match_id or not FOOTBALL_DATA_KEY:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    url = f"https://api.football-data.org/v4/matches/{match_id}"
    try:
        m = fd_get(url)
    except Exception:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    status = m.get("status", "")
    if status not in ["FINISHED", "AWARDED"]:
        return {"result": "PENDING", "home_goals": None, "away_goals": None}

    score_ft = (m.get("score") or {}).get("fullTime", {}) or {}
    hg = score_ft.get("home")
    ag = score_ft.get("away")
    if hg is None or ag is None:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    bet_type = pred_row.get("bet_type")
    selection = pred_row.get("selection")
    line = pred_row.get("line")

    if bet_type == "H2H":
        if team_match_score(selection, pred_row.get("home", "")) >= 0.7:
            res = "WON" if hg > ag else "LOST"
        elif team_match_score(selection, pred_row.get("away", "")) >= 0.7:
            res = "WON" if ag > hg else "LOST"
        else:
            res = "UNKNOWN"

    elif bet_type == "TOTALS":
        total = hg + ag
        try:
            ln = float(line)
        except Exception:
            return {"result": "UNKNOWN", "home_goals": int(hg), "away_goals": int(ag)}

        if abs(total - ln) < 1e-9:
            res = "VOID"
        else:
            if str(selection).lower() == "over":
                res = "WON" if total > ln else "LOST"
            elif str(selection).lower() == "under":
                res = "WON" if total < ln else "LOST"
            else:
                res = "UNKNOWN"

    elif bet_type == "SPREADS":
        try:
            ln = float(line)
        except Exception:
            return {"result": "UNKNOWN", "home_goals": int(hg), "away_goals": int(ag)}

        sel = str(selection).upper()
        if sel == "HOME":
            adj = (hg + ln) - ag
        elif sel == "AWAY":
            adj = (ag + ln) - hg
        else:
            adj = None

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


def refresh_past_results():
    con = db()
    df = pd.read_sql_query("SELECT * FROM predictions WHERE result IN ('PENDING','UNKNOWN')", con)
    con.close()
    if df.empty:
        return 0

    updated = 0
    now = now_utc()

    for _, row in df.iterrows():
        kickoff = iso_to_dt(row.get("kickoff_utc", ""))
        if not kickoff:
            continue
        if now < kickoff + timedelta(hours=2):
            continue

        settle = fd_settle_prediction(row.to_dict())
        if settle["result"] == "PENDING":
            continue

        con2 = db()
        cur = con2.cursor()
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
        con2.commit()
        con2.close()
        updated += 1

    return updated


# =========================================================
#  KÜLSŐ ADAT (opcionális)
# =========================================================
@st.cache_data(ttl=600)
def get_weather_basic(city_guess="London"):
    if not WEATHER_KEY:
        return {"temp": None, "desc": "—", "wind": None}
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_guess, "appid": WEATHER_KEY, "units": "metric", "lang": "hu"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "temp": safe_float((data.get("main") or {}).get("temp")),
            "desc": ((data.get("weather") or [{}])[0] or {}).get("description", "—"),
            "wind": safe_float((data.get("wind") or {}).get("speed")),
        }
    except Exception:
        return {"temp": None, "desc": "—", "wind": None}


@st.cache_data(ttl=600)
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
#  ODDS API
# =========================================================
@st.cache_data(ttl=120)
def odds_api_get(league_key: str, markets: List[str], regions: str = "eu"):
    if not ODDS_API_KEY:
        return {"ok": False, "status": "NO_KEY", "detail": "Nincs ODDS_API_KEY", "events": [], "headers": {}}

    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        headers = {
            "x-requests-remaining": r.headers.get("x-requests-remaining"),
            "x-requests-used": r.headers.get("x-requests-used"),
            "x-requests-last": r.headers.get("x-requests-last"),
        }

        if r.status_code != 200:
            detail = ""
            try:
                detail = (r.text or "")[:600]
            except Exception:
                detail = ""
            return {"ok": False, "status": f"HTTP {r.status_code}", "detail": detail, "events": [], "headers": headers}

        js = r.json()
        if not isinstance(js, list):
            return {"ok": False, "status": "BAD_FORMAT", "detail": "Nem lista válasz", "events": [], "headers": headers}

        return {"ok": True, "status": "OK", "detail": "", "events": js, "headers": headers}
    except Exception as e:
        return {"ok": False, "status": "EXCEPTION", "detail": str(e), "events": [], "headers": {}}


def odds_error_hu(status: str, detail: str) -> str:
    s = (status or "").upper()
    d = (detail or "").lower()

    if s == "NO_KEY":
        return "❌ Nincs ODDS_API_KEY → odds alapú tipp nem készül, csak RIZIKÓS (fixtures alapú) javaslat."
    if "HTTP 401" in s and "quota" in d:
        return "❌ The Odds API quota elfogyott (401). Odds nélkül csak RIZIKÓS (fixtures alapú) javaslat."
    if "HTTP 401" in s:
        return "❌ The Odds API 401: hibás kulcs / hozzáférés / quota. Odds nélkül csak RIZIKÓS mód."
    if "HTTP 429" in s:
        return "❌ The Odds API 429 (rate limit). Odds nélkül csak RIZIKÓS mód."
    if s in ("BAD_FORMAT", "EXCEPTION"):
        return f"❌ Odds API hiba: {status} – {short(detail, 160)}"
    return f"❌ Odds API hiba: {status} – {short(detail, 160)}"


def extract_candidates_from_match(m: dict, min_odds: float, max_odds: float) -> List[dict]:
    """
    Valós odds jelöltek.
    """
    out = []
    home = m.get("home_team")
    away = m.get("away_team")
    kickoff = iso_to_dt(m.get("commence_time"))
    if not home or not away or not kickoff:
        return out

    bookmakers = m.get("bookmakers", []) or []

    def collect_prices(market_key: str):
        prices = []
        for b in bookmakers:
            for mk in b.get("markets", []) or []:
                if mk.get("key") != market_key:
                    continue
                prices.append((b.get("key") or "book", mk.get("outcomes", []) or []))
        return prices

    # ---------- H2H ----------
    h2h_blocks = collect_prices("h2h")
    team_prices = {home: [], away: []}
    for _bk, outs in h2h_blocks:
        for o in outs:
            nm = o.get("name")
            pr = safe_float(o.get("price"))
            if pr is None:
                continue
            if nm == home:
                team_prices[home].append(pr)
            elif nm == away:
                team_prices[away].append(pr)

    if team_prices[home] and team_prices[away]:
        avg_home = sum(team_prices[home]) / len(team_prices[home])
        avg_away = sum(team_prices[away]) / len(team_prices[away])
        fav = home if avg_home < avg_away else away

        best_odds = max(team_prices[fav])
        avg_odds = (sum(team_prices[fav]) / len(team_prices[fav])) if team_prices[fav] else best_odds

        if min_odds <= best_odds <= max_odds:
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
                    "odds": float(best_odds),
                    "avg_odds": float(avg_odds),
                    "data_quality": "LIVE",
                }
            )

    # ---------- TOTALS ----------
    totals_blocks = collect_prices("totals")
    totals_map = {}  # (point, name)-> list[price]
    for _bk, outs in totals_blocks:
        for o in outs:
            nm = (o.get("name") or "").strip().capitalize()
            pt = safe_float(o.get("point"))
            pr = safe_float(o.get("price"))
            if nm.lower() not in ("over", "under"):
                continue
            if pt is None or pr is None:
                continue
            totals_map.setdefault((float(pt), nm), []).append(pr)

    for target_line in (2.5, 3.5, 1.5):
        found_any = False
        for nm in ("Over", "Under"):
            ps = totals_map.get((float(target_line), nm), [])
            if not ps:
                continue
            best_odds = max(ps)
            avg_odds = sum(ps) / len(ps)
            if min_odds <= best_odds <= max_odds:
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
                        "odds": float(best_odds),
                        "avg_odds": float(avg_odds),
                        "data_quality": "LIVE",
                    }
                )
                found_any = True
        if found_any:
            break

    # ---------- SPREADS ----------
    spreads_blocks = collect_prices("spreads")
    spreads_map = {}  # (point, teamname)-> list[price]
    for _bk, outs in spreads_blocks:
        for o in outs:
            nm = o.get("name")
            pt = safe_float(o.get("point"))
            pr = safe_float(o.get("price"))
            if not nm or pt is None or pr is None:
                continue
            spreads_map.setdefault((float(pt), nm), []).append(pr)

    preferred_points = (-1.0, -0.5, 0.5, 1.0)
    for p in preferred_points:
        keys = [k for k in spreads_map.keys() if abs(k[0] - float(p)) < 1e-9]
        if not keys:
            continue
        for (pt, team_nm) in keys:
            ps = spreads_map.get((pt, team_nm), [])
            if not ps:
                continue
            best_odds = max(ps)
            avg_odds = sum(ps) / len(ps)
            if min_odds <= best_odds <= max_odds:
                sel = "HOME" if team_match_score(team_nm, home) >= team_match_score(team_nm, away) else "AWAY"
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
                        "odds": float(best_odds),
                        "avg_odds": float(avg_odds),
                        "data_quality": "LIVE",
                    }
                )
        break

    return out


# =========================================================
#  PONTOZÁS (odds-alapú) + magyar indoklás
# =========================================================
def score_candidate(c: dict) -> Tuple[float, str, dict]:
    odds = safe_float(c.get("odds"), 0.0) or 0.0
    avg_odds = safe_float(c.get("avg_odds"), odds) or odds

    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))

    value_score = (odds / avg_odds) - 1.0 if avg_odds > 0 else 0.0
    value_bonus = max(-10.0, min(10.0, 60.0 * value_score))

    city_guess = (c.get("home", "London").split()[-1] if c.get("home") else "London")
    w = get_weather_basic(city_guess)

    weather_pen = 0.0
    if w.get("wind") is not None and w["wind"] >= 12:
        weather_pen -= 6
    if isinstance(w.get("desc"), str) and any(x in w["desc"].lower() for x in ["eső", "zápor", "vihar"]):
        weather_pen -= 4

    news_home = news_brief(c.get("home", ""))
    time.sleep(0.03)
    news_away = news_brief(c.get("away", ""))

    news_bias = 0
    if c.get("bet_type") == "H2H":
        if team_match_score(c.get("selection", ""), c.get("home", "")) >= 0.7:
            news_bias = int(news_home.get("score", 0))
        else:
            news_bias = int(news_away.get("score", 0))
    else:
        news_bias = int(news_home.get("score", 0)) + int(news_away.get("score", 0))

    news_score = float(news_bias) * 6.0

    raw = 55.0 + odds_score + value_bonus + news_score + weather_pen
    final = max(0.0, min(100.0, raw))

    bt = c.get("bet_type")
    if bt == "H2H":
        bet_label = f"Végkimenetel: **{c.get('selection')}**"
    elif bt == "TOTALS":
        bet_label = f"Gólok száma: **{c.get('selection')} {c.get('line')}**"
    elif bt == "SPREADS":
        side = "Hazai" if c.get("selection") == "HOME" else "Vendég"
        bet_label = f"Hendikep: **{side} {c.get('line')}**"
    else:
        bet_label = f"Piac: {bt}"

    why = []
    why.append(f"Odds: **{odds:.2f}** (piaci átlag ~{avg_odds:.2f}, value: {value_score*100:.1f}%).")
    why.append(f"Cél: 2 tipp össz-odds ~**{TARGET_TOTAL_ODDS:.2f}**.")
    if news_bias > 0:
        why.append("Hírek: **inkább pozitív**.")
    elif news_bias < 0:
        why.append("Hírek: **kockázati jel** (sérülés/hiányzó gyanú).")
    else:
        why.append("Hírek: nincs erős extra jel.")
    if w.get("temp") is not None:
        why.append(f"Időjárás (tipp): {w['temp']:.0f}°C, {w.get('desc','—')}, szél: {w.get('wind','?')} m/s.")

    reasoning = bet_label + "\n\n" + " ".join(why)
    meta = {"weather": w, "news_home": news_home, "news_away": news_away, "value_score": value_score}
    return final, reasoning, meta


# =========================================================
#  RISK MODE – ha nincs odds
#  (fixtures + standings-heurisztika)
# =========================================================
def risk_pick_from_fixtures(fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mindig visszaad 2 meccset, ha van legalább 2 fixture.
    Tipp: rang alapján favorit (ha van standings), különben "NO BET" jellegű figyelmeztetés.
    """
    if len(fixtures) < 2:
        return []

    picks = []
    for fx in fixtures:
        home = fx["home"]
        away = fx["away"]
        ko = fx["kickoff_utc"]
        comp_code = fx.get("competition_code") or ""

        # standings próbálkozás
        home_rank = away_rank = None
        standings = None
        if comp_code:
            standings = fd_standings_by_competition_code(comp_code)
            home_rank = rank_from_standings(standings, home)
            away_rank = rank_from_standings(standings, away)

        # döntés
        if home_rank is not None and away_rank is not None:
            fav = home if home_rank < away_rank else away
            gap = abs(home_rank - away_rank)
            # gap -> confidence
            conf = min(0.75, 0.40 + 0.03 * gap)
            risk = "KÖZEPES" if gap >= 6 else "MAGAS"
            bet = f"RIZIKÓS favorit (állás alapján): {fav}"
            why = f"Állás: {home_rank}. vs {away_rank}. (minél nagyobb a különbség, annál jobb esély, de odds nélkül ez nem value-alapú)."
        else:
            fav = home  # csak hogy legyen selection
            conf = 0.25
            risk = "NAGYON MAGAS"
            bet = "NEM AJÁNLOTT tipp (nincs odds/állás adat) – csak meccslista"
            why = "Nincs megbízható odds (The Odds API quota/hiba) és/vagy nem elérhető standings. Ez csak “RISK MODE”."

        # news/weather minimál
        nh = news_brief(home) if NEWS_API_KEY else {"score": 0, "lines": []}
        na = news_brief(away) if NEWS_API_KEY else {"score": 0, "lines": []}

        news_bias = (nh.get("score", 0) or 0) + (na.get("score", 0) or 0)
        if news_bias < 0:
            conf = max(0.15, conf - 0.10)
            why += " Hírekben kockázati jel → bizalom csökkent."
        elif news_bias > 0:
            conf = min(0.85, conf + 0.05)

        score = int(round(conf * 100))

        picks.append(
            {
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": fx.get("competition") or comp_code or "ismeretlen",
                "kickoff": ko,
                "bet_type": "RISK_MODE",
                "market_key": "risk",
                "selection": fav,
                "line": None,
                "bookmaker": "N/A",
                "odds": None,
                "avg_odds": None,
                "data_quality": "RISK",
                "score": score,
                "reasoning": f"⚠️ **RISK MODE** – {bet}\n\n**Miért:** {why}\n**Rizikó:** {risk}\n**Bizalom:** {score}/100",
                "meta": {"news_home": nh, "news_away": na},
                "football_data_match_id": fx.get("match_id"),
            }
        )

    # legjobb 2 (score alapján) + külön meccs
    picks = sorted(picks, key=lambda x: x.get("score", 0), reverse=True)
    out = []
    seen = set()
    for p in picks:
        if p["match"] in seen:
            continue
        out.append(p)
        seen.add(p["match"])
        if len(out) == 2:
            break
    return out


# =========================================================
#  DUÓ (odds-alapú)
# =========================================================
def pick_best_duo(cands: List[dict]) -> Tuple[List[dict], float]:
    if len(cands) < 2:
        return [], 0.0

    best = (None, None, -1e18, 0.0)
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

            if util > best[2]:
                best = (i, j, util, total_odds)

    if best[0] is None:
        top2 = sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)[:2]
        if len(top2) < 2:
            return [], 0.0
        return top2, float(top2[0]["odds"]) * float(top2[1]["odds"])

    return [cands[best[0]], cands[best[1]]], best[3]


# =========================================================
#  Mentés
# =========================================================
def save_ticket(ticket: List[dict]):
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
             score, reasoning, football_data_match_id,
             opening_odds, closing_odds, clv_percent, data_quality)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                (float(t.get("odds")) if t.get("odds") is not None else None),
                float(t.get("score", 0.0)),
                t.get("reasoning"),
                t.get("football_data_match_id"),
                (float(t.get("odds")) if t.get("odds") is not None else None),
                None,
                None,
                t.get("data_quality", "LIVE"),
            ),
        )
    con.commit()
    con.close()


# =========================================================
#  FŐ FUTÁS
# =========================================================
def run_engine(
    leagues: List[str],
    window_hours: int,
    min_odds: float,
    max_odds: float,
    regions: str,
    debug: bool,
) -> Dict[str, Any]:
    updated_results = refresh_past_results()

    fixtures, fx_err = fd_fixtures_window(hours_ahead=int(window_hours))

    candidates: List[dict] = []
    debug_rows = []
    quota_info = {"remaining": None, "used": None, "last": None}
    odds_ok_any = False
    first_odds_error: Optional[str] = None

    # odds lekérés – ha lehet
    now = now_utc()
    limit = now + timedelta(hours=int(window_hours))

    for lg in leagues:
        resp = odds_api_get(lg, REQUEST_MARKETS, regions=regions)

        headers = resp.get("headers", {}) or {}
        if headers:
            quota_info["remaining"] = headers.get("x-requests-remaining") or quota_info["remaining"]
            quota_info["used"] = headers.get("x-requests-used") or quota_info["used"]
            quota_info["last"] = headers.get("x-requests-last") or quota_info["last"]

        if not resp.get("ok"):
            if first_odds_error is None:
                first_odds_error = odds_error_hu(resp.get("status", ""), resp.get("detail", ""))
            if debug:
                debug_rows.append((lg, resp.get("status", "ERR"), 0, short(resp.get("detail", ""), 220)))
            continue

        odds_ok_any = True
        events = resp.get("events", []) or []
        if debug:
            debug_rows.append((lg, "OK", len(events), ""))

        for m in events:
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
                # settlementhez football-data match id (best effort)
                # itt nem keresünk külön, mert fixtures listában ott a match_id;
                # de odds event nem ad FD match_id-t. (Best effort: később is lehet.)
                c["football_data_match_id"] = None
                candidates.append(c)

            time.sleep(0.01)

    candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

    # 1) ha van odds, odds-alapú ticket
    if odds_ok_any and candidates:
        ticket, total_odds = pick_best_duo(candidates)
        mode = "LIVE_ODDS"
        warning = None
    else:
        # 2) fallback: RISK MODE – fixtures-ből mindig ad 2-t
        ticket = risk_pick_from_fixtures(fixtures)
        total_odds = 0.0
        mode = "RISK_MODE"
        warning = first_odds_error or "⚠️ Nincs odds adat. RISK MODE aktív."

    return {
        "mode": mode,
        "warning": warning,
        "updated_results": updated_results,
        "fixtures": fixtures,
        "fixtures_error": fx_err,
        "candidates": candidates,
        "ticket": ticket,
        "total_odds": total_odds,
        "debug_rows": debug_rows,
        "quota_info": quota_info,
    }


# =========================================================
#  UI – TELJESEN ÚJ, innovatív (tabs + glass cards + KPI)
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@300;400;600;700&display=swap');

:root{
  --bg0:#050512;
  --bg1:#080a1f;
  --bg2:#0d1233;
  --glass: rgba(255,255,255,0.06);
  --glass2: rgba(255,255,255,0.035);
  --stroke: rgba(255,255,255,0.11);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --a:#00e5ff;
  --b:#a855f7;
  --c:#ff2d95;
  --ok:#5CFF7A;
  --warn:#ffcc66;
  --bad:#ff5d5d;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp{
  background:
    radial-gradient(800px 500px at 15% 10%, rgba(0,229,255,0.18), transparent 60%),
    radial-gradient(900px 600px at 85% 25%, rgba(255,45,149,0.16), transparent 62%),
    radial-gradient(700px 500px at 65% 85%, rgba(168,85,247,0.14), transparent 55%),
    linear-gradient(140deg, var(--bg0) 0%, var(--bg2) 55%, var(--bg1) 100%);
}

.titanTop{
  border:1px solid var(--stroke);
  background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border-radius: 22px;
  padding: 18px 18px 14px 18px;
  box-shadow: 0 22px 70px rgba(0,0,0,0.45);
}

.brand{
  font-family:'Orbitron', sans-serif;
  letter-spacing: 0.6px;
  font-weight: 800;
  font-size: 2.05rem;
  background: linear-gradient(90deg, var(--a), var(--b), var(--c));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin:0;
}
.tagline{ color: var(--muted); margin-top: 4px; }

.kpiRow{ display:flex; gap:10px; flex-wrap:wrap; margin-top: 12px;}
.kpi{
  flex: 1 1 180px;
  border:1px solid var(--stroke);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 10px 12px;
}
.kpi .t{ color: var(--muted); font-size: 0.82rem; }
.kpi .v{ font-weight: 800; font-size: 1.12rem; margin-top: 2px; }

.ribbon{
  margin-top: 12px;
  border: 1px dashed rgba(0,229,255,0.35);
  background: rgba(0,229,255,0.06);
  border-radius: 16px;
  padding: 10px 12px;
  color: rgba(255,255,255,0.88);
}

.card{
  border:1px solid var(--stroke);
  background: rgba(255,255,255,0.045);
  border-radius: 22px;
  padding: 16px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.42);
}

.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding: 4px 10px;
  border-radius: 999px;
  border:1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  font-size: 0.86rem;
  color: rgba(255,255,255,0.88);
  margin-right: 8px;
}

.pill.ok{ border-color: rgba(92,255,122,0.35); background: rgba(92,255,122,0.07); }
.pill.warn{ border-color: rgba(255,204,102,0.35); background: rgba(255,204,102,0.07); }
.pill.bad{ border-color: rgba(255,93,93,0.40); background: rgba(255,93,93,0.08); }

.hr{ height:1px; background: rgba(255,255,255,0.09); margin: 14px 0; }

.confWrap{ margin-top: 10px; }
.confBar{
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  overflow:hidden;
}
.confFill{
  height: 100%;
  width: var(--w);
  background: linear-gradient(90deg, var(--c), var(--b), var(--a));
}

.small{ color: var(--muted); font-size: 0.90rem; }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar – minimal és tiszta
with st.sidebar:
    st.markdown("### ⚙️ Motor beállítások")
    DEBUG = st.toggle("Debug tábla", value=True)
    leagues = st.multiselect("Odds ligák", DEFAULT_LEAGUES, default=DEFAULT_LEAGUES)
    window_hours = st.slider("Időablak (óra)", 12, 168, 24, 12)
    min_odds = st.number_input("Min odds / tipp", value=1.25, step=0.01, format="%.2f")
    max_odds = st.number_input("Max odds / tipp", value=1.95, step=0.01, format="%.2f")
    regions = st.selectbox("Odds régió", ["eu", "uk", "eu,uk"], index=2)

    st.markdown("---")
    st.markdown("### 🔑 Kulcsok")
    st.write(f"ODDS_API_KEY: {'✅' if ODDS_API_KEY else '❌'}")
    st.write(f"FOOTBALL_DATA_KEY: {'✅' if FOOTBALL_DATA_KEY else '❌'}")
    st.write(f"WEATHER_KEY: {'✅' if WEATHER_KEY else '—'}")
    st.write(f"NEWS_API_KEY: {'✅' if NEWS_API_KEY else '—'}")


# Session init
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

# Header + Actions
colL, colR = st.columns([1.5, 1.0])
with colL:
    st.markdown(
        f"""
<div class="titanTop">
  <div class="brand">TITAN X – Risk-Aware Intelligence</div>
  <div class="tagline">Mindig kapsz meccset. Ha nincs odds → <b>RISK MODE</b> (nem value-alapú, csak tájékoztató).</div>
</div>
""",
        unsafe_allow_html=True,
    )

with colR:
    st.markdown("<div class='titanTop'>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Futtatás", type="primary", use_container_width=True)
    save_btn = st.button("💾 Ticket mentése DB-be", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if run_btn:
    with st.spinner("Futtatás: settle + adatlekérés + ticket…"):
        st.session_state["last_run"] = run_engine(
            leagues=leagues,
            window_hours=int(window_hours),
            min_odds=float(min_odds),
            max_odds=float(max_odds),
            regions=regions,
            debug=DEBUG,
        )

# Render tabs
tab1, tab2, tab3 = st.tabs(["🏆 Dashboard", "📅 Fixtures", "📜 History"])

res = st.session_state["last_run"]

with tab1:
    if res is None:
        st.info("Indíts egy futtatást a bal felső 🚀 gombbal.")
    else:
        qi = res.get("quota_info", {}) or {}
        mode = res.get("mode")
        warning = res.get("warning")
        upd = res.get("updated_results", 0)

        # KPIs
        remaining = qi.get("remaining") or "—"
        used = qi.get("used") or "—"
        last_cost = qi.get("last") or "—"
        ticket = res.get("ticket", []) or []
        total_odds = res.get("total_odds", 0.0)

        mode_pill = "LIVE_ODDS" if mode == "LIVE_ODDS" else "RISK_MODE"
        mode_class = "ok" if mode == "LIVE_ODDS" else "warn"

        st.markdown(
            f"""
<div class="titanTop">
  <div class="kpiRow">
    <div class="kpi"><div class="t">Mód</div><div class="v"><span class="pill {mode_class}">{mode_pill}</span></div></div>
    <div class="kpi"><div class="t">Settled frissítés</div><div class="v">{upd} db</div></div>
    <div class="kpi"><div class="t">Odds quota remaining</div><div class="v">{remaining}</div></div>
    <div class="kpi"><div class="t">Odds used / last cost</div><div class="v">{used} / {last_cost}</div></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if warning:
            st.markdown(f"<div class='ribbon'>{warning}</div>", unsafe_allow_html=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # Ticket cards
        st.subheader("🎫 TOP 2 ajánlás (mindig ad meccset)")
        if not ticket:
            st.warning("Nincs elég meccs az időablakban (vagy nincs football-data kulcs).")
        else:
            if mode == "LIVE_ODDS":
                st.markdown(
                    f"<span class='pill ok'>Össz-odds: {total_odds:.2f}</span> "
                    f"<span class='pill'>Cél: ~{TARGET_TOTAL_ODDS:.2f}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='pill warn'>RISK MODE</span> <span class='pill'>Odds nélkül: nem value-alapú</span>",
                    unsafe_allow_html=True,
                )

            for i, t in enumerate(ticket, start=1):
                score = int(t.get("score", 0))
                klass = "ok" if score >= 70 else ("warn" if score >= 45 else "bad")

                kickoff = fmt_dt_local(t.get("kickoff"))
                league = t.get("league") or t.get("league_key") or "—"
                bet_type = t.get("bet_type")
                selection = t.get("selection")
                odds = t.get("odds")

                odds_txt = f"{odds:.2f}" if isinstance(odds, (int, float)) else "—"
                data_quality = t.get("data_quality", "—")

                st.markdown(
                    f"""
<div class="card">
  <div style="display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div>
      <div style="font-weight:800; font-size:1.08rem;">#{i} · {t.get("match","—")}</div>
      <div class="small">Liga: <b>{league}</b> · Kezdés: <b>{kickoff}</b> · Adat: <b>{data_quality}</b></div>
    </div>
    <div>
      <span class="pill {klass}">Bizalom: {score}/100</span>
      <span class="pill">Piac: {bet_type}</span>
      <span class="pill">Odds: {odds_txt}</span>
    </div>
  </div>

  <div class="confWrap">
    <div class="small">Confidence bar</div>
    <div class="confBar"><div class="confFill" style="--w:{score}%;"></div></div>
  </div>

  <div class="hr"></div>
  <div style="font-weight:700;">Javaslat:</div>
  <div style="margin-top:6px;">{selection}</div>
  <div class="hr"></div>
  <div style="white-space:pre-wrap; color:rgba(255,255,255,0.88);">{t.get("reasoning","")}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # Debug table
        if DEBUG:
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.subheader("🔎 Debug – Odds státusz ligánként")
            dbg = pd.DataFrame(res.get("debug_rows", []), columns=["league", "status", "events", "details"])
            st.dataframe(dbg, use_container_width=True)

# Save
if save_btn:
    if res is None or not (res.get("ticket") or []):
        st.warning("Nincs mit menteni. Futtass előbb.")
    else:
        save_ticket(res["ticket"])
        st.success("✅ Ticket mentve DB-be.")

with tab2:
    st.subheader("📅 Valós meccsek (football-data.org)")
    if res is None:
        fixtures, fx_err = fd_fixtures_window(hours_ahead=int(window_hours))
    else:
        fixtures, fx_err = res.get("fixtures", []), res.get("fixtures_error", "")

    if fx_err:
        st.warning(fx_err)
    elif not fixtures:
        st.info("Nincs meccs az időablakban.")
    else:
        fx_df = pd.DataFrame(
            [
                {
                    "Kezdés (helyi)": fmt_dt_local(x["kickoff_utc"]),
                    "Liga": x["competition"],
                    "Meccs": f"{x['home']} vs {x['away']}",
                    "Státusz": x.get("status", ""),
                    "match_id": x["match_id"],
                    "code": x.get("competition_code"),
                }
                for x in fixtures[:400]
            ]
        )
        st.dataframe(fx_df, use_container_width=True)

with tab3:
    st.subheader("📜 Előzmények + statisztika")
    con = db()
    df = pd.read_sql_query(
        """
        SELECT id, created_at, match, league, kickoff_utc,
               bet_type, selection, line, odds, opening_odds, closing_odds, clv_percent,
               score, result, data_quality
        FROM predictions
        ORDER BY id DESC
        LIMIT 500
        """,
        con,
    )
    con.close()

    st.dataframe(df, use_container_width=True)

    if not df.empty:
        decided = df[df["result"].isin(["WON", "LOST"])]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Összes tipp", len(df))
        with c2:
            st.metric("Lezárt tippek", len(decided))
        with c3:
            hit = (decided["result"].eq("WON").mean() * 100.0) if len(decided) else 0.0
            st.metric("Találat % (W/L)", f"{hit:.0f}%")
        with c4:
            clv_mean = pd.to_numeric(df["clv_percent"], errors="coerce").dropna()
            st.metric("Átlag CLV%", f"{clv_mean.mean():.2f}%" if not clv_mean.empty else "—")
