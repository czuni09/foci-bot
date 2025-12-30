from __future__ import annotations

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

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
WEATHER_KEY = os.getenv("WEATHER_API_KEY", "").strip()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "").strip()

DB_PATH = "titan.db"

# Duplázó cél
TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.90
TOTAL_ODDS_MAX = 2.10

# Egy tippre “ideális” odds kb. sqrt(2) ≈ 1.414
TARGET_LEG_ODDS = sqrt(2)

# Odds API v4: dokumentált alappiacok
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


def fmt_dt_local(dt_utc: datetime):
    if not dt_utc:
        return "—"
    try:
        return dt_utc.astimezone().strftime("%Y.%m.%d %H:%M")
    except Exception:
        return dt_utc.strftime("%Y.%m.%d %H:%M")


# =========================================================
#  FOOTBALL-DATA.ORG (fixtures + settlement)
# =========================================================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_KEY} if FOOTBALL_DATA_KEY else {}


@st.cache_data(ttl=300)
def fd_get(url, params=None, timeout=15):
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
            mid = m.get("id")

            if ht and at:
                out.append(
                    {
                        "match_id": mid,
                        "competition": comp or comp_code or "ismeretlen",
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


def fd_find_match_id(home: str, away: str, kickoff_utc: datetime):
    """
    Megpróbálja a football-data match ID-t megtalálni csapatnév + idő illesztéssel (±1 nap, ±8 óra).
    """
    if not FOOTBALL_DATA_KEY or not kickoff_utc:
        return None

    date_from = (kickoff_utc.date() - timedelta(days=1)).isoformat()
    date_to = (kickoff_utc.date() + timedelta(days=1)).isoformat()

    try:
        data = fd_get("https://api.football-data.org/v4/matches", params={"dateFrom": date_from, "dateTo": date_to})
        candidates = data.get("matches", []) or []
    except Exception:
        return None

    best = (0.0, None)
    for m in candidates:
        try:
            fd_home = (m.get("homeTeam") or {}).get("name", "")
            fd_away = (m.get("awayTeam") or {}).get("name", "")
            fd_utc = iso_to_dt(m.get("utcDate"))
        except Exception:
            continue

        if not fd_home or not fd_away or not fd_utc:
            continue

        if abs((fd_utc - kickoff_utc).total_seconds()) > 8 * 3600:
            continue

        s = (team_match_score(home, fd_home) + team_match_score(away, fd_away)) / 2.0
        if s > best[0]:
            best = (s, m.get("id"))

    return best[1] if best[0] >= 0.60 else None


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
#  KÜLSŐ ADAT – időjárás / hírek (opcionális)
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
#  ODDS API – biztonságos hívás + quota headerek
# =========================================================
@st.cache_data(ttl=120)
def odds_api_get(league_key: str, markets: list[str], regions: str = "eu"):
    if not ODDS_API_KEY:
        return {"ok": False, "status": "Nincs ODDS_API_KEY", "events": [], "headers": {}}

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
                detail = (r.text or "")[:500]
            except Exception:
                detail = ""
            return {"ok": False, "status": f"HTTP {r.status_code}", "detail": detail, "events": [], "headers": headers}

        js = r.json()
        if not isinstance(js, list):
            return {"ok": False, "status": "Nem lista válasz", "events": [], "headers": headers}
        return {"ok": True, "status": "OK", "events": js, "headers": headers}
    except Exception as e:
        return {"ok": False, "status": f"Exception: {e}", "events": [], "headers": {}}


def extract_candidates_from_match(m: dict, min_odds: float, max_odds: float) -> list[dict]:
    """
    Valós odds jelöltek:
    - H2H: favorit (alacsonyabb odds) *best price* (max) a bookok között
    - TOTALS: 2.5/3.5/1.5 (Over/Under) *best price*
    - SPREADS: -1/-0.5/0.5/1 (HOME/AWAY) *best price*
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
            nm = (o.get("name") or "").strip().capitalize()  # Over/Under
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
#  PONTOZÁS (súlyozás) + magyar indoklás
# =========================================================
def score_candidate(c: dict) -> tuple[float, str, dict]:
    odds = safe_float(c.get("odds"), 0.0) or 0.0
    avg_odds = safe_float(c.get("avg_odds"), odds) or odds

    # odds-komponens: közel a sqrt(2)-höz
    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))

    # value (best vs avg): ha egyik book “jobb árat ad”
    value_score = 0.0
    if avg_odds > 0:
        value_score = (odds / avg_odds) - 1.0
    value_bonus = max(-10.0, min(10.0, 60.0 * value_score))  # korlát

    # időjárás / hírek (opcionális)
    city_guess = (c.get("home", "London").split()[-1] if c.get("home") else "London")
    w = get_weather_basic(city_guess)
    weather_pen = 0.0
    if w.get("wind") is not None and w["wind"] >= 12:
        weather_pen -= 6
    if isinstance(w.get("desc"), str) and any(x in w["desc"].lower() for x in ["eső", "zápor", "vihar"]):
        weather_pen -= 4

    news_home = news_brief(c.get("home", ""))
    time.sleep(0.05)
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

    raw = 50.0 + odds_score + value_bonus + news_score + weather_pen
    final = max(0.0, min(100.0, raw))

    # Fogadás label
    bt = c.get("bet_type")
    if bt == "H2H":
        bet_label = f"Végkimenetel (1X2/H2H): **{c.get('selection')}**"
    elif bt == "TOTALS":
        bet_label = f"Gólok száma (Over/Under): **{c.get('selection')} {c.get('line')}**"
    elif bt == "SPREADS":
        side = "Hazai" if c.get("selection") == "HOME" else "Vendég"
        bet_label = f"Hendikep: **{side} {c.get('line')}**"
    else:
        bet_label = f"Piac: {bt}"

    why = []
    why.append(f"Odds: **{odds:.2f}** (piaci átlag ~{avg_odds:.2f}, value: {value_score*100:.1f}%).")
    why.append(f"A cél a stabil duplázó: 2 tipp össz-odds ~{TARGET_TOTAL_ODDS:.2f}.")
    if news_bias > 0:
        why.append("Hírek összképe: **pozitív**.")
    elif news_bias < 0:
        why.append("Hírekben van **kockázati jel** (hiányzó/sérülés gyanú).")
    else:
        why.append("Hírek alapján nincs erős extra jel.")
    if w.get("temp") is not None:
        why.append(f"Időjárás (tipp): {w['temp']:.0f}°C, {w.get('desc','—')} (szél: {w.get('wind','?')} m/s).")

    reasoning = bet_label + "\n\n" + " ".join(why)
    meta = {"weather": w, "news_home": news_home, "news_away": news_away, "value_score": value_score}
    return final, reasoning, meta


# =========================================================
#  DUÓ KIVÁLASZTÁS (2 tipp) – csak valós odds esetén
# =========================================================
def pick_best_duo(cands: list[dict]) -> tuple[list[dict], float]:
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
#  Mentés + CLV frissítés (csak ha Odds API él)
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
                float(t.get("odds", 0.0)),
                float(t.get("score", 0.0)),
                t.get("reasoning"),
                t.get("football_data_match_id"),
                float(t.get("odds", 0.0)),  # opening = aktuális
                None,
                None,
                "LIVE",
            ),
        )
    con.commit()
    con.close()


def update_clv_for_pending(regions: str, markets: list[str]):
    """
    CLV frissítés:
    - pending tippek, amik 0-3 órán belül kezdődnek vagy már elkezdődtek
    - újra lekéri a ligát, match+selection alapján best closing odds
    - closing_odds + clv_percent frissítés
    """
    if not ODDS_API_KEY:
        return 0, "Nincs ODDS_API_KEY, CLV nem frissíthető."

    con = db()
    df = pd.read_sql_query(
        """
        SELECT id, league, match, home, away, kickoff_utc, bet_type, market_key, selection, line,
               opening_odds, closing_odds, clv_percent
        FROM predictions
        WHERE result='PENDING'
        ORDER BY id DESC
        LIMIT 300
        """,
        con,
    )
    con.close()

    if df.empty:
        return 0, ""

    now = now_utc()
    targets = []
    for _, r in df.iterrows():
        ko = iso_to_dt(r.get("kickoff_utc", ""))
        if not ko:
            continue
        if ko < now - timedelta(hours=3) or ko > now + timedelta(hours=3):
            continue
        targets.append(r.to_dict())

    if not targets:
        return 0, ""

    updated = 0
    by_league = {}
    for t in targets:
        lg = t.get("league") or ""
        by_league.setdefault(lg, []).append(t)

    for lg, rows in by_league.items():
        if not lg:
            continue

        resp = odds_api_get(lg, markets, regions=regions)
        if not resp.get("ok"):
            return updated, f"CLV: Odds API nem elérhető ({resp.get('status')})."

        events = resp.get("events", []) or []

        for row in rows:
            home = row.get("home", "")
            away = row.get("away", "")
            bet_type = row.get("bet_type")
            selection = row.get("selection")
            line = row.get("line")
            ko = iso_to_dt(row.get("kickoff_utc", ""))

            best = None
            for ev in events:
                eh = ev.get("home_team")
                ea = ev.get("away_team")
                ek = iso_to_dt(ev.get("commence_time"))
                if not eh or not ea or not ek or not ko:
                    continue
                if abs((ek - ko).total_seconds()) > 3 * 3600:
                    continue
                if (team_match_score(eh, home) + team_match_score(ea, away)) / 2.0 < 0.75:
                    continue

                cands = extract_candidates_from_match(ev, min_odds=1.01, max_odds=100.0)
                for c in cands:
                    if c.get("bet_type") != bet_type:
                        continue

                    if bet_type == "H2H":
                        if team_match_score(c.get("selection", ""), selection) < 0.9:
                            continue
                    elif bet_type == "TOTALS":
                        if str(c.get("selection")).lower() != str(selection).lower():
                            continue
                        if abs(float(c.get("line", 0.0)) - float(line or 0.0)) > 1e-9:
                            continue
                    elif bet_type == "SPREADS":
                        if str(c.get("selection")).upper() != str(selection).upper():
                            continue
                        if abs(float(c.get("line", 0.0)) - float(line or 0.0)) > 1e-9:
                            continue

                    cand_odds = safe_float(c.get("odds"))
                    if cand_odds is None:
                        continue
                    if best is None or cand_odds > best:
                        best = cand_odds

            if best is None:
                continue

            opening = safe_float(row.get("opening_odds"))
            if opening and best:
                clv = ((opening - best) / opening) * 100.0  # closing kisebb -> jó CLV (pozitív)
            else:
                clv = None

            con2 = db()
            cur2 = con2.cursor()
            cur2.execute(
                """
                UPDATE predictions
                SET closing_odds=?, clv_percent=?
                WHERE id=?
                """,
                (float(best), (float(clv) if clv is not None else None), int(row["id"])),
            )
            con2.commit()
            con2.close()
            updated += 1

    return updated, ""


# =========================================================
#  FŐ ELEMZÉS
# =========================================================
def run_analysis(leagues: list[str], window_hours: int, min_odds: float, max_odds: float, regions: str, debug: bool) -> dict:
    updated_results = refresh_past_results()

    candidates = []
    now = now_utc()
    limit = now + timedelta(hours=int(window_hours))

    debug_rows = []
    quota_info = {"remaining": None, "used": None, "last": None}

    for lg in leagues:
        resp = odds_api_get(lg, REQUEST_MARKETS, regions=regions)
        headers = resp.get("headers", {}) or {}
        if headers:
            quota_info["remaining"] = headers.get("x-requests-remaining") or quota_info["remaining"]
            quota_info["used"] = headers.get("x-requests-used") or quota_info["used"]
            quota_info["last"] = headers.get("x-requests-last") or quota_info["last"]

        if not resp.get("ok"):
            if debug:
                debug_rows.append((lg, resp.get("status", "ERR"), 0, (resp.get("detail", "") or "")[:220]))
            continue

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

                try:
                    mid = fd_find_match_id(c["home"], c["away"], c["kickoff"])
                except Exception:
                    mid = None
                c["football_data_match_id"] = mid

                candidates.append(c)

            time.sleep(0.02)  # óvatos quota/terhelés

    candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    ticket, total_odds = pick_best_duo(candidates)

    return {
        "updated_results": updated_results,
        "candidates": candidates,
        "ticket": ticket,
        "total_odds": total_odds,
        "debug_rows": debug_rows,
        "quota_info": quota_info,
    }


# =========================================================
#  UI – Modern neon dashboard (Streamlit limitations mellett)
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@300;400;600;700&display=swap');
:root{
  --bg1:#070a1a; --bg2:#0f1633;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.72);
  --cyan:#00d4ff; --vio:#7b2cbf; --pink:#ff006e; --lime:#5CFF7A;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp{
  background:
    radial-gradient(1200px 600px at 15% 10%, rgba(0,212,255,0.18), transparent 60%),
    radial-gradient(900px 500px at 85% 20%, rgba(255,0,110,0.16), transparent 55%),
    linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 55%, var(--bg1) 100%);
}
.hdr{
  font-family:'Orbitron', sans-serif;
  letter-spacing: 0.5px;
  font-weight: 900;
  font-size: 2.35rem;
  background: linear-gradient(90deg, var(--cyan), var(--vio), var(--pink));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin: 0.2rem 0 0.2rem 0;
}
.subhdr{ color: var(--muted); margin-bottom: 0.8rem; }
.panel{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px 10px 16px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.42);
}
.card{
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  margin: 10px 0;
  box-shadow: 0 14px 45px rgba(0,0,0,0.40);
}
.badge{
  display:inline-flex; align-items:center; gap:8px;
  padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(0,212,255,0.35);
  background: rgba(123,44,191,0.20);
  color: rgba(255,255,255,0.90);
  font-size: 0.86rem;
}
.badge2{
  display:inline-flex; align-items:center; gap:8px;
  padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(255,0,110,0.45);
  background: rgba(255,0,110,0.14);
  color: rgba(255,255,255,0.90);
  font-size: 0.86rem;
}
.small{ color: var(--muted); font-size: 0.9rem; }
hr{ border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr">⚽ TITAN – Strategic Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='subhdr'>Valós meccsek (football-data.org) + valós odds (The Odds API). Tipp <b>csak odds-szal</b>. Hibát nem dobunk: állapotot jelzünk.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    DEBUG = st.toggle("🔎 Debug (státusz ligánként)", value=True)

    leagues = st.multiselect("Ligák (Odds API kulcsok)", DEFAULT_LEAGUES, default=DEFAULT_LEAGUES)
    window_hours = st.slider("Időablak (óra)", min_value=12, max_value=168, value=24, step=12)
    min_odds = st.number_input("Min odds / tipp", value=1.25, step=0.01, format="%.2f")
    max_odds = st.number_input("Max odds / tipp", value=1.95, step=0.01, format="%.2f")
    regions = st.selectbox("Odds API régió", options=["eu", "uk", "eu,uk"], index=0)

    st.markdown("---")
    st.markdown("### 🔑 Kulcs státusz (gyors)")
    st.write(f"ODDS_API_KEY: {'✅' if ODDS_API_KEY else '❌'}")
    st.write(f"FOOTBALL_DATA_KEY: {'✅' if FOOTBALL_DATA_KEY else '❌'}")
    st.write(f"WEATHER_KEY: {'✅' if WEATHER_KEY else '—'}")
    st.write(f"NEWS_API_KEY: {'✅' if NEWS_API_KEY else '—'}")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    run_btn = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)
with colB:
    save_btn = st.button("💾 Két tipp mentése DB-be", use_container_width=True)
with colC:
    clv_btn = st.button("📉 CLV frissítés (pending)", use_container_width=True)

if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

if run_btn:
    with st.spinner("Futtatás: korábbi tippek settle + odds/score kalkuláció…"):
        res = run_analysis(leagues, window_hours, float(min_odds), float(max_odds), regions, DEBUG)
        st.session_state["last_run"] = res

        if res.get("updated_results", 0) > 0:
            st.success(f"✅ Korábbi tippek frissítve: {res['updated_results']} db lezárva.")
        else:
            st.info("ℹ️ Korábbi tippeknél nincs frissítés (vagy még nem értek véget).")

        qi = res.get("quota_info", {}) or {}
        if qi.get("remaining") or qi.get("used"):
            st.caption(
                f"Odds API quota: remaining={qi.get('remaining','?')} | used={qi.get('used','?')} | last_cost={qi.get('last','?')}"
            )

        if DEBUG:
            with st.expander("🔎 Debug: Odds API státusz ligánként", expanded=True):
                dbg = pd.DataFrame(res.get("debug_rows", []), columns=["league", "status", "events", "details"])
                st.dataframe(dbg, use_container_width=True)

if clv_btn:
    with st.spinner("CLV frissítés… (csak Odds API él esetén)"):
        upd, msg = update_clv_for_pending(regions=regions, markets=REQUEST_MARKETS)
        if msg:
            st.warning(msg)
        st.success(f"CLV frissítve: {upd} rekord.")

if st.session_state["last_run"] is not None:
    res = st.session_state["last_run"]
    ticket = res.get("ticket", [])
    total_odds = res.get("total_odds", 0.0)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("🎫 Napi TOP 2 (súlyozás után)")
    if not ticket:
        st.warning(
            "Nincs ajánlható TOP 2, mert **nincs elérhető odds** (pl. quota/401), vagy nincs a szűrőknek megfelelő piac.\n\n"
            "➡️ A rendszer nem hibázik: **odds nélkül nem ad tippet**."
        )
    else:
        st.markdown(
            f"**Össz-odds:** `{total_odds:.2f}`  "
            f"<span class='badge'>cél: ~{TARGET_TOTAL_ODDS:.2f}</span>",
            unsafe_allow_html=True,
        )

        for idx, t in enumerate(ticket, start=1):
            meta = t.get("meta", {}) or {}
            w = meta.get("weather", {}) or {}
            nh = meta.get("news_home", {}) or {}
            na = meta.get("news_away", {}) or {}

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{idx}  {t['match']}")
            st.markdown(
                f"<span class='small'>Liga:</span> <code>{t.get('league','—')}</code> "
                f"| <span class='small'>Kezdés:</span> <b>{fmt_dt_local(t.get('kickoff'))}</b>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='badge'>Piac: {t.get('bet_type')}</span> "
                f"<span class='badge'>Odds: {float(t.get('odds',0.0)):.2f}</span> "
                f"<span class='badge2'>Score: {float(t.get('score',0.0)):.0f}/100</span>",
                unsafe_allow_html=True,
            )

            if t.get("bet_type") == "H2H":
                st.write(f"**Tipp:** {t.get('selection')} (rendes játékidő)")
            elif t.get("bet_type") == "TOTALS":
                st.write(f"**Tipp:** {t.get('selection')} {t.get('line')}")
            elif t.get("bet_type") == "SPREADS":
                side = "Hazai" if t.get("selection") == "HOME" else "Vendég"
                st.write(f"**Tipp:** {side} {t.get('line')}")
            else:
                st.write(f"**Tipp:** {t.get('selection')}")

            st.markdown("**Miért ezt hozta ki a súlyozás:**")
            st.write(t.get("reasoning", ""))

            if w and (w.get("temp") is not None or w.get("desc")):
                st.caption(f"🌦️ Időjárás: {w.get('temp','?')}°C, {w.get('desc','—')}, szél: {w.get('wind','?')} m/s")

            if (nh.get("lines") or na.get("lines")):
                with st.expander("📰 Friss hírcímek (forrással)", expanded=False):
                    st.write(f"**{t.get('home','')}**")
                    for line in (nh.get("lines") or []):
                        st.write(line)
                    st.write(f"**{t.get('away','')}**")
                    for line in (na.get("lines") or []):
                        st.write(line)

            st.caption(f"football-data match_id: {t.get('football_data_match_id')}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if save_btn:
    if st.session_state["last_run"] is None or not st.session_state["last_run"].get("ticket"):
        st.warning("Előbb futtasd az elemzést, hogy legyen TOP 2 (odds szükséges).")
    else:
        save_ticket(st.session_state["last_run"]["ticket"])
        st.success("✅ A TOP 2 mentve az adatbázisba (opening_odds rögzítve).")

st.markdown("---")

st.subheader("📅 Valós meccsek az időablakban (football-data.org)")
fixtures, fx_err = fd_fixtures_window(hours_ahead=int(window_hours))
if fx_err:
    st.warning(fx_err)
elif not fixtures:
    st.info("Nincs meccs a megadott időablakban a football-data szerint.")
else:
    fx_df = pd.DataFrame(
        [
            {
                "Kezdés (helyi)": fmt_dt_local(x["kickoff_utc"]),
                "Liga": x["competition"],
                "Meccs": f"{x['home']} vs {x['away']}",
                "Státusz": x["status"],
                "match_id": x["match_id"],
            }
            for x in fixtures[:250]
        ]
    )
    st.dataframe(fx_df, use_container_width=True)

st.markdown("---")

st.subheader("📜 Előzmények + statisztika")
con = db()
df = pd.read_sql_query(
    """
    SELECT id, created_at, match, league, kickoff_utc,
           bet_type, selection, line, odds, opening_odds, closing_odds, clv_percent,
           score, result
    FROM predictions
    ORDER BY id DESC
    LIMIT 400
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

