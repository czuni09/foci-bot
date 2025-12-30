"""
TITAN – Strategic Intelligence
Odds + hírek + időjárás alapú jelöltek, duplázó ~2.00 cél, DB naplózás + CLV.
ENV: ODDS_API_KEY, WEATHER_API_KEY, NEWS_API_KEY, FOOTBALL_DATA_KEY
"""

import streamlit as st
import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import re
from math import sqrt
from difflib import SequenceMatcher
import os

# =========================================================
#  KONFIG
# =========================================================
st.set_page_config(page_title="⚽ TITAN – Strategic Intelligence", layout="wide", page_icon="⚽")

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")

missing = []
if not ODDS_API_KEY: missing.append("ODDS_API_KEY")
if not WEATHER_KEY: missing.append("WEATHER_API_KEY")
if not NEWS_API_KEY: missing.append("NEWS_API_KEY")
if not FOOTBALL_DATA_KEY: missing.append("FOOTBALL_DATA_KEY")

if missing:
    st.error(f"⚠️ Hiányzó környezeti változók: {', '.join(missing)}")
    st.stop()

DB_PATH = "titan.db"

# Duplázó cél
TARGET_TOTAL_ODDS = 2.00
TOTAL_ODDS_MIN = 1.90
TOTAL_ODDS_MAX = 2.10

TARGET_LEG_ODDS = sqrt(2)

# Ablak + odds szűrés
WINDOW_HOURS = 24
MIN_LEG_ODDS = 1.25
MAX_LEG_ODDS = 1.95

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

# Debug kapcsoló
with st.sidebar:
    DEBUG = st.toggle("🔎 Debug mód (hiba okok kiírása)", value=True)

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
        away_goals INTEGER,

        closing_odds REAL,
        clv_percent REAL
    )
    """)
    # Régi DB esetén ALTER (ha nem létezne)
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
def now_utc():
    return datetime.now(timezone.utc)

def iso_to_dt(s: str) -> datetime:
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
    url = "https://api.football-data.org/v4/matches"
    try:
        data = fd_get(url, params={"dateFrom": date_from, "dateTo": date_to})
    except Exception:
        return None

    best = (0.0, None)
    for m in data.get("matches", []) or []:
        try:
            fd_home = m["homeTeam"]["name"]
            fd_away = m["awayTeam"]["name"]
            fd_utc = iso_to_dt(m["utcDate"])
        except Exception:
            continue
        if abs((fd_utc - kickoff_utc).total_seconds()) > 8 * 3600:
            continue
        s = (team_match_score(home, fd_home) + team_match_score(away, fd_away)) / 2.0
        if s > best[0]:
            best = (s, m.get("id"))
    return best[1] if best[0] >= 0.55 else None

def fd_settle_prediction(pred_row: dict) -> dict:
    match_id = pred_row.get("football_data_match_id")
    if not match_id:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    url = f"https://api.football-data.org/v4/matches/{match_id}"
    try:
        m = fd_get(url)
    except Exception:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    status = m.get("status", "")
    if status not in ["FINISHED", "AWARDED"]:
        return {"result": "PENDING", "home_goals": None, "away_goals": None}

    score_ft = (m.get("score", {}) or {}).get("fullTime", {}) or {}
    hg = score_ft.get("home")
    ag = score_ft.get("away")
    if hg is None or ag is None:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    bet_type = pred_row.get("bet_type")
    selection = pred_row.get("selection")
    line = pred_row.get("line")

    if bet_type == "H2H":
        if norm_team(selection) == norm_team(pred_row.get("home")):
            res = "WON" if hg > ag else "LOST"
        elif norm_team(selection) == norm_team(pred_row.get("away")):
            res = "WON" if ag > hg else "LOST"
        else:
            res = "UNKNOWN"

    elif bet_type == "TOTALS":
        total = hg + ag
        if (selection or "").lower() == "over":
            res = "WON" if total > float(line) else ("VOID" if total == float(line) else "LOST")
        elif (selection or "").lower() == "under":
            res = "WON" if total < float(line) else ("VOID" if total == float(line) else "LOST")
        else:
            res = "UNKNOWN"

    elif bet_type == "SPREADS":
        try:
            ln = float(line)
        except Exception:
            ln = None
        if ln is None:
            return {"result": "UNKNOWN", "home_goals": int(hg), "away_goals": int(ag)}
        if (selection or "").upper() == "HOME":
            adj = (hg + ln) - ag
        elif (selection or "").upper() == "AWAY":
            adj = (ag + ln) - hg
        else:
            adj = None
        if adj is None:
            res = "UNKNOWN"
        else:
            res = "WON" if adj > 0 else ("VOID" if adj == 0 else "LOST")
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
        try:
            kickoff = iso_to_dt(row["kickoff_utc"])
        except Exception:
            continue

        if kickoff is None or now < kickoff + timedelta(hours=2):
            continue

        settle = fd_settle_prediction(row.to_dict())
        if settle["result"] == "PENDING":
            continue

        con2 = db()
        cur2 = con2.cursor()
        cur2.execute("""
            UPDATE predictions
            SET result=?, settled_at=?, home_goals=?, away_goals=?
            WHERE id=?
        """, (
            settle["result"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            settle["home_goals"],
            settle["away_goals"],
            int(row["id"])
        ))
        con2.commit()
        con2.close()
        updated += 1

    return updated

# =========================================================
#  CLOSING ODDS / CLV
#  Megjegyzés: te mindig kezdés előtt fogadsz -> placed_odds a mentett odds.
#  A "closing_odds"-t a meccs közelében (90p-10p) próbáljuk elkapni.
# =========================================================
def update_closing_odds():
    con = db()
    df = pd.read_sql_query("""
        SELECT * FROM predictions
        WHERE (closing_odds IS NULL OR clv_percent IS NULL)
          AND result IN ('PENDING','UNKNOWN')
    """, con)
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
        if kickoff is None:
            continue

        # csak meccs közelében (closing line)
        if not (kickoff - timedelta(minutes=90) <= now <= kickoff + timedelta(minutes=10)):
            continue

        lg = row.get("league")
        if not lg:
            continue

        try:
            data = odds_api_get(lg, REQUEST_MARKETS)
        except Exception:
            continue

        # meccs azonosítás odds feedből
        best_match = None
        best_score = 0.0
        for m in data if isinstance(data, list) else []:
            mh = m.get("home_team")
            ma = m.get("away_team")
            mt = iso_to_dt(m.get("commence_time"))
            if not mh or not ma or not mt:
                continue
            if abs((mt - kickoff).total_seconds()) > 3 * 3600:
                continue
            s = (team_match_score(row["home"], mh) + team_match_score(row["away"], ma)) / 2.0
            if s > best_score:
                best_score = s
                best_match = m
        if best_match is None or best_score < 0.55:
            continue

        bet_type = row.get("bet_type")
        selection = row.get("selection")
        line = row.get("line")

        closing_odds = None

        # segéd: odds kinyerése az összes bookie-ból -> max (bettor szempontból)
        def max_price(prices):
            return max(prices) if prices else None

        if bet_type == "H2H":
            prices = []
            for b in best_match.get("bookmakers", []) or []:
                for mk in b.get("markets", []) or []:
                    if mk.get("key") != "h2h":
                        continue
                    for o in mk.get("outcomes", []) or []:
                        name = o.get("name")
                        if name and team_match_score(name, selection) >= 0.7:
                            try:
                                prices.append(float(o.get("price")))
                            except Exception:
                                pass
            closing_odds = max_price(prices)

        elif bet_type == "TOTALS":
            try:
                target_line = float(line)
            except Exception:
                target_line = None
            prices = []
            if target_line is not None:
                for b in best_match.get("bookmakers", []) or []:
                    for mk in b.get("markets", []) or []:
                        if mk.get("key") != "totals":
                            continue
                        for o in mk.get("outcomes", []) or []:
                            try:
                                pt = float(o.get("point"))
                                pr = float(o.get("price"))
                            except Exception:
                                continue
                            nm = (o.get("name") or "").capitalize()
                            if abs(pt - target_line) < 1e-6 and nm == selection:
                                prices.append(pr)
            closing_odds = max_price(prices)

        elif bet_type == "SPREADS":
            try:
                target_line = float(line)
            except Exception:
                target_line = None
            prices = []
            if target_line is not None:
                for b in best_match.get("bookmakers", []) or []:
                    for mk in b.get("markets", []) or []:
                        if mk.get("key") != "spreads":
                            continue
                        for o in mk.get("outcomes", []) or []:
                            try:
                                pt = float(o.get("point"))
                                pr = float(o.get("price"))
                            except Exception:
                                continue
                            if abs(pt - target_line) > 1e-6:
                                continue
                            team_name = o.get("name")
                            if not team_name:
                                continue
                            is_home = team_match_score(team_name, row["home"]) >= team_match_score(team_name, row["away"])
                            sel_code = "HOME" if is_home else "AWAY"
                            if sel_code == selection:
                                prices.append(pr)
            closing_odds = max_price(prices)

        if closing_odds is not None:
            try:
                placed_odds = float(row["odds"])
                clv = (closing_odds / placed_odds) - 1.0 if placed_odds > 0 else None
            except Exception:
                clv = None

            con2 = db()
            cur2 = con2.cursor()
            cur2.execute(
                "UPDATE predictions SET closing_odds=?, clv_percent=? WHERE id=?",
                (closing_odds, clv, int(row["id"]))
            )
            con2.commit()
            con2.close()
            updated += 1

    return updated

# =========================================================
#  Cache-elt külső adatok
# =========================================================
@st.cache_data(ttl=900)
def get_weather_basic(city_guess="London"):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_guess, "appid": WEATHER_KEY, "units": "metric", "lang": "hu"}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        return {
            "temp": float(data["main"]["temp"]),
            "desc": data["weather"][0]["description"],
            "wind": float(data["wind"]["speed"]),
        }
    except Exception:
        return {"temp": None, "desc": "ismeretlen", "wind": None}

@st.cache_data(ttl=900)
def news_brief(team_name: str):
    try:
        url = "https://newsapi.org/v2/everything"
        q = f'"{team_name}" (injury OR injured OR out OR doubt OR suspended OR return OR fit OR lineup)'
        params = {
            "q": q,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 3,
            "apiKey": NEWS_API_KEY
        }
        r = requests.get(url, params=params, timeout=10)
        js = r.json()
        arts = js.get("articles", []) or []
        if not arts:
            return {"score": 0, "lines": []}

        lines, score = [], 0
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
#  ODDS API – fontos: itt NEM nyeljük el csendben a hibát
# =========================================================
@st.cache_data(ttl=120)
def odds_api_get(league_key: str, markets: list[str]):
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
    out = []
    home = m.get("home_team")
    away = m.get("away_team")
    kickoff = iso_to_dt(m.get("commence_time"))
    if not home or not away or not kickoff:
        return out

    bookmakers = m.get("bookmakers", []) or []

    def avg_best(prices):
        if not prices:
            return None, None
        avg = sum(prices) / len(prices)
        best = max(prices)  # bettor: nagyobb odds jobb
        return avg, best

    # ---- H2H ----
    h2h_prices = {}
    for b in bookmakers:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "h2h":
                continue
            for o in mk.get("outcomes", []) or []:
                nm = o.get("name")
                if nm not in [home, away]:
                    continue
                try:
                    pr = float(o.get("price"))
                except Exception:
                    continue
                h2h_prices.setdefault(nm, []).append(pr)

    if h2h_prices:
        avg_map = {tm: sum(ps)/len(ps) for tm, ps in h2h_prices.items() if ps}
        if avg_map:
            fav = min(avg_map, key=avg_map.get)  # favorit = legalacsonyabb átlag odds
            avg_o, best_o = avg_best(h2h_prices.get(fav, []))
            if avg_o and best_o and MIN_LEG_ODDS <= best_o <= MAX_LEG_ODDS:
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append({
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
                })

    # ---- TOTALS ----
    totals = {}
    for b in bookmakers:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "totals":
                continue
            for o in mk.get("outcomes", []) or []:
                nm = (o.get("name") or "").capitalize()
                if nm.lower() not in ["over", "under"]:
                    continue
                try:
                    pt = float(o.get("point"))
                    pr = float(o.get("price"))
                except Exception:
                    continue
                totals.setdefault((pt, nm), []).append(pr)

    for target_line in [2.5, 3.5, 1.5]:
        hit_any = False
        for nm in ["Over", "Under"]:
            ps = totals.get((target_line, nm), [])
            if not ps:
                continue
            avg_o, best_o = avg_best(ps)
            if avg_o and best_o and MIN_LEG_ODDS <= best_o <= MAX_LEG_ODDS:
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append({
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
                })
                hit_any = True
        if hit_any:
            break

    # ---- SPREADS ----
    spreads = {}
    for b in bookmakers:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "spreads":
                continue
            for o in mk.get("outcomes", []) or []:
                team_nm = o.get("name")
                try:
                    pt = float(o.get("point"))
                    pr = float(o.get("price"))
                except Exception:
                    continue
                spreads.setdefault((pt, team_nm), []).append(pr)

    preferred_points = [-1.0, -0.5, 0.5, 1.0]
    for p in preferred_points:
        keys = [k for k in spreads.keys() if abs(k[0] - p) < 1e-6]
        if not keys:
            continue
        for (pt, team_nm) in keys:
            ps = spreads.get((pt, team_nm), [])
            avg_o, best_o = avg_best(ps)
            if avg_o and best_o and MIN_LEG_ODDS <= best_o <= MAX_LEG_ODDS:
                sel = "HOME" if team_match_score(team_nm, home) >= team_match_score(team_nm, away) else "AWAY"
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append({
                    "match": f"{home} vs {away}",
                    "home": home,
                    "away": away,
                    "league": None,
                    "kickoff": kickoff,
                    "bet_type": "SPREADS",
                    "market_key": "spreads",
                    "selection": sel,   # HOME/AWAY
                    "line": float(p),
                    "bookmaker": "best_of",
                    "odds": best_o,
                    "avg_odds": avg_o,
                    "value_score": value,
                })
        break

    return out

# =========================================================
#  PONTOZÁS + INDOKLÁS
# =========================================================
def score_candidate(c: dict):
    odds = float(c["odds"])
    avg_odds = float(c.get("avg_odds", odds))
    value_score = float(c.get("value_score", 0.0))

    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))
    value_bonus = 20.0 * value_score

    city_guess = (c["home"].split()[-1] if c["home"] else "London")
    w = get_weather_basic(city_guess)

    weather_pen = 0.0
    if w["wind"] is not None and w["wind"] >= 12:
        weather_pen -= 6
    if isinstance(w["desc"], str) and any(x in w["desc"].lower() for x in ["eső", "zápor", "vihar"]):
        weather_pen -= 4

    news_home = news_brief(c["home"])
    time.sleep(0.15)
    news_away = news_brief(c["away"])

    if c["bet_type"] == "H2H":
        news_bias = news_home["score"] if team_match_score(c["selection"], c["home"]) >= 0.7 else news_away["score"]
    else:
        news_bias = news_home["score"] + news_away["score"]

    news_score = float(news_bias) * 6.0

    raw = 50.0 + odds_score + value_bonus + news_score + weather_pen
    final = max(0.0, min(100.0, raw))

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
    why.append(f"Odds: **{odds:.2f}** (átlag: {avg_odds:.2f}, value: {value_score*100:.1f}%).")
    if news_bias > 0:
        why.append("Hírek összképe: **pozitív**.")
    elif news_bias < 0:
        why.append("Hírekben van **kockázati jel** (sérülés/hiányzás).")
    else:
        why.append("Hírek alapján nincs egyértelmű extra jel.")
    if w["temp"] is not None:
        why.append(f"Időjárás (város tipp): {w['temp']:.0f}°C, {w['desc']} (szél: {w['wind'] if w['wind'] is not None else '?'} m/s).")

    reasoning = bet_label + "\n\n" + " ".join(why)
    meta = {"weather": w, "news_home": news_home, "news_away": news_away}
    return final, reasoning, meta

# =========================================================
#  DUÓ KIVÁLASZTÁS
# =========================================================
def pick_best_duo(cands: list[dict]):
    if len(cands) < 2:
        return [], 0.0

    best = (None, None, -1e9, 0.0)
    n = len(cands)

    for i in range(n):
        for j in range(i+1, n):
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
        return top2, float(top2[0]["odds"]) * float(top2[1]["odds"])

    return [cands[best[0]], cands[best[1]]], best[3]

# =========================================================
#  FŐ ELEMZÉS (diagnosztikával)
# =========================================================
def run_analysis(leagues: list[str]):
    # 0) CLV frissítés + 1) múlt lezárás
    clv_updated = update_closing_odds()
    updated = refresh_past_results()

    candidates = []
    now = now_utc()
    limit = now + timedelta(hours=WINDOW_HOURS)

    debug_rows = []  # league -> status

    for lg in leagues:
        try:
            data = odds_api_get(lg, REQUEST_MARKETS)
            if not isinstance(data, list):
                debug_rows.append((lg, "NEM LISTA válasz", 0))
                continue
            debug_rows.append((lg, "OK", len(data)))
        except requests.exceptions.HTTPError as e:
            txt = ""
            try:
                txt = (e.response.text or "")[:250]
            except Exception:
                pass
            debug_rows.append((lg, f"HTTPError: {e} | {txt}", 0))
            continue
        except Exception as e:
            debug_rows.append((lg, f"Error: {e}", 0))
            continue

        for m in data:
            try:
                kickoff = iso_to_dt(m.get("commence_time"))
            except Exception:
                continue
            if kickoff is None or not (now <= kickoff <= limit):
                continue

            cands = extract_candidates_from_match(m)
            for c in cands:
                c["league"] = lg
                sc, reason, meta = score_candidate(c)
                c["score"] = sc
                c["reasoning"] = reason
                c["meta"] = meta
                candidates.append(c)

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    ticket, total_odds = pick_best_duo(candidates)

    for t in ticket:
        try:
            t["football_data_match_id"] = fd_find_match_id(t["home"], t["away"], t["kickoff"])
        except Exception:
            t["football_data_match_id"] = None

    return {
        "updated_results": updated,
        "clv_updated": clv_updated,
        "candidates": candidates,
        "ticket": ticket,
        "total_odds": total_odds,
        "debug_rows": debug_rows,
    }

def save_ticket(ticket: list[dict]):
    if not ticket:
        return
    con = db()
    cur = con.cursor()
    for t in ticket:
        cur.execute("""
            INSERT INTO predictions
            (created_at, match, home, away, league, kickoff_utc,
             bet_type, market_key, selection, line, bookmaker, odds,
             score, reasoning, football_data_match_id, closing_odds, clv_percent)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            t["match"], t["home"], t["away"], t["league"], t["kickoff"].isoformat(),
            t["bet_type"], t["market_key"], t["selection"], t["line"], t["bookmaker"], float(t["odds"]),
            float(t["score"]), t["reasoning"], t.get("football_data_match_id"),
            None, None
        ))
    con.commit()
    con.close()

# =========================================================
#  UI
# =========================================================
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown('<div class="hdr">⚽ TITAN – Strategic Intelligence</div>', unsafe_allow_html=True)
st.caption("Manuális futtatás | 24 órán belüli meccsek | 2 tipp ~ 2.00 össz-odds | több piac (h2h/totals/spreads)")

with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    leagues = st.multiselect("Ligák", DEFAULT_LEAGUES, default=DEFAULT_LEAGUES)
    st.write(f"Piacok: `{', '.join(REQUEST_MARKETS)}`")
    st.write(f"Leg odds: {MIN_LEG_ODDS:.2f} – {MAX_LEG_ODDS:.2f}")
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

    decided_df = df_all[df_all["result"].isin(["WON","LOST"])] if total else pd.DataFrame()
    hit = (decided_df["result"].eq("WON").mean() * 100.0) if len(decided_df) else 0.0

    clv_mean = None
    if total and "clv_percent" in df_all.columns:
        tmp = df_all[df_all["result"].isin(["WON","LOST"])].copy()
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
    with st.spinner("Elemzés fut…"):
        res = run_analysis(leagues)
        st.session_state["last_run"] = res

        if DEBUG and res.get("debug_rows"):
            with st.expander("🔎 Debug: Odds API státusz ligánként", expanded=True):
                st.dataframe(pd.DataFrame(res["debug_rows"], columns=["league", "status", "events"]), use_container_width=True)

        if res["clv_updated"] > 0:
            st.success(f"✅ Closing odds/CLV frissítve: {res['clv_updated']} db.")
        if res["updated_results"] > 0:
            st.success(f"✅ Korábbi tippek lezárva: {res['updated_results']} db.")
        if res["clv_updated"] == 0 and res["updated_results"] == 0:
            st.info("ℹ️ Nincs frissítendő tipp/CLV (vagy nem a meccsek közelében futtattad).")

if st.session_state["last_run"] is not None:
    res = st.session_state["last_run"]
    ticket = res["ticket"]
    total_odds = res["total_odds"]

    st.subheader("🎫 Ajánlott duplázó (2 tipp)")
    if not ticket:
        st.warning("Nincs elég jelölt a 24 órás ablakban (VAGY az Odds API hibázott / kvóta / rossz kulcs / üres piac). Kapcsold be a Debug módot.")
    else:
        st.markdown(f"**Össz-odds:** `{total_odds:.2f}`  <span class='badge'>cél: ~{TARGET_TOTAL_ODDS:.2f}</span>", unsafe_allow_html=True)

        for idx, t in enumerate(ticket, start=1):
            kickoff_local = t["kickoff"].astimezone()
            meta = t.get("meta", {})
            w = meta.get("weather", {})
            nh = meta.get("news_home", {})
            na = meta.get("news_away", {})

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{idx}  {t['match']}")
            st.markdown(f"<span class='muted'>Liga:</span> `{t['league']}`  |  <span class='muted'>Kezdés:</span> **{kickoff_local.strftime('%Y.%m.%d %H:%M')}**", unsafe_allow_html=True)
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
df = pd.read_sql_query("""
    SELECT id, created_at, match, league, kickoff_utc,
           bet_type, selection, line, odds,
           closing_odds, clv_percent,
           score, result, home_goals, away_goals
    FROM predictions
    ORDER BY id DESC
    LIMIT 400
""", con)
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
