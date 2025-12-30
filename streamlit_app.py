import streamlit as st
import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import time
import re
from math import sqrt

# =========================================================
#  KONFIG
# =========================================================
st.set_page_config(page_title="⚽ TITAN – Strategic Intelligence", layout="wide", page_icon="⚽")

try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    FOOTBALL_DATA_KEY = st.secrets["FOOTBALL_DATA_KEY"]  # X-Auth-Token
except Exception:
    st.error("⚠️ Hiányzó kulcs(ok) a Streamlit Secrets-ben: ODDS_API_KEY, WEATHER_API_KEY, NEWS_API_KEY, FOOTBALL_DATA_KEY")
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
    return sqlite3.connect(DB_PATH, check_same_thread=False)

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

        bet_type TEXT,          -- pl. "H2H", "TOTALS", "SPREADS"
        market_key TEXT,        -- "h2h" / "totals" / "spreads"
        selection TEXT,         -- pl. "Arsenal" vagy "Over" vagy "Under"
        line REAL,              -- pl. 2.5 totalsnál, spreadnél -1.0
        bookmaker TEXT,
        odds REAL,

        score REAL,
        reasoning TEXT,         -- magyar indoklás röviden (pár mondat)

        football_data_match_id INTEGER,  -- ha megtaláltuk
        result TEXT DEFAULT 'PENDING',   -- PENDING / WON / LOST / VOID / UNKNOWN
        settled_at TEXT,
        home_goals INTEGER,
        away_goals INTEGER
    )
    """)
    con.commit()
    con.close()

init_db()

# =========================================================
#  SEGÉDFÜGGVÉNYEK – normalizálás / összeillesztés
# =========================================================
def now_utc():
    return datetime.now(timezone.utc)

def iso_to_dt(s: str) -> datetime:
    # Odds API: "2025-01-01T20:00:00Z"
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def norm_team(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9áéíóöőúüű\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # gyakori rövidítések / eltérések finomítása
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
    a2, b2 = norm_team(a), norm_team(b)
    if a2 == b2:
        return 1.0
    # részleges egyezés
    if a2 in b2 or b2 in a2:
        return 0.7
    # token egyezés
    at = set(a2.split())
    bt = set(b2.split())
    inter = len(at & bt)
    union = max(1, len(at | bt))
    return inter / union

# =========================================================
#  FOOTBALL-DATA.ORG – meccsek/eredmények/frissítés
# =========================================================
def fd_headers():
    return {"X-Auth-Token": FOOTBALL_DATA_KEY}

def fd_get(url, params=None, timeout=15):
    r = requests.get(url, headers=fd_headers(), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fd_find_match_id(home: str, away: str, kickoff_utc: datetime) -> int | None:
    """
    Megpróbálja megtalálni a football-data meccs ID-t időablak + csapatnév illesztéssel.
    Megbízhatóság: közepes (a csapatnév-eltérések miatt), de stabilabb, mint web scrape.
    """
    # 1) Aznap környéke (kickoff napja)
    d = kickoff_utc.date().isoformat()
    url = "https://api.football-data.org/v4/matches"
    data = fd_get(url, params={"dateFrom": d, "dateTo": d})
    candidates = data.get("matches", [])

    best = (0.0, None)
    for m in candidates:
        try:
            fd_home = m["homeTeam"]["name"]
            fd_away = m["awayTeam"]["name"]
            fd_utc = iso_to_dt(m["utcDate"].replace("Z", "+00:00")) if "Z" in m["utcDate"] else datetime.fromisoformat(m["utcDate"])
        except Exception:
            continue

        # idő közelség (±6 óra)
        if abs((fd_utc - kickoff_utc).total_seconds()) > 6 * 3600:
            continue

        s = (team_match_score(home, fd_home) + team_match_score(away, fd_away)) / 2.0
        if s > best[0]:
            best = (s, m.get("id"))

    return best[1] if best[0] >= 0.55 else None

def fd_settle_prediction(pred_row: dict) -> dict:
    """
    Egy DB rekord alapján lekéri az eredményt és kiszámolja a WON/LOST státuszt.
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
    # csak akkor settle-ünk, ha befejezett
    if status not in ["FINISHED", "AWARDED"]:
        return {"result": "PENDING", "home_goals": None, "away_goals": None}

    score_ft = m.get("score", {}).get("fullTime", {})
    hg = score_ft.get("home")
    ag = score_ft.get("away")
    if hg is None or ag is None:
        return {"result": "UNKNOWN", "home_goals": None, "away_goals": None}

    # értékelés a bet típus szerint
    bet_type = pred_row.get("bet_type")
    selection = pred_row.get("selection")
    line = pred_row.get("line")

    # H2H: kiválasztott csapat nyer
    if bet_type == "H2H":
        # döntetlen = LOST (mert nem DNB)
        # melyik csapat volt a selection?
        if norm_team(selection) == norm_team(pred_row.get("home")):
            res = "WON" if hg > ag else "LOST"
        elif norm_team(selection) == norm_team(pred_row.get("away")):
            res = "WON" if ag > hg else "LOST"
        else:
            res = "UNKNOWN"

    # TOTALS: Over/Under line
    elif bet_type == "TOTALS":
        total = hg + ag
        if selection.lower() == "over":
            res = "WON" if total > float(line) else "LOST"
        elif selection.lower() == "under":
            res = "WON" if total < float(line) else "LOST"
        else:
            res = "UNKNOWN"

    # SPREADS: ázsiai jellegű spread (egyszerűsített: push = VOID)
    elif bet_type == "SPREADS":
        # selection = "HOME" vagy "AWAY"
        # line: pl. -1.0 home -1
        if selection.upper() == "HOME":
            adj = (hg + float(line)) - ag
        elif selection.upper() == "AWAY":
            # away line pl. +1.0 => ag+1 - hg
            adj = (ag + float(line)) - hg
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
    """
    Minden futtatáskor: a már elindult (kickoff+2h) tippeket megpróbálja lezárni.
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

        # csak ha már biztos vége lehetett (kickoff + 2 óra)
        if now < kickoff + timedelta(hours=2):
            continue

        settle = fd_settle_prediction(row.to_dict())
        if settle["result"] in ["PENDING"]:
            continue

        con = db()
        cur = con.cursor()
        cur.execute("""
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
        con.commit()
        con.close()
        updated += 1

    return updated

# =========================================================
#  KÜLSŐ ADAT – időjárás / hírek (rövid, magyar indoklás)
# =========================================================
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

def news_brief(team_name: str):
    """
    Nem ígérünk 100% név szerinti hiányzót (NewsAPI cikkcím/lead), viszont:
    - 1-2 friss cím + forrás
    - egyszerű “poz/neg” jelző kulcsszavak alapján
    """
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

def choose_best_bookmaker(bookmakers: list[dict]):
    # preferált sorrend
    prefer = ["pinnacle", "bet365", "unibet", "williamhill", "marathonbet"]
    by_key = {b.get("key"): b for b in bookmakers}
    for k in prefer:
        if k in by_key:
            return by_key[k]
    return bookmakers[0] if bookmakers else None

def extract_candidates_from_match(m: dict) -> list[dict]:
    """
    Egy meccsből bet jelölteket csinál:
    - H2H: favorit (legalacsonyabb odds) kiválasztása
    - TOTALS: ha van 2.5 line, Over/Under a “normális” odds sávban
    - SPREADS: ha van -1 / +1 közeli line, home/away
    """
    out = []
    home = m["home_team"]
    away = m["away_team"]
    kickoff = iso_to_dt(m["commence_time"].replace("Z", "+00:00"))

    bookie = choose_best_bookmaker(m.get("bookmakers", []))
    if not bookie:
        return out

    markets = bookie.get("markets", [])
    mk_by_key = {x.get("key"): x for x in markets}

    # -------- H2H --------
    if "h2h" in mk_by_key:
        outcomes = mk_by_key["h2h"].get("outcomes", [])
        # 3-way esetén döntetlen is lehet; mi csapat favoritot keressük
        team_outcomes = [o for o in outcomes if o.get("name") in [home, away]]
        if team_outcomes:
            fav = min(team_outcomes, key=lambda o: float(o.get("price", 999)))
            odds = float(fav["price"])
            if MIN_LEG_ODDS <= odds <= MAX_LEG_ODDS:
                out.append({
                    "match": f"{home} vs {away}",
                    "home": home,
                    "away": away,
                    "league": None,
                    "kickoff": kickoff,
                    "bet_type": "H2H",
                    "market_key": "h2h",
                    "selection": fav["name"],
                    "line": None,
                    "bookmaker": bookie.get("key", "book"),
                    "odds": odds,
                })

    # -------- TOTALS --------
    if "totals" in mk_by_key:
        outcomes = mk_by_key["totals"].get("outcomes", [])
        # TheOddsAPI totals: outcome has name Over/Under + point
        # pl. {"name":"Over","price":1.85,"point":2.5}
        for target_line in [2.5, 3.5, 1.5]:
            cand = [o for o in outcomes if float(o.get("point", -999)) == float(target_line)]
            if cand:
                for o in cand:
                    name = (o.get("name") or "").strip()
                    odds = float(o.get("price"))
                    if name.lower() in ["over", "under"] and MIN_LEG_ODDS <= odds <= MAX_LEG_ODDS:
                        out.append({
                            "match": f"{home} vs {away}",
                            "home": home,
                            "away": away,
                            "league": None,
                            "kickoff": kickoff,
                            "bet_type": "TOTALS",
                            "market_key": "totals",
                            "selection": name.capitalize(),  # Over/Under
                            "line": float(target_line),
                            "bookmaker": bookie.get("key", "book"),
                            "odds": odds,
                        })
                break  # csak egy preferált line-t vegyünk (2.5 előny)

    # -------- SPREADS --------
    if "spreads" in mk_by_key:
        outcomes = mk_by_key["spreads"].get("outcomes", [])
        # spreads outcome: name = team, point = handicap
        # cél: -1, -0.5, +0.5, +1 környék (a “biztonságos” tartomány)
        preferred_points = [-1.0, -0.5, 0.5, 1.0]
        for p in preferred_points:
            cand = [o for o in outcomes if float(o.get("point", -999)) == float(p)]
            if cand:
                for o in cand:
                    team_name = o.get("name")
                    odds = float(o.get("price"))
                    if MIN_LEG_ODDS <= odds <= MAX_LEG_ODDS:
                        # selection = HOME/AWAY a settle logika miatt
                        sel = "HOME" if team_match_score(team_name, home) >= team_match_score(team_name, away) else "AWAY"
                        out.append({
                            "match": f"{home} vs {away}",
                            "home": home,
                            "away": away,
                            "league": None,
                            "kickoff": kickoff,
                            "bet_type": "SPREADS",
                            "market_key": "spreads",
                            "selection": sel,          # HOME / AWAY
                            "line": float(p),          # handicap
                            "bookmaker": bookie.get("key", "book"),
                            "odds": odds,
                        })
                break

    return out

# =========================================================
#  PONTOZÁS + INDOKLÁS
# =========================================================
def score_candidate(c: dict) -> tuple[float, str, dict]:
    """
    Score (0-100) + magyar indoklás (pár mondat).
    """
    odds = float(c["odds"])

    # odds-komponens: minél közelebb a TARGET_LEG_ODDS-hoz, annál jobb
    # abs diff 0.0 -> +30, diff 0.6 -> ~0
    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))

    # időjárás: egyszerű kockázat (szél/eső)
    city_guess = (c["home"].split()[-1] if c["home"] else "London")
    w = get_weather_basic(city_guess)
    weather_pen = 0.0
    if w["wind"] is not None and w["wind"] >= 12:
        weather_pen -= 6
    if isinstance(w["desc"], str) and any(x in w["desc"].lower() for x in ["eső", "zápor", "vihar"]):
        weather_pen -= 4

    # hírek: team szintű jelző
    # (nem ígérünk 100% név szerinti hiányzót; forrás-címeket adunk)
    news_home = news_brief(c["home"])
    time.sleep(0.2)
    news_away = news_brief(c["away"])

    # Ha H2H és a pick az egyik csapat: annak a hír-score-ja számít jobban
    news_bias = 0
    if c["bet_type"] == "H2H":
        if team_match_score(c["selection"], c["home"]) >= 0.7:
            news_bias = news_home["score"]
        else:
            news_bias = news_away["score"]
    else:
        # totals/spreads: mindkettő számít kicsit
        news_bias = news_home["score"] + news_away["score"]

    news_score = float(news_bias) * 6.0  # -2..+2 környék -> -12..+12

    # alap
    raw = 50.0 + odds_score + news_score + weather_pen
    final = max(0.0, min(100.0, raw))

    # indoklás röviden (pár mondat)
    bet_label = ""
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
    why.append(f"Az odds **{odds:.2f}**, ami jól illeszkedik a duplázó (~2.00) célhoz.")
    if news_bias > 0:
        why.append("A friss hírek összképe inkább **pozitív** a választás szempontjából.")
    elif news_bias < 0:
        why.append("A friss hírekben van **kockázati jel** (sérülés/hiányzás gyanú), ezért óvatosabb.")
    else:
        why.append("A hírek alapján nincs egyértelmű extra kockázat vagy boost.")

    if w["temp"] is not None:
        why.append(f"Időjárás: {w['temp']:.0f}°C, {w['desc']} (szél: {w['wind'] if w['wind'] is not None else '?'} m/s).")

    reasoning = bet_label + "\n\n" + " ".join(why)

    meta = {"weather": w, "news_home": news_home, "news_away": news_away}
    return final, reasoning, meta

# =========================================================
#  DUÓ KIVÁLASZTÁS (össz-odds ~ 2.00)
# =========================================================
def pick_best_duo(cands: list[dict]) -> tuple[list[dict], float]:
    """
    Két tipp kiválasztása:
    - össz-odds a [TOTAL_ODDS_MIN, TOTAL_ODDS_MAX] tartományba essen, és minél közelebb a TARGET_TOTAL_ODDS-hoz
    - pontszám maximalizálás
    """
    if len(cands) < 2:
        return [], 0.0

    best = (None, None, -1e9, 0.0)  # i, j, utility, total_odds
    n = len(cands)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]

            # ne legyen ugyanaz a meccs (diverzifikáció)
            if a["match"] == b["match"]:
                continue

            total_odds = float(a["odds"]) * float(b["odds"])
            if not (TOTAL_ODDS_MIN <= total_odds <= TOTAL_ODDS_MAX):
                continue

            # utility: pontszámok + odds közelség bónusz
            closeness = 1.0 - min(1.0, abs(total_odds - TARGET_TOTAL_ODDS) / 0.15)  # 0..1
            utility = float(a["score"]) + float(b["score"]) + 20.0 * closeness

            if utility > best[2]:
                best = (i, j, utility, total_odds)

    if best[0] is None:
        # fallback: legjobb 2 score, akkor is, ha nincs 2.00 körül
        top2 = sorted(cands, key=lambda x: x["score"], reverse=True)[:2]
        return top2, float(top2[0]["odds"]) * float(top2[1]["odds"])

    return [cands[best[0]], cands[best[1]]], best[3]

# =========================================================
#  FŐ ELEMZÉS
# =========================================================
def run_analysis(leagues: list[str]) -> dict:
    # 1) múlt frissítése
    updated = refresh_past_results()

    # 2) jelöltek gyűjtése
    candidates = []
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
                kickoff = iso_to_dt(m["commence_time"].replace("Z", "+00:00"))
            except Exception:
                continue

            if not (now <= kickoff <= limit):
                continue

            cands = extract_candidates_from_match(m)
            for c in cands:
                c["league"] = lg
                # score + reasoning
                sc, reason, meta = score_candidate(c)
                c["score"] = sc
                c["reasoning"] = reason
                c["meta"] = meta
                candidates.append(c)

            time.sleep(0.05)

    # 3) rendezés
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # 4) duó kiválasztás 2.00 körül
    ticket, total_odds = pick_best_duo(candidates)

    # 5) football-data match id (a ticketre)
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
        "total_odds": total_odds
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
             score, reasoning, football_data_match_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
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
            t.get("football_data_match_id")
        ))
    con.commit()
    con.close()

# =========================================================
#  UI (magyar)
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

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Összes tipp", total)
        st.metric("Találat %", f"{hit:.0f}%")
    with c2:
        st.metric("Nyert", won)
        st.metric("Vesztett", lost)

    st.caption(f"VOID: {void} | PENDING: {pending}")

colA, colB = st.columns([1,1])
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

            st.markdown(
                f"**Piac:** `{t['bet_type']}`  |  **Odds:** `{t['odds']:.2f}`  |  **Score:** `{t['score']:.0f}/100`"
            )

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

            # rövid indoklás pár mondatban
            st.markdown("**Miért ezt ajánlja:**")
            st.write(t["reasoning"])

            # időjárás
            if w:
                st.caption(f"🌦️ Időjárás (város tipp): {w.get('temp','?')}°C, {w.get('desc','?')}, szél: {w.get('wind','?')} m/s")

            # hírek (forráscímek)
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
df = pd.read_sql_query("""
    SELECT id, created_at, match, league, kickoff_utc, bet_type, selection, line, odds, score, result, home_goals, away_goals
    FROM predictions
    ORDER BY id DESC
    LIMIT 400
""", con)
con.close()

st.dataframe(df, use_container_width=True)

# Egyszerű összegző grafikon
if not df.empty:
    df2 = df.copy()
    df2["odds"] = pd.to_numeric(df2["odds"], errors="coerce")
    df2["score"] = pd.to_numeric(df2["score"], errors="coerce")
    st.caption("Megjegyzés: a találati arányt csak a lezárt (WON/LOST) tippekre számoljuk, VOID/UNKNOWN nélkül.")
    decided = df2[df2["result"].isin(["WON", "LOST"])]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Lezárt tippek", len(decided))
    with c2:
        st.metric("Átlag odds", f"{decided['odds'].mean():.2f}" if len(decided) else "—")
    with c3:
        hit = (decided["result"].eq("WON").mean() * 100.0) if len(decided) else 0.0
        st.metric("Találat %", f"{hit:.0f}%")
