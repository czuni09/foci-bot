import os
import re
import time
import sqlite3
from math import sqrt
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple

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
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")  # itt most nem használjuk, de meghagyjuk

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
        con.execute("PRAGMA busy_timeout=30000;")
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

    # backward compatible: ha régi db van, oszlopok pótlása
    try:
        cur.execute("PRAGMA table_info(predictions)")
        cols = [r[1] for r in cur.fetchall()]
        if "closing_odds" not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN closing_odds REAL")
        if "clv_percent" not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN clv_percent REAL")
        if "data_quality" not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN data_quality TEXT")
    except Exception:
        pass

    con.commit()
    con.close()


init_db()


# =========================================================
#  SEGÉD
# =========================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
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


# =========================================================
#  Külső adatok (cache) – ha nincs kulcs, DEMO/0
# =========================================================
@st.cache_data(ttl=600)
def get_weather_basic(city_guess: str = "London") -> Dict[str, Any]:
    if not WEATHER_KEY:
        return {"temp": None, "desc": "nincs időjárás kulcs (DEMO)", "wind": None}
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
def news_brief(team_name: str) -> Dict[str, Any]:
    if not NEWS_API_KEY:
        return {"score": 0, "lines": ["• nincs NewsAPI kulcs (DEMO)"]}
    try:
        url = "https://newsapi.org/v2/everything"
        q = f'"{team_name}" (injury OR injured OR out OR doubt OR suspended OR return OR fit OR lineup)'
        params = {"q": q, "language": "en", "sortBy": "publishedAt", "pageSize": 3, "apiKey": NEWS_API_KEY}
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return {"score": 0, "lines": [f"• NewsAPI HTTP {r.status_code} (DEMO)"]}

        js = r.json()
        arts = js.get("articles", []) or []
        if not arts:
            return {"score": 0, "lines": []}

        lines: List[str] = []
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
#  ODDS API
# =========================================================
@st.cache_data(ttl=120)
def odds_api_get(league_key: str, markets: List[str], regions: str = "eu") -> List[Dict[str, Any]]:
    if not ODDS_API_KEY:
        raise requests.exceptions.HTTPError("Hiányzó ODDS_API_KEY")
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


# =========================================================
#  Jelöltek (LIVE)
# =========================================================
def extract_candidates_from_match(m: Dict[str, Any], min_odds: float, max_odds: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    home = m.get("home_team")
    away = m.get("away_team")
    kickoff = iso_to_dt(m.get("commence_time"))
    if not home or not away or not kickoff:
        return out

    bookmakers = m.get("bookmakers", []) or []

    def avg_best(prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not prices:
            return None, None
        avg = sum(prices) / len(prices)
        best = max(prices)
        return avg, best

    # ---- H2H: favorit (legalacsonyabb átlag), és arra a legjobb odds
    h2h_prices: Dict[str, List[float]] = {}
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
            fav = min(avg_map, key=avg_map.get)
            avg_o, best_o = avg_best(h2h_prices.get(fav, []))
            if avg_o and best_o and min_odds <= best_o <= max_odds:
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append({
                    "match": f"{home} vs {away}",
                    "home": home, "away": away,
                    "league": None, "kickoff": kickoff,
                    "bet_type": "H2H", "market_key": "h2h",
                    "selection": fav, "line": None,
                    "bookmaker": "best_of",
                    "odds": float(best_o),
                    "avg_odds": float(avg_o),
                    "value_score": float(value),
                    "data_quality": "LIVE",
                })

    # ---- TOTALS: 2.5 / 3.5 / 1.5
    totals: Dict[Tuple[float, str], List[float]] = {}
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
                totals.setdefault((float(pt), nm), []).append(float(pr))

    for target_line in (2.5, 3.5, 1.5):
        hit_any = False
        for nm in ("Over", "Under"):
            ps = totals.get((float(target_line), nm), [])
            if not ps:
                continue
            avg_o, best_o = avg_best(ps)
            if avg_o and best_o and min_odds <= best_o <= max_odds:
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append({
                    "match": f"{home} vs {away}",
                    "home": home, "away": away,
                    "league": None, "kickoff": kickoff,
                    "bet_type": "TOTALS", "market_key": "totals",
                    "selection": nm, "line": float(target_line),
                    "bookmaker": "best_of",
                    "odds": float(best_o),
                    "avg_odds": float(avg_o),
                    "value_score": float(value),
                    "data_quality": "LIVE",
                })
                hit_any = True
        if hit_any:
            break

    # ---- SPREADS: -1/-0.5/+0.5/+1
    spreads: Dict[Tuple[float, str], List[float]] = {}
    for b in bookmakers:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "spreads":
                continue
            for o in mk.get("outcomes", []) or []:
                team_nm = o.get("name")
                pt = safe_float(o.get("point"))
                pr = safe_float(o.get("price"))
                if team_nm and pt is not None and pr is not None:
                    spreads.setdefault((float(pt), team_nm), []).append(float(pr))

    for p in (-1.0, -0.5, 0.5, 1.0):
        keys = [k for k in spreads.keys() if abs(k[0] - float(p)) < 1e-9]
        if not keys:
            continue
        for (pt, team_nm) in keys:
            ps = spreads.get((pt, team_nm), [])
            avg_o, best_o = avg_best(ps)
            if avg_o and best_o and min_odds <= best_o <= max_odds:
                sel = "HOME" if team_match_score(team_nm, home) >= team_match_score(team_nm, away) else "AWAY"
                value = (best_o / avg_o) - 1.0 if avg_o else 0.0
                out.append({
                    "match": f"{home} vs {away}",
                    "home": home, "away": away,
                    "league": None, "kickoff": kickoff,
                    "bet_type": "SPREADS", "market_key": "spreads",
                    "selection": sel, "line": float(p),
                    "bookmaker": "best_of",
                    "odds": float(best_o),
                    "avg_odds": float(avg_o),
                    "value_score": float(value),
                    "data_quality": "LIVE",
                })
        break

    return out


# =========================================================
#  DEMO jelöltek – JAVÍTVA: külön meccsekből is ad 2.00 körüli ticketet
# =========================================================
def demo_candidates(now: datetime, leagues: List[str]) -> List[Dict[str, Any]]:
    # Biztos ticket: 1.42 * 1.41 = 2.0022 (tartományban)
    demo = [
        ("DEMO City", "Example United", "H2H", "DEMO City", None, 1.42),
        ("Sample FC", "Mock Town", "TOTALS", "Over", 2.5, 1.41),
        ("Alpha FC", "Beta FC", "SPREADS", "HOME", -0.5, 1.45),
        ("Northside", "Southside", "TOTALS", "Under", 3.5, 1.44),
    ]

    out: List[Dict[str, Any]] = []
    for i, (h, a, bt, sel, line, odds) in enumerate(demo):
        kickoff = now + timedelta(hours=2 + i * 2)
        lg = leagues[i % max(1, len(leagues))] if leagues else "soccer_epl"
        avg_odds = odds * 0.985  # “átlag” csak UI-hoz
        value_score = (odds / avg_odds) - 1.0

        mk = "h2h" if bt == "H2H" else ("totals" if bt == "TOTALS" else "spreads")
        out.append({
            "match": f"{h} vs {a}",
            "home": h, "away": a,
            "league": lg, "kickoff": kickoff,
            "bet_type": bt, "market_key": mk,
            "selection": sel, "line": line,
            "bookmaker": "DEMO",
            "odds": float(odds),
            "avg_odds": float(avg_odds),
            "value_score": float(value_score),
            "data_quality": "DEMO (NEM AJÁNLOTT)",
        })
    return out


# =========================================================
#  PONTOZÁS + INDOKLÁS
# =========================================================
def score_candidate(c: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    odds = safe_float(c.get("odds"), 0.0) or 0.0
    avg_odds = safe_float(c.get("avg_odds"), odds) or odds
    value_score = safe_float(c.get("value_score"), 0.0) or 0.0

    diff = abs(odds - TARGET_LEG_ODDS)
    odds_score = max(0.0, 30.0 * (1.0 - (diff / 0.6)))
    value_bonus = 20.0 * value_score

    is_demo = str(c.get("data_quality", "")).startswith("DEMO")
    demo_pen = -25.0 if is_demo else 0.0

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
        if team_match_score(str(c.get("selection", "")), str(c.get("home", ""))) >= 0.7:
            news_bias = int(news_home.get("score", 0))
        else:
            news_bias = int(news_away.get("score", 0))
    else:
        news_bias = int(news_home.get("score", 0)) + int(news_away.get("score", 0))

    news_score = float(news_bias) * 6.0

    raw = 50.0 + odds_score + value_bonus + news_score + weather_pen + demo_pen
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

    why: List[str] = []
    if is_demo:
        why.append("⚠️ **DEMO mód:** nincs élő odds adat (kvóta/hiba), ez UI/teszt – **nem ajánlott éles fogadásra**.")
    why.append(f"Odds: **{odds:.2f}** (átlag: {avg_odds:.2f}, value: {value_score*100:.1f}%).")

    if news_bias > 0:
        why.append("Hírek összképe: **pozitív**.")
    elif news_bias < 0:
        why.append("Hírekben van **kockázati jel** (sérülés/hiányzás).")
    else:
        why.append("Hírek alapján nincs egyértelmű extra jel.")

    if w.get("temp") is not None:
        why.append(f"Időjárás: {w['temp']:.0f}°C, {w.get('desc','?')} (szél: {w.get('wind','?')} m/s).")

    reasoning = bet_label + "\n\n" + " ".join(why)
    meta = {"weather": w, "news_home": news_home, "news_away": news_away}
    return float(final), reasoning, meta


# =========================================================
#  DUÓ KIVÁLASZTÁS – JAVÍTVA: ha nincs 2.00 körüli pár, akkor is ad 2 tippet
# =========================================================
def pick_best_duo(cands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    if len(cands) < 2:
        return [], 0.0

    # 1) Próbáljuk: külön meccs + 2.00 tartomány
    best_pair = None
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
                best_pair = (i, j)
                best_total = total_odds

    if best_pair is not None:
        a, b = cands[best_pair[0]], cands[best_pair[1]]
        return [a, b], float(best_total)

    # 2) Fallback: top2 score – akkor is, ha nem 2.00 körüli / akár ugyanaz a meccs
    top2 = sorted(cands, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:2]
    total_odds = float(top2[0].get("odds", 0.0)) * float(top2[1].get("odds", 0.0))
    return top2, float(total_odds)


# =========================================================
#  Mentés
# =========================================================
def save_ticket(ticket: List[Dict[str, Any]]):
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
#  FŐ ELEMZÉS – ha nincs LIVE adat (kvóta/401), DEMO-t ad, de ticket biztos lesz
# =========================================================
def run_analysis(
    leagues: List[str],
    window_hours: int,
    min_odds: float,
    max_odds: float,
    regions: str,
    debug: bool,
    force_demo_if_fail: bool
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    now = now_utc()
    limit = now + timedelta(hours=int(window_hours))

    debug_rows: List[Tuple[str, str, int, str]] = []
    any_live_ok = False

    for lg in leagues:
        try:
            data = odds_api_get(lg, REQUEST_MARKETS, regions=regions)
            any_live_ok = True
            debug_rows.append((lg, "OK", len(data), ""))
        except requests.exceptions.HTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", "")
            txt = ""
            try:
                txt = (e.response.text or "")[:300] if getattr(e, "response", None) is not None else str(e)[:300]
            except Exception:
                txt = str(e)[:300]
            debug_rows.append((lg, f"HTTPError {code}", 0, txt))
            continue
        except Exception as e:
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

            time.sleep(0.02)

    used_demo = False
    if (not candidates) and force_demo_if_fail:
        used_demo = True
        candidates = demo_candidates(now, leagues)
        for c in candidates:
            sc, reason, meta = score_candidate(c)
            c["score"] = sc
            c["reasoning"] = reason
            c["meta"] = meta

    candidates = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    ticket, total_odds = pick_best_duo(candidates)

    return {
        "candidates": candidates,
        "ticket": ticket,
        "total_odds": total_odds,
        "debug_rows": debug_rows,
        "used_demo": used_demo,
        "any_live_ok": any_live_ok,
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
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  background: rgba(123,44,191,0.25); border: 1px solid rgba(0,212,255,0.35);
  margin-left: 8px;
}
.badge_demo {
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  background: rgba(255,0,110,0.18); border: 1px solid rgba(255,0,110,0.45);
  margin-left: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr">⚽ TITAN – Strategic Intelligence</div>', unsafe_allow_html=True)
st.caption("Manuális futtatás | X órán belüli meccsek | 2 tipp | ha nincs LIVE adat: DEMO ticket (nem ajánlott)")

with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    DEBUG = st.toggle("🔎 Debug mód (ligánkénti státusz + hibák)", value=True)
    FORCE_DEMO = st.toggle("🧪 DEMO ticket automatikusan (ha nincs LIVE adat)", value=True)

    leagues = st.multiselect("Ligák", DEFAULT_LEAGUES, default=DEFAULT_LEAGUES)
    window_hours = st.slider("Időablak (óra)", min_value=12, max_value=168, value=24, step=12)
    min_odds = st.number_input("Min odds / tipp", value=1.25, step=0.01, format="%.2f")
    max_odds = st.number_input("Max odds / tipp", value=1.95, step=0.01, format="%.2f")
    regions = st.selectbox("Odds API régió", options=["eu", "uk", "eu,uk"], index=0)

colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)
with colB:
    save_btn = st.button("💾 Két tipp mentése DB-be", use_container_width=True)

if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

if run_btn:
    with st.spinner("Elemzés fut…"):
        res = run_analysis(leagues, window_hours, float(min_odds), float(max_odds), regions, DEBUG, FORCE_DEMO)
        st.session_state["last_run"] = res

        if res.get("used_demo"):
            st.warning("⚠️ DEMO ticket: nincs élő odds adat (kvóta/401/üres piac). UI/teszt – **nem ajánlott éles fogadásra**.")

        if DEBUG:
            with st.expander("🔎 Debug: Odds API státusz ligánként", expanded=True):
                dbg = pd.DataFrame(res["debug_rows"], columns=["league", "status", "events", "details"])
                st.dataframe(dbg, use_container_width=True)

if st.session_state["last_run"] is not None:
    res = st.session_state["last_run"]
    ticket = res["ticket"]
    total_odds = res["total_odds"]

    st.subheader("🎫 Ajánlott duplázó (2 tipp)")

    if not ticket:
        st.error("Nem sikerült ticketet generálni. (Ezt most már DEMO mellett sem kellene látnod.)")
    else:
        badge = "<span class='badge'>LIVE</span>"
        if res.get("used_demo"):
            badge = "<span class='badge_demo'>DEMO / NEM AJÁNLOTT</span>"

        st.markdown(f"**Össz-odds:** `{total_odds:.2f}` {badge}", unsafe_allow_html=True)

        for idx, t in enumerate(ticket, start=1):
            kickoff_local = t["kickoff"].astimezone() if t.get("kickoff") else None
            meta = t.get("meta", {})
            w = meta.get("weather", {})
            nh = meta.get("news_home", {})
            na = meta.get("news_away", {})

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{idx}  {t['match']}")
            if kickoff_local:
                st.markdown(
                    f"<span class='muted'>Liga:</span> `{t['league']}`  |  "
                    f"<span class='muted'>Kezdés:</span> **{kickoff_local.strftime('%Y.%m.%d %H:%M')}**",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<span class='muted'>Liga:</span> `{t['league']}`", unsafe_allow_html=True)

            st.markdown(
                f"**Minőség:** `{t.get('data_quality','LIVE')}`  |  "
                f"**Piac:** `{t['bet_type']}`  |  **Odds:** `{t['odds']:.2f}`  |  **Score:** `{t['score']:.0f}/100`"
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

            st.markdown("**Miért ezt ajánlja:**")
            st.write(t["reasoning"])

            if w:
                st.caption(f"🌦️ Időjárás: {w.get('temp','?')}°C, {w.get('desc','?')}, szél: {w.get('wind','?')} m/s")

            if nh.get("lines") or na.get("lines"):
                with st.expander("📰 Friss hírcímek (forrással)", expanded=False):
                    st.write(f"**{t['home']}**")
                    for line in (nh.get("lines") or ["• nincs releváns friss cím"]):
                        st.write(line)
                    st.write(f"**{t['away']}**")
                    for line in (na.get("lines") or ["• nincs releváns friss cím"]):
                        st.write(line)

            st.markdown("</div>", unsafe_allow_html=True)

if save_btn:
    if st.session_state["last_run"] is None or not st.session_state["last_run"]["ticket"]:
        st.warning("Előbb futtasd az elemzést, hogy legyen 2 tipp.")
    else:
        save_ticket(st.session_state["last_run"]["ticket"])
        st.success("✅ A két tipp mentve az adatbázisba.")

st.divider()
st.subheader("📜 Előzmények")

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
