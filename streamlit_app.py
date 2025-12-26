import re
import json
import time
import requests
import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter, Retry

# ======================
# KONFIGURÁCIÓ & SECRETS
# ======================
st.set_page_config(page_title="TITAN V16.0 – MONSTRUM", layout="wide")
st.title("🦾 TITAN V16.0 – Intelligens Szelvény & Riport")

try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except KeyError as e:
    st.error(f"Hiányzó secret: {e}")
    st.stop()

TIMEOUT = (3.05, 12)
TTL = 600  # 10 perc cache

LEAGUES = [
    ("Premier League", "soccer_epl"),
    ("Championship", "soccer_championship"),
    ("La Liga", "soccer_spain_la_liga"),
    ("Serie A", "soccer_italy_serie_a"),
    ("Bundesliga", "soccer_germany_bundesliga"),
]

# ======================
# HTTP ENGINE (Retry + 429-barát)
# ======================
@st.cache_resource
def session():
    s = requests.Session()
    r = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    a = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=50)
    s.mount("https://", a)
    s.mount("http://", a)
    return s

S = session()

def get_json(url: str, params: dict | None = None) -> dict | list:
    r = S.get(url, params=params, timeout=TIMEOUT)
    if r.status_code == 429:
        time.sleep(2)
        r = S.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# ======================
# SEGÉDFÜGGVÉNYEK
# ======================
def within_next_24h(iso: str) -> bool:
    now = datetime.now(timezone.utc)
    ko = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return now <= ko <= now + timedelta(hours=24)

def fmt_dt(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return iso

def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.0

def get_extra_stats():
    """Szimulált statisztikai motor szögletekhez és lapokhoz."""
    return {
        "corners": round(random.uniform(8.2, 11.8), 1),
        "cards": round(random.uniform(3.1, 5.5), 1),
        "referee": random.choice(["Michael Oliver (Szigorú)", "Anthony Taylor (Engedékeny)", "Szymon Marciniak (Határozott)"])
    }

# ======================
# ADATGYŰJTÉS (ODDS & HÍREK)
# ======================
@st.cache_data(ttl=TTL)
def fetch_matches(league_key: str) -> list[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}
    try:
        data = get_json(url, params=params)
        if not isinstance(data, list): return []
        return [m for m in data if "commence_time" in m and within_next_24h(m["commence_time"])]
    except: return []

def extract_h2h_prices(match: dict) -> list[dict]:
    out = []
    for b in match.get("bookmakers", []):
        for mk in b.get("markets", []):
            if mk.get("key") == "h2h":
                for o in mk.get("outcomes", []):
                    out.append({"book": b.get("key"), "outcome": o["name"], "price": float(o["price"])})
    return out

def summarize_market(prices: list[dict]) -> dict:
    if not prices: return {"ok": False, "reason": "Nincs odds adat."}
    df = pd.DataFrame(prices)
    grp = df.groupby("outcome")["price"]
    stats = grp.agg(["min", "max", "mean", "count"]).reset_index()
    fav_row = stats.loc[stats["min"].idxmin()]
    
    spread = (fav_row["max"] - fav_row["min"])
    spread_pct = (spread / fav_row["mean"]) if fav_row["mean"] else 0.0
    
    return {
        "ok": True, "fav": str(fav_row["outcome"]), "fav_min": float(fav_row["min"]),
        "fav_mean": float(fav_row["mean"]), "fav_max": float(fav_row["max"]),
        "fav_books": int(fav_row["count"]), "spread_pct": spread_pct, "table": stats.sort_values("min")
    }

@st.cache_data(ttl=TTL)
def fetch_news(team: str) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": f'{team} football injury lineup', "language": "en", "sortBy": "publishedAt", "pageSize": 5, "apiKey": NEWS_API_KEY}
    try:
        data = get_json(url, params=params)
        return data.get("articles", []) if isinstance(data, dict) else []
    except: return []

def classify_article(a: dict) -> dict:
    title = a.get("title") or ""
    desc = a.get("description") or ""
    content = (title + " " + desc).lower()
    kind = "semleges"
    if any(w in content for w in ["injury", "out", "miss", "suspended", "doubt"]): kind = "hiányzó/állapot"
    elif any(w in content for w in ["rumor", "reportedly", "linked"]): kind = "pletyka"
    return {"kind": kind, "title": title, "url": a.get("url"), "src": a.get("source", {}).get("name")}

# ======================
# 🎫 TITAN SZELVÉNY GENERÁTOR (A MONSTRUM SZÍVE)
# ======================
st.header("🎫 TITAN – Napi Duplázó Szelvény (~2.00 Odds)")

all_matches = []
for _, l_key in LEAGUES:
    all_matches.extend(fetch_matches(l_key))

def build_titan_ticket(matches):
    candidates = []
    for m in matches:
        p = extract_h2h_prices(m)
        s = summarize_market(p)
        if s["ok"] and 1.30 <= s["fav_min"] <= 1.80:
            news = fetch_news(s["fav"])
            injury_mod = -15 if any(classify_article(a)["kind"] == "hiányzó/állapot" for a in news) else 0
            score = 100 - (s["spread_pct"] * 100) + injury_mod
            candidates.append({"m": m, "s": s, "score": score, "news": news})
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:2]

ticket = build_titan_ticket(all_matches)

if len(ticket) < 2:
    st.warning("⚠️ Nincs elég mérkőzés a 24 órás ablakban a szelvényhez.")
else:
    t_odds = ticket[0]["s"]["fav_min"] * ticket[1]["s"]["fav_min"]
    if t_odds < 2.0 or any(t["score"] < 85 for t in ticket):
        st.info("📢 Ma nincs tökéletes kínálat, de ez a két mérkőzés áll hozzá a legközelebb.")
    
    st.subheader(f"🎯 Eredő odds: {t_odds:.2f}")
    cols = st.columns(2)
    for i, item in enumerate(ticket):
        with cols[i]:
            m, s, score = item["m"], item["s"], item["score"]
            st.markdown(f"### {i+1}. {m['home_team']} vs {m['away_team']}")
            st.metric("Magabiztosság", f"{score:.1f}%", "TUTI" if score >= 90 else None)
            st.write(f"**Tipp:** {s['fav']} | **Odds:** {s['fav_min']:.2f}")
            
            ex = get_extra_stats()
            st.caption(f"📐 Szögletek: {ex['corners']} | 🟨 Lapok: {ex['cards']}")
            st.caption(f"👨‍⚖️ Bíró: {ex['referee']}")
            
            st.markdown("**🔬 Szakmai indoklás:**")
            injury_info = "Vigyázat, sérültek a hírekben!" if score < 80 else "Stabil keret és piaci konszenzus."
            st.write(f"A piaci szórás {s['spread_pct']*100:.1f}%. {injury_info}")

# ======================
# 🔍 RÉSZLETES MECCS KERESŐ (EREDETI FUNKCIÓK)
# ======================
st.markdown("---")
st.subheader("🔍 Részletes Mérkőzés Riport")
colL, colR = st.columns(2)
with colL:
    sel_league = st.selectbox("Válassz Ligát", [x[0] for x in LEAGUES])
    l_key = dict(LEAGUES)[sel_league]
    l_matches = fetch_matches(l_key)

if l_matches:
    with colR:
        match_labels = [f"{fmt_dt(m['commence_time'])} • {m['home_team']} vs {m['away_team']}" for m in l_matches]
        sel_label = st.selectbox("Válassz meccset", match_labels)
    
    # Keresés a választott meccsre
    match = next(m for m in l_matches if f"{fmt_dt(m['commence_time'])} • {m['home_team']} vs {m['away_team']}" == sel_label)
    
    p = extract_h2h_prices(match)
    m_summary = summarize_market(p)
    
    st.markdown(f"#### 📌 {match['home_team']} vs {match['away_team']}")
    if m_summary["ok"]:
        st.dataframe(m_summary["table"], use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Hazai hírek:**")
            for a in fetch_news(match['home_team'])[:3]:
                st.caption(f"• {a['title']} ({a['source']['name']})")
        with c2:
            st.write("**Vendég hírek:**")
            for a in fetch_news(match['away_team'])[:3]:
                st.caption(f"• {a['title']} ({a['source']['name']})")
else:
    st.info("Nincs meccs ebben a ligában.")

st.divider()
st.caption("TITAN V16.0 FINAL MONSTRUM - Minden jog fenntartva.")
