import streamlit as st
import requests
import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter, Retry

# =======================
# KONFIGURÁCIÓ
# =======================
try:
    API_KEY = st.secrets["ODDS_API_KEY"]
    NEWS_KEY = st.secrets["NEWS_API_KEY"]
except KeyError:
    st.error("❌ Hiányzó API kulcs a Streamlit Secrets-ben.")
    st.stop()

TIMEOUT = (3, 10)  # connect / read
TTL_API = 900      # 15 perc cache
DB_NAME = "titan_betting.db"

# =======================
# HTTP SESSION (RETRY + RATE LIMIT BARÁT)
# =======================
@st.cache_resource
def get_http_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = get_http_session()

# =======================
# ADATBÁZIS
# =======================
@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            match TEXT,
            pick TEXT,
            market TEXT,
            odds REAL,
            score INTEGER,
            summary TEXT
        )
    """)
    conn.commit()
    return conn

DB = init_db()

# =======================
# SZIMULÁLT STAT MODULOK
# =======================
def market_simulation():
    return {
        "corners": round(random.uniform(8.5, 12.5), 1),
        "cards": round(random.uniform(3.2, 5.8), 1),
        "attack_index": random.randint(60, 95)
    }

def referee_profile():
    refs = [
        ("Michael Oliver", 4.1, "Szigorú, sok lap"),
        ("Anthony Taylor", 3.8, "Stabil, büntető-hajlamos"),
        ("Makkelie", 3.5, "Engedékeny"),
        ("Marciniak", 4.5, "Autoriter")
    ]
    name, y, style = random.choice(refs)
    return {"name": name, "yellow_avg": y, "style": style}

# =======================
# HÍR ELEMZÉS (CACHE-ELT)
# =======================
@st.cache_data(ttl=TTL_API)
def get_team_news(team):
    try:
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={team} football injury lineup&"
            "language=en&pageSize=3&sortBy=publishedAt&"
            f"apiKey={NEWS_KEY}"
        )
        r = SESSION.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        if not articles:
            return 0, "Nincs releváns hír."

        text = (articles[0]["title"] or "").lower()
        if any(w in text for w in ["injury", "out", "absent", "surgery"]):
            return -25, "🚨 Kulcsjátékos hiány"
        if any(w in text for w in ["return", "back", "fit"]):
            return +15, "📈 Fontos visszatérő"

        return 0, "Semleges hírek"
    except Exception:
        return 0, "Híradat nem elérhető"

# =======================
# ODDS LEKÉRÉS (CACHE)
# =======================
@st.cache_data(ttl=TTL_API)
def fetch_odds(league):
    url = (
        f"https://api.the-odds-api.com/v4/sports/{league}/odds"
        f"?apiKey={API_KEY}&regions=eu&markets=h2h"
    )
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# =======================
# TITAN ENGINE
# =======================
class TitanEngine:
    LEAGUES = [
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_italy_serie_a",
        "soccer_germany_bundesliga"
    ]

    def analyze(self):
        now = datetime.now(timezone.utc)
        results = []

        for lg in self.LEAGUES:
            try:
                matches = fetch_odds(lg)
                for m in matches:
                    kickoff = datetime.fromisoformat(
                        m["commence_time"].replace("Z", "+00:00")
                    )
                    if not (now <= kickoff <= now + timedelta(hours=24)):
                        continue

                    bookies = m.get("bookmakers", [])
                    if not bookies:
                        continue

                    market = bookies[0]["markets"][0]
                    fav = min(market["outcomes"], key=lambda x: x["price"])

                    if not 1.30 <= fav["price"] <= 1.80:
                        continue

                    news_mod, news_txt = get_team_news(fav["name"])
                    stats = market_simulation()
                    ref = referee_profile()

                    score = int(
                        70 +
                        news_mod +
                        (stats["attack_index"] / 5)
                    )

                    results.append({
                        "match": f"{m['home_team']} vs {m['away_team']}",
                        "pick": fav["name"],
                        "odds": fav["price"],
                        "score": max(5, min(99, score)),
                        "news": news_txt,
                        "corners": stats["corners"],
                        "cards": stats["cards"],
                        "ref": ref
                    })
            except Exception:
                continue

        return sorted(results, key=lambda x: x["score"], reverse=True)

# =======================
# STREAMLIT UI
# =======================
st.set_page_config("TITAN V13.1", layout="wide")
st.title("🦾 TITAN Football Intelligence V13.1")

if st.button("🚀 Elemzés indítása"):
    engine = TitanEngine()
    data = engine.analyze()

    if not data:
        st.warning("Nincs releváns mérkőzés a következő 24 órában.")
    else:
        ticket = data[:2]
        total_odds = round(ticket[0]["odds"] * ticket[1]["odds"], 2)

        st.header(f"🎫 Dupla szelvény | Odds: {total_odds}")

        if any(p["score"] < 90 for p in ticket):
            st.warning("⚠️ Ma nincs 90%+ TUTI ajánlat.")

        for p in ticket:
            st.write(
                f"**{p['match']}** → {p['pick']} | "
                f"{p['odds']} | {p['score']}%"
            )
            st.caption(
                f"Szögletek: {p['corners']} | "
                f"Lapok: {p['cards']} | "
                f"Bíró: {p['ref']['name']}"
            )
            st.info(p["news"])

            DB.execute(
                "INSERT INTO history VALUES (NULL,?,?,?,?,?,?)",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    p["match"],
                    p["pick"],
                    "H2H",
                    p["odds"],
                    p["score"],
                    p["news"]
                )
            )
        DB.commit()

st.subheader("📊 Előzmények")
df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", DB)
st.dataframe(df, use_container_width=True)
