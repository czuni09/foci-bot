import streamlit as st
import requests
import sqlite3
import pandas as pd
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler

# --- KONFIGURÁCIÓ ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    GMAIL_APP_PASSWORD = st.secrets["GMAIL_APP_PASSWORD"]
    SAJAT_EMAIL = st.secrets["SAJAT_EMAIL"]
except KeyError as e:
    st.error(f"Hiányzó kulcs: {e}")
    st.stop()

# --- ADATBÁZIS ÉS STATISZTIKA ---
def init_db():
    conn = sqlite3.connect('football_stats.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (date TEXT, match TEXT, pick TEXT, odds REAL, score INTEGER, rec TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- MODULOK ---
def get_referee_stats(referee_name="Unknown"):
    ref_db = {
        "Michael Oliver": {"yellow_avg": 3.8, "red_avg": 0.12, "penalties": 0.25, "bias": "Hazai barát"},
        "Anthony Taylor": {"yellow_avg": 3.9, "red_avg": 0.15, "penalties": 0.30, "bias": "Szigorú"},
        "Szymon Marciniak": {"yellow_avg": 4.2, "red_avg": 0.10, "penalties": 0.35, "bias": "Semleges"},
        "Felix Zwayer": {"yellow_avg": 4.5, "red_avg": 0.18, "penalties": 0.40, "bias": "Szigorú"},
        "Danny Makkelie": {"yellow_avg": 3.4, "red_avg": 0.08, "penalties": 0.22, "bias": "Engedékeny"}
    }
    return ref_db.get(referee_name, {"name": referee_name, "yellow_avg": 3.9, "red_avg": 0.13, "penalties": 0.26, "bias": "Átlagos"})

def get_news_sentiment(team):
    try:
        url = f"https://newsapi.org/v2/everything?q={team} football&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        res = requests.get(url, timeout=5).json()
        articles = res.get('articles', [])
        if not articles: return "Nincs friss hír", 0
        sentiment = 1 if any(x in articles[0]['title'].lower() for x in ['win', 'fit', 'ready']) else 0
        return articles[0]['title'], sentiment
    except: return "Hírek nem elérhetők", 0

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric&lang=hu"
        res = requests.get(url, timeout=5).json()
        return {'temp': res['main']['temp'], 'desc': res['weather'][0]['description'], 'wind': res['wind']['speed']}
    except: return {'temp': 15, 'desc': 'Felhős', 'wind': 5}

# --- ANALÍZIS ---
class FootballFullEngine:
    def __init__(self):
        self.leagues = ['soccer_epl', 'soccer_championship', 'soccer_spain_la_liga', 'soccer_italy_serie_a', 'soccer_germany_bundesliga']

    def get_recommendation(self, score):
        if score >= 90: return "💎 TUTI TIPP"
        if score >= 75: return "✅ AJÁNLOTT"
        return "⚠️ ÁTGONDOLÁSRA (Rizikós, csak saját felelősségre)"

    def process_match(self, m):
        home, away = m['home_team'], m['away_team']
        bookie = next((b for b in m.get('bookmakers', []) if b['key'] == 'bet365'), m['bookmakers'][0] if m.get('bookmakers') else None)
        if not bookie: return None
        
        market = next((mk for mk in bookie['markets'] if mk['key'] == 'h2h'), None)
        if not market: return None
        
        fav = min(market['outcomes'], key=lambda x: x['price'])
        odds = fav['price']
        
        news, sent = get_news_sentiment(fav['name'])
        weather = get_weather(home.split()[-1])
        ref = get_referee_stats("Michael Oliver") # Példa adat
        
        # Súlyozott pontszám
        score = 65 + (sent * 15)
        if 1.40 <= odds <= 1.70: score += 10
        if weather['wind'] > 15: score -= 10
        
        final_score = min(100, max(0, score))
        rec = self.get_recommendation(final_score)
        
        return {
            'date': m['commence_time'], 'match': f"{home} vs {away}", 'pick': fav['name'],
            'odds': odds, 'score': final_score, 'rec': rec, 'news': news,
            'weather': weather, 'referee': ref
        }

    def run_analysis(self):
        all_results = []
        for lg in self.leagues:
            url = f"https://api.the-odds-api.com/v4/sports/{lg}/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
            data = requests.get(url).json()
            if isinstance(data, list):
                for m in data:
                    res = self.process_match(m)
                    if res: all_results.append(res)
        return all_results

# --- FELÜLET ---
st.set_page_config(page_title="Football Intelligence V7.0", layout="wide")
st.title("🛡️ Football Intelligence V7.0 PRO")

tab1, tab2 = st.tabs(["🚀 Élő Elemzés", "📊 Statisztikai Táblázat"])

with tab1:
    if st.button("FUTTATÁS"):
        engine = FootballFullEngine()
        results = engine.run_analysis()
        
        # Adatbázisba mentés
        conn = sqlite3.connect('football_stats.db')
        for r in results:
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?)", 
                         (r['date'], r['match'], r['pick'], r['odds'], r['score'], r['rec']))
        conn.commit()
        conn.close()

        for r in results:
            color = "blue" if "TUTI" in r['rec'] else "green" if "AJÁNLOTT" in r['rec'] else "orange"
            with st.expander(f"{r['match']} | {r['score']}% | {r['rec']}", expanded=("TUTI" in r['rec'])):
                st.markdown(f"### :{color}[{r['rec']}]")
                c1, c2, c3 = st.columns(3)
                c1.metric("Százalék", f"{r['score']}%")
                c1.write(f"**Tipp:** {r['pick']} (@{r['odds']})")
                c2.write(f"**👨‍⚖️ Bíró:** {r['referee']['bias']} (Sárga: {r['referee']['yellow_avg']})")
                c2.write(f"**☁️ Időjárás:** {r['weather']['temp']}°C, {r['weather']['desc']}")
                c3.write("**📰 Hírek:**")
                c3.caption(r['news'])

with tab2:
    st.header("Múltbéli Elemzések és Statisztika")
    conn = sqlite3.connect('football_stats.db')
    df = pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.download_button("Adatok letöltése (CSV)", df.to_csv(index=False), "stats.csv")
    else:
        st.info("Még nincs mentett adat.")
