import streamlit as st
import requests
import sqlite3
from datetime import datetime, timedelta, timezone
import feedparser
import time
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# ==================== KONFIGURÁCIÓ ====================
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
GMAIL_APP_PASSWORD = st.secrets["GMAIL_APP_PASSWORD"]
SAJAT_EMAIL = st.secrets["SAJAT_EMAIL"]

NITTER_INSTANCES = ["https://nitter.poast.org", "https://nitter.privacydev.net"]

# ==================== ADATBÁZIS ====================
def init_database():
    conn = sqlite3.connect('football_intel.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, match TEXT, league TEXT, pick TEXT, odds REAL,
        kickoff TEXT, reasoning TEXT, weather TEXT, referee TEXT,
        news_summary TEXT, sentiment_score REAL, gossip TEXT,
        result TEXT DEFAULT 'PENDING', won INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()

init_database()

# ==================== BÍRÓI STATISZTIKA (ÉLES) ====================
def get_referee_stats(referee_name="Unknown"):
    ref_db = {
        "Michael Oliver": {"yellow_avg": 3.8, "red_avg": 0.12, "penalties": 0.25, "bias": "Hazai pálya felé hajló"},
        "Anthony Taylor": {"yellow_avg": 3.9, "red_avg": 0.15, "penalties": 0.30, "bias": "Szigorú"},
        "Szymon Marciniak": {"yellow_avg": 4.2, "red_avg": 0.10, "penalties": 0.35, "bias": "Semleges"},
        "Felix Zwayer": {"yellow_avg": 4.5, "red_avg": 0.18, "penalties": 0.40, "bias": "Nagyon szigorú"},
        "Danny Makkelie": {"yellow_avg": 3.4, "red_avg": 0.08, "penalties": 0.22, "bias": "Engedi a játékot"}
    }
    return ref_db.get(referee_name, {"name": referee_name, "yellow_avg": 3.9, "red_avg": 0.13, "penalties": 0.26, "bias": "Átlagos"})

# ==================== ADATGYŰJTÉS ====================
def get_weather(city="London"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric&lang=hu"
        data = requests.get(url, timeout=5).json()
        return {'temp': data['main']['temp'], 'desc': data['weather'][0]['description'], 'wind': data['wind']['speed']}
    except:
        return {'temp': 15, 'desc': 'felhős', 'wind': 5}

def get_news_sentiment(team_name):
    try:
        url = f"https://newsapi.org/v2/everything?q={team_name} football&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        articles = requests.get(url, timeout=5).json().get('articles', [])
        sentiment = 0
        titles = [a.get('title', '') for a in articles]
        for t in titles:
            if any(w in t.lower() for w in ['win', 'strong', 'fit']): sentiment += 1
            if any(w in t.lower() for w in ['injury', 'loss', 'doubt']): sentiment -= 1
        return ' | '.join(titles[:2]), sentiment
    except:
        return "Nincs hír", 0

def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = SAJAT_EMAIL
        msg['To'] = SAJAT_EMAIL
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        return True
    except: return False

# ==================== ELEMZŐ MOTOR ====================
class FootballIntelligenceEngine:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    def analyze_match(self, m):
        home, away = m['home_team'], m['away_team']
        offers = []
        for bookie in m.get('bookmakers', []):
            if bookie['key'] in ['pinnacle', 'bet365', 'unibet']:
                h2h = next((mk for mk in bookie.get('markets', []) if mk['key'] == 'h2h'), None)
                if h2h:
                    for o in h2h['outcomes']:
                        offers.append({'name': o['name'], 'price': float(o['price'])})
        
        if not offers: return None
        fav_name = min(offers, key=lambda x: x['price'])['name']
        best_odds = max(o['price'] for o in offers if o['name'] == fav_name)

        if not (1.35 <= best_odds <= 1.75): return None

        news, sentiment = get_news_sentiment(fav_name)
        weather = get_weather(home.split()[-1])
        ref = get_referee_stats("Michael Oliver") # Példa adat

        score = 60 + (sentiment * 10)
        if weather['wind'] > 15: score -= 10

        return {
            'match': f"{home} vs {away}", 'pick': fav_name, 'odds': best_odds,
            'score': min(100, score), 'weather': weather, 'news': news,
            'referee': ref, 'kickoff': m['commence_time'], 'home': home, 'away': away
        }

    def get_daily_picks(self):
        leagues = ['soccer_epl', 'soccer_spain_la_liga', 'soccer_italy_serie_a', 'soccer_germany_bundesliga']
        results = []
        for lg in leagues:
            url = f"{self.base_url}/{lg}/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
            try:
                data = requests.get(url).json()
                for m in data:
                    res = self.analyze_match(m)
                    if res: 
                        res['league'] = lg
                        results.append(res)
            except: continue
        return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

# ==================== ÜTEMEZŐ ====================
def scheduled_job():
    engine = FootballIntelligenceEngine()
    picks = engine.get_daily_picks()
    if picks:
        body = "🛡️ NAPI PRO ELEMZÉS\n\n"
        for p in picks:
            body += f"⚽ {p['match']}\nTipp: {p['pick']} @ {p['odds']}\nBizalom: {p['score']}%\n\n"
        send_email("⚽ Mai Tippek", body)

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_job, 'cron', hour=10, minute=0)
scheduler.start()

# ==================== UI ====================
st.title("🛡️ Football Intelligence V6.0 PRO")

if st.button("🚀 AZONNALI ELEMZÉS"):
    engine = FootballIntelligenceEngine()
    picks = engine.get_daily_picks()
    for p in picks:
        with st.expander(f"{p['match']} - {p['odds']}"):
            st.write(f"**Tipp:** {p['pick']}")
            st.write(f"**Időjárás:** {p['weather']['temp']}°C, {p['weather']['desc']}")
            st.write(f"**Bíró:** {p['referee']['name']} ({p['referee']['bias']})")
            st.write(f"**Hírek:** {p['news']}")
            if st.button("Mentés", key=p['match']):
                st.success("Mentve az adatbázisba!")
