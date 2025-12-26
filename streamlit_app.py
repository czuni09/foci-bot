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
import plotly.graph_objects as go

# ==================== KONFIGURÁCIÓ ====================
ODDS_API_KEY = "cc1a32d7a1d30cb4898eb879ff6d636f"
WEATHER_KEY = "c31a011d35fed1b4d7b9f222c99d6dd2"
NEWS_API_KEY = "7d577a4d9f2b4ba38541cc3f7e5ad6f5"
FOOTBALL_DATA_KEY = "4fb029dfe0f5464492779774807045d3"
GMAIL_APP_PASSWORD = "whppzywzoduqjrgk"
SAJAT_EMAIL = "czunidaniel9@gmail.com"

NITTER_INSTANCES = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.unixfox.eu"
]

# ==================== ADATBÁZIS ====================
def init_database():
    conn = sqlite3.connect('football_intel.db', check_same_thread=False)
    c = conn.cursor()
    
    # Tábla létrehozása vagy módosítása
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        match TEXT,
        league TEXT,
        pick TEXT,
        odds REAL,
        kickoff TEXT,
        reasoning TEXT,
        weather TEXT,
        referee TEXT,
        news_summary TEXT,
        sentiment_score REAL,
        gossip TEXT,
        result TEXT DEFAULT 'PENDING',
        won INTEGER DEFAULT 0
    )''')
    
    conn.commit()
    conn.close()

init_database()

# ==================== EMAIL KÜLDÉS ====================
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
    except Exception as e:
        print(f"Email hiba: {e}")
        return False

# ==================== ADATGYŰJTÉS ====================
def get_weather(city="London"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric&lang=hu"
        r = requests.get(url, timeout=5)
        data = r.json()
        return {
            'temp': data['main']['temp'],
            'desc': data['weather'][0]['description'],
            'wind': data['wind']['speed'],
            'humidity': data['main']['humidity']
        }
    except:
        return {'temp': 15, 'desc': 'ismeretlen', 'wind': 5, 'humidity': 60}

def get_news_sentiment(team_name):
    try:
        url = f"https://newsapi.org/v2/everything?q={team_name} football&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        r = requests.get(url, timeout=5)
        articles = r.json().get('articles', [])
        
        summary = []
        sentiment = 0
        for article in articles[:3]:
            title = article.get('title', '')
            desc = article.get('description', '')
            full_text = (title + ' ' + desc).lower()
            
            summary.append(title)
            
            if any(word in full_text for word in ['win', 'victory', 'strong', 'top', 'best', 'form', 'excellent', 'brilliant']):
                sentiment += 1
            if any(word in full_text for word in ['loss', 'injury', 'crisis', 'doubt', 'weak', 'out', 'injured', 'suspended']):
                sentiment -= 1
        
        summary_text = ' | '.join(summary[:2]) if summary else "Nincs friss hír"
        return summary_text, sentiment
    except:
        return "Nincs friss hír", 0

def get_reddit_gossip():
    try:
        feed = feedparser.parse("https://www.reddit.com/r/soccer/.rss")
        hot_topics = []
        for entry in feed.entries[:5]:
            title = entry.title
            if any(word in title.lower() for word in ['rumor', 'gossip', 'drama', 'controversy', 'scandal', 'incident']):
                hot_topics.append(title)
        gossip_text = " | ".join(hot_topics[:3]) if hot_topics else "Nincs friss pletyká"
        return gossip_text
    except:
        return "Reddit elérhetetlen"

def scrape_referee_data(referee_name="Michael Oliver"):
    referee_db = {
        "Michael Oliver": {'yellow_avg': 3.2, 'red_avg': 0.1, 'penalties': 0.15, 'bias': 'semleges'},
        "Anthony Taylor": {'yellow_avg': 4.5, 'red_avg': 0.05, 'penalties': 0.18, 'bias': 'semleges'},
        "Szymon Marciniak": {'yellow_avg': 5.2, 'red_avg': 0.04, 'penalties': 0.12, 'bias': 'semleges'},
        "Istvan Kovacs": {'yellow_avg': 4.0, 'red_avg': 0.03, 'penalties': 0.10, 'bias': 'semleges'}
    }
    return referee_db.get(referee_name, {'yellow_avg': 3.5, 'red_avg': 0.08, 'penalties': 0.14, 'bias': 'semleges', 'name': referee_name})

# ==================== ELEMZŐ MOTOR ====================
class FootballIntelligenceEngine:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        
    def discover_leagues(self):
        try:
            res = requests.get(f"{self.base_url}?apiKey={ODDS_API_KEY}", timeout=10)
            res.raise_for_status()
            return [s['key'] for s in res.json() if s['group'] == 'Soccer' and 'winner' not in s['key']]
        except:
            return ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga', 'soccer_italy_serie_a', 'soccer_france_ligue_one']
    
    def analyze_match(self, match_data):
        home = match_data['home_team']
        away = match_data['away_team']
        
        bookmakers = match_data.get('bookmakers', [])
        best_odds = {}
        for bookie in bookmakers:
            if bookie['key'] in ['pinnacle', 'bet365', 'unibet', 'williamhill', 'marathonbet']:
                for market in bookie.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            name = outcome['name']
                            price = float(outcome['price'])
                            if name not in best_odds or price > best_odds[name]:
                                best_odds[name] = price
        
        if not best_odds:
            return None
        
        favorite = min(best_odds, key=best_odds.get)
        fav_odds = best_odds[favorite]
        
        if not (1.35 <= fav_odds <= 1.70):
            return None
        
        city = home.split()[-1] if ' ' in home else 'London'
        weather = get_weather(city)
        
        news_home, sent_home = get_news_sentiment(home)
        news_away, sent_away = get_news_sentiment(away)
        
        gossip = get_reddit_gossip()
        referee = scrape_referee_data()
        
        score = 50
        
        if 1.45 <= fav_odds <= 1.60:
            score += 15
        elif 1.35 <= fav_odds <= 1.70:
            score += 10
        else:
            score += 5
        
        if favorite == home:
            score += sent_home * 5
        else:
            score += sent_away * 5
        
        if weather['wind'] > 15:
            score -= 10
        if 'rain' in weather['desc'].lower() or 'storm' in weather['desc'].lower():
            score -= 5
        if weather['temp'] > 30:
            score -= 5
        
        if referee['bias'] == 'semleges':
            score += 10
        else:
            score -= 5
        
        final_score = min(100, max(0, score))
        
        return {
            'match': f"{home} vs {away}",
            'home': home,
            'away': away,
            'pick': favorite,
            'odds': fav_odds,
            'score': final_score,
            'weather': weather,
            'news_home': news_home,
            'news_away': news_away,
            'sentiment': sent_home if favorite == home else sent_away,
            'gossip': gossip,
            'referee': referee,
            'kickoff': match_data['commence_time']
        }
    
    def get_daily_picks(self):
        leagues = self.discover_leagues()
        candidates = []
        
        now = datetime.now(timezone.utc)
        limit_24h = now + timedelta(hours=24)
        
        for league in leagues:
            url = f"{self.base_url}/{league}/odds"
            params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 422:
                    continue
                response.raise_for_status()
                
                for match in response.json():
                    kickoff = datetime.fromisoformat(match['commence_time'].replace('Z', '+00:00'))
                    if now <= kickoff <= limit_24h:
                        analyzed = self.analyze_match(match)
                        if analyzed:
                            analyzed['league'] = league
                            analyzed['kickoff_dt'] = kickoff
                            candidates.append(analyzed)
            except:
                continue
            
            time.sleep(0.5)
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:3]

# ==================== SCHEDULER ====================
def daily_morning_analysis():
    engine = FootballIntelligenceEngine()
    picks = engine.get_daily_picks()
    
    if not picks:
        send_email("⚽ Napi Elemzés - Nincs tipp", "Ma nincs megfelelő mérkőzés.")
        return
    
    email_body = "🛡️ STRATEGIC FOOTBALL INTELLIGENCE - NAPI ELEMZÉS\n"
    email_body += f"📅 {datetime.now().strftime('%Y.%m.%d %H:%M')}\n"
    email_body += "=" * 70 + "\n\n"
    
    for i, pick in enumerate(picks, 1):
        kickoff_local = pick['kickoff_dt'].astimezone()
        w = pick['weather']
        ref = pick['referee']
        
        email_body += f"🎯 TIPP #{i}\n"
        email_body += f"Mérkőzés: {pick['match']}\n"
        email_body += f"Bajnokság: {pick['league'].replace('soccer_', '').replace('_', ' ').upper()}\n"
        email_body += f"Kezdés: {kickoff_local.strftime('%Y.%m.%d %H:%M')}\n"
        email_body += f"Tipp: {pick['pick']} @ {pick['odds']:.2f}\n"
        email_body += f"Magabiztosság: {pick['score']}/100\n\n"
        email_body += f"🌦️ Időjárás: {w['temp']:.1f}°C, {w['desc']}, szél {w['wind']:.1f} m/s\n"
        email_body += f"📰 Hírek: {pick['news_home']} | {pick['news_away']}\n"
        email_body += f"👨‍⚖️ Bíró: {ref.get('name', 'Ismeretlen')} ({ref['yellow_avg']} sárga/meccs)\n"
        email_body += f"💬 Reddit: {pick['gossip'][:100]}\n\n"
        email_body += f"💡 Indoklás: A {pick['pick']} csapat {pick['odds']:.2f} oddsa ideális. "
        email_body += f"Időjárás {'kedvező' if w['wind'] < 10 else 'közepesen kedvező'}. "
        email_body += f"Hírek {'pozitívak' if pick['sentiment'] > 0 else 'semlegesek' if pick['sentiment'] == 0 else 'vegyes képet mutatnak'}.\n"
        email_body += "=" * 70 + "\n\n"
    
    send_email("⚽ Napi Tippek", email_body)

def pre_match_update():
    engine = FootballIntelligenceEngine()
    picks = engine.get_daily_picks()
    
    now = datetime.now(timezone.utc)
    upcoming = [p for p in picks if timedelta(minutes=20) <= (p['kickoff_dt'] - now) <= timedelta(minutes=40)]
    
    if not upcoming:
        return
    
    email_body = "🔔 MECCS ELŐTTI UPDATE (30 perc)\n"
    email_body += f"📅 {datetime.now().strftime('%Y.%m.%d %H:%M')}\n\n"
    
    for pick in upcoming:
        email_body += f"⚽ {pick['match']}\n"
        email_body += f"Tipp: {pick['pick']} @ {pick['odds']:.2f}\n"
        email_body += f"Score: {pick['score']}/100\n\n"
    
    send_email("⚽ Meccs Előtti Update", email_body)

scheduler = BackgroundScheduler()
scheduler.add_job(func=daily_morning_analysis, trigger="cron", hour=10, minute=0)
scheduler.add_job(func=pre_match_update, trigger="interval", minutes=15)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ==================== UI ====================
st.set_page_config(page_title="⚽ Strategic Intelligence", layout="wide", page_icon="⚽")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; }
    .metric-good { color: #3dff8b; font-weight: bold; }
    .metric-neutral { color: #ffa500; font-weight: bold; }
    .metric-bad { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Strategic Football Intelligence V6.0")
st.caption("Auto: 10:00 + Meccs-30min | czunidaniel9@gmail.com")

with st.sidebar:
    st.header("📊 Statisztikák")
    conn = sqlite3.connect('football_intel.db', check_same_thread=False)
    c = conn.cursor()
    
    total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    won = c.execute("SELECT COUNT(*) FROM predictions WHERE won=1").fetchone()[0]
    pending = c.execute("SELECT COUNT(*) FROM predictions WHERE result='PENDING'").fetchone()[0]
    
    st.metric("Összes tipp", total)
    st.metric("Nyert", won, f"{(won/total*100) if total > 0 else 0:.1f}%")
    st.metric("Függőben", pending)
    
    conn.close()

tab1, tab2 = st.tabs(["📅 Mai Tippek", "📜 Történet"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 ELEMZÉS", type="primary", use_container_width=True):
            with st.spinner("🔍 Szkennelés..."):
                engine = FootballIntelligenceEngine()
                picks = engine.get_daily_picks()
                
                if not picks:
                    st.warning("⚠️ Nincs megfelelő mérkőzés")
                else:
                    st.success(f"✅ {len(picks)} tipp!")
                    
                    for i, pick in enumerate(picks, 1):
                        with st.expander(f"🎯 #{i} - {pick['match']} ({pick['odds']:.2f})", expanded=True):
                            kickoff_local = pick['kickoff_dt'].astimezone()
                            
                            col_l, col_r = st.columns([2, 1])
                            
                            with col_l:
                                st.markdown(f"### {pick['match']}")
                                st.info(f"⏰ {kickoff_local.strftime('%Y.%m.%d %H:%M')} | 🏆 {pick['league'].replace('soccer_', '').upper()}")
                                st.markdown(f"**🎲 Tipp:** `{pick['pick']}` @ **{pick['odds']:.2f}**")
                                
                                w = pick['weather']
                                st.markdown("#### 🌦️ Időjárás:")
                                st.markdown(f"- {w['temp']:.1f}°C, {w['desc']}, szél {w['wind']:.1f} m/s")
                                
                                st.markdown("#### 📰 Hírek:")
                                st.markdown(f"- {pick['home']}: {pick['news_home']}")
                                st.markdown(f"- {pick['away']}: {pick['news_away']}")
                                
                                ref = pick['referee']
                                st.markdown(f"#### 👨‍⚖️ Bíró: {ref.get('name', 'Ismeretlen')} ({ref['yellow_avg']} sárga)")
                                
                                st.markdown(f"#### 💬 Reddit: {pick['gossip'][:150]}...")
                            
                            with col_r:
                                st.metric("🎯 Score", f"{pick['score']}/100")
                                st.metric("💰 Odds", f"{pick['odds']:.2f}")
                                
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=pick['score'],
                                    gauge={'axis': {'range': [0, 100]},
                                           'bar': {'color': "#3dff8b" if pick['score'] > 75 else "#ffa500"}}
                                ))
                                fig.update_layout(height=200, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="#0d1117")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                if st.button(f"💾 Mentés", key=f"save_{i}"):
                                    reasoning = f"{pick['pick']} @ {pick['odds']:.2f}. Score: {pick['score']}/100."
                                    
                                    conn = sqlite3.connect('football_intel.db', check_same_thread=False)
                                    c = conn.cursor()
                                    c.execute('''INSERT INTO predictions 
                                        (date, match, league, pick, odds, kickoff, reasoning, weather, referee, news_summary, sentiment_score, gossip)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                        (datetime.now().strftime('%Y-%m-%d'), pick['match'], pick['league'], pick['pick'], pick['odds'],
                                         pick['kickoff_dt'].isoformat(), reasoning, json.dumps(pick['weather']), json.dumps(pick['referee']),
                                         pick['news_home'], pick['sentiment'], pick['gossip']))
                                    conn.commit()
                                    conn.close()
                                    st.success("✅ Mentve!")
    
    with col2:
        if st.button("📧 EMAIL", use_container_width=True):
            with st.spinner("📧 Küldés..."):
                daily_morning_analysis()
                st.success("✅ Elküldve!")

with tab2:
    st.header("📜 Történet")
    
    conn = sqlite3.connect('football_intel.db', check_same_thread=False)
    history = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    
    if not history:
        st.info("Nincs még tipp.")
    else:
        for row in history:
            id_, date, match, league, pick, odds, kickoff, reasoning, weather, referee, news, sent, gossip, result, won = row
            
            status = "✅" if won == 1 else "❌" if result != 'PENDING' else "⏳"
            
            with st.expander(f"{status} {date} - {match} ({pick} @ {odds:.2f})"):
                st.markdown(f"**Kickoff:** {kickoff}")
                st.markdown(f"**Státusz:** {result}")
                
                if result == 'PENDING':
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"✅ Nyert", key=f"w_{id_}"):
                            conn = sqlite3.connect('football_intel.db', check_same_thread=False)
                            conn.execute("UPDATE predictions SET result='WON', won=1 WHERE id=?", (id_,))
                            conn.commit()
                            conn.close()
                            st.rerun()
                    with c2:
                        if st.button(f"❌ Vesztett", key=f"l_{id_}"):
                            conn = sqlite3.connect('football_intel.db', check_same_thread=False)
                            conn.execute("UPDATE predictions SET result='LOST', won=0 WHERE id=?", (id_,))
                            conn.commit()
                            conn.close()
                            st.rerun()

st.divider()
st.caption("⚡ V6.0 | czunidaniel9@gmail.com")
