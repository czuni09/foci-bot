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
import hashlib

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

# ==================== EMAIL ====================
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

def get_team_specific_news(team_name):
    """Specifikus csapat hírek keresése több forrásból"""
    try:
        # Specifikus keresési kifejezések
        queries = [
            f'"{team_name}" injury OR sérülés',
            f'"{team_name}" lineup OR kezdőcsapat',
            f'"{team_name}" form OR forma',
            f'"{team_name}" news OR hírek'
        ]
        
        all_articles = []
        sentiment = 0
        
        for query in queries:
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
            r = requests.get(url, timeout=5)
            articles = r.json().get('articles', [])
            
            for article in articles:
                title = article.get('title', '')
                desc = article.get('description', '')
                full_text = (title + ' ' + desc).lower()
                
                # Csak releváns cikkek
                if team_name.lower() in full_text:
                    all_articles.append(title)
                    
                    # Sentiment
                    if any(word in full_text for word in ['win', 'victory', 'strong', 'excellent', 'brilliant', 'top form']):
                        sentiment += 2
                    if any(word in full_text for word in ['injury', 'injured', 'out', 'suspended', 'crisis', 'loss', 'doubt']):
                        sentiment -= 2
            
            time.sleep(0.3)  # Rate limit
        
        # Deduplikáció
        unique_articles = list(set(all_articles))
        summary_text = ' | '.join(unique_articles[:3]) if unique_articles else f"Nincs specifikus hír a(z) {team_name} csapatról"
        
        return summary_text, sentiment
    except Exception as e:
        print(f"News API hiba: {e}")
        return f"Hír keresési hiba: {team_name}", 0

def get_reddit_gossip():
    try:
        feed = feedparser.parse("https://www.reddit.com/r/soccer/.rss")
        hot_topics = []
        for entry in feed.entries[:10]:
            title = entry.title
            if any(word in title.lower() for word in ['rumor', 'gossip', 'drama', 'controversy', 'scandal', 'incident', 'injury', 'transfer']):
                hot_topics.append(title)
        gossip_text = " | ".join(hot_topics[:3]) if hot_topics else "Nincs friss pletyká"
        return gossip_text
    except:
        return "Reddit elérhetetlen"

def scrape_referee_data(referee_name="Michael Oliver"):
    referee_db = {
        "Michael Oliver": {'yellow_avg': 3.2, 'red_avg': 0.1, 'bias': 'semleges'},
        "Anthony Taylor": {'yellow_avg': 4.5, 'red_avg': 0.05, 'bias': 'semleges'},
        "Szymon Marciniak": {'yellow_avg': 5.2, 'red_avg': 0.04, 'bias': 'semleges'},
        "Istvan Kovacs": {'yellow_avg': 4.0, 'red_avg': 0.03, 'bias': 'semleges'}
    }
    return referee_db.get(referee_name, {'yellow_avg': 3.5, 'red_avg': 0.08, 'bias': 'semleges', 'name': referee_name})

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
            return ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga', 'soccer_italy_serie_a']
    
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
        
        # Specifikus hírek keresése
        news_home, sent_home = get_team_specific_news(home)
        time.sleep(0.5)
        news_away, sent_away = get_team_specific_news(away)
        
        gossip = get_reddit_gossip()
        referee = scrape_referee_data()
        
        score = 50
        
        if 1.45 <= fav_odds <= 1.60:
            score += 15
        elif 1.35 <= fav_odds <= 1.70:
            score += 10
        
        if favorite == home:
            score += sent_home * 3
        else:
            score += sent_away * 3
        
        if weather['wind'] > 15:
            score -= 10
        if 'rain' in weather['desc'].lower():
            score -= 5
        if weather['temp'] > 30:
            score -= 5
        
        if referee['bias'] == 'semleges':
            score += 10
        
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
        send_email("⚽ Napi Elemzés", "Ma nincs megfelelő mérkőzés.")
        return
    
    email_body = "🛡️ STRATEGIC FOOTBALL INTELLIGENCE\n"
    email_body += f"📅 {datetime.now().strftime('%Y.%m.%d %H:%M')}\n"
    email_body += "=" * 70 + "\n\n"
    
    for i, pick in enumerate(picks, 1):
        kickoff_local = pick['kickoff_dt'].astimezone()
        w = pick['weather']
        ref = pick['referee']
        
        email_body += f"🎯 TIPP #{i}: {pick['match']}\n"
        email_body += f"Liga: {pick['league'].replace('soccer_', '').upper()}\n"
        email_body += f"Kezdés: {kickoff_local.strftime('%Y.%m.%d %H:%M')}\n"
        email_body += f"Tipp: {pick['pick']} @ {pick['odds']:.2f}\n"
        email_body += f"Score: {pick['score']}/100\n\n"
        email_body += f"🌦️ Időjárás: {w['temp']:.1f}°C, {w['desc']}, szél {w['wind']:.1f} m/s\n"
        email_body += f"📰 {pick['home']}: {pick['news_home'][:150]}\n"
        email_body += f"📰 {pick['away']}: {pick['news_away'][:150]}\n"
        email_body += f"👨‍⚖️ Bíró: {ref.get('name', 'Ismeretlen')} ({ref['yellow_avg']} sárga)\n"
        email_body += f"💬 Reddit: {pick['gossip'][:100]}\n"
        email_body += "=" * 70 + "\n\n"
    
    send_email("⚽ Napi Tippek", email_body)

def pre_match_update():
    engine = FootballIntelligenceEngine()
    picks = engine.get_daily_picks()
    
    now = datetime.now(timezone.utc)
    upcoming = [p for p in picks if timedelta(minutes=20) <= (p['kickoff_dt'] - now) <= timedelta(minutes=40)]
    
    if not upcoming:
        return
    
    email_body = "🔔 MECCS ELŐTTI UPDATE\n\n"
    for pick in upcoming:
        email_body += f"⚽ {pick['match']}\n"
        email_body += f"Tipp: {pick['pick']} @ {pick['odds']:.2f}\n\n"
    
    send_email("⚽ Update", email_body)

scheduler = BackgroundScheduler()
scheduler.add_job(func=daily_morning_analysis, trigger="cron", hour=10, minute=0)
scheduler.add_job(func=pre_match_update, trigger="interval", minutes=15)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ==================== MODERN UI ====================
st.set_page_config(page_title="⚽ Strategic Intelligence", layout="wide", page_icon="⚽")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .stApp { 
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .good { color: #00ff88; }
    .neutral { color: #ffa500; }
    .bad { color: #ff4b6e; }
    .neon-border {
        border: 2px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0,212,255,0.5);
        border-radius: 12px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ Strategic Football Intelligence V7.0</h1>', unsafe_allow_html=True)
st.caption("🤖 AI-Powered Analysis | Auto: 10:00 + Pre-Match | czunidaniel9@gmail.com")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Dashboard")
    conn = sqlite3.connect('football_intel.db', check_same_thread=False)
    c = conn.cursor()
    
    total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    won = c.execute("SELECT COUNT(*) FROM predictions WHERE won=1").fetchone()[0]
    pending = c.execute("SELECT COUNT(*) FROM predictions WHERE result='PENDING'").fetchone()[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Total", total)
        st.metric("✅ Won", won)
    with col2:
        st.metric("📊 Rate", f"{(won/total*100) if total > 0 else 0:.0f}%")
        st.metric("⏳ Pending", pending)
    
    conn.close()
    
    st.divider()
    st.info("🔄 **Scheduler**\n- 10:00 Analysis\n- 30min Pre-Match")

# Tabs
tab1, tab2 = st.tabs(["🎯 Today's Picks", "📜 History"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)
    with col2:
        email_btn = st.button("📧 Send Email", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("🔍 Scanning markets & gathering intel..."):
            engine = FootballIntelligenceEngine()
            picks = engine.get_daily_picks()
            
            if not picks:
                st.warning("⚠️ No suitable matches in the next 24h")
            else:
                st.success(f"✅ {len(picks)} high-quality picks found!")
                
                for i, pick in enumerate(picks, 1):
                    # Unique key generálása
                    match_id = hashlib.md5(f"{pick['match']}_{pick['kickoff']}_{i}".encode()).hexdigest()[:8]
                    
                    kickoff_local = pick['kickoff_dt'].astimezone()
                    
                    with st.container():
                        st.markdown(f'<div class="neon-border">', unsafe_allow_html=True)
                        
                        col_info, col_viz = st.columns([2, 1])
                        
                        with col_info:
                            st.markdown(f"### 🎯 Pick #{i}: {pick['match']}")
                            st.markdown(f"**🏆 League:** {pick['league'].replace('soccer_', '').replace('_', ' ').upper()}")
                            st.markdown(f"**⏰ Kickoff:** {kickoff_local.strftime('%Y.%m.%d %H:%M')}")
                            st.markdown(f"**💰 Recommendation:** `{pick['pick']}` @ **{pick['odds']:.2f}**")
                            
                            # Weather
                            w = pick['weather']
                            weather_class = "good" if w['wind'] < 10 else "neutral" if w['wind'] < 15 else "bad"
                            st.markdown(f"🌦️ **Weather:** <span class='{weather_class}'>{w['temp']:.1f}°C, {w['desc']}, wind {w['wind']:.1f}m/s</span>", unsafe_allow_html=True)
                            
                            # News
                            st.markdown(f"📰 **{pick['home']}:** {pick['news_home'][:120]}...")
                            st.markdown(f"📰 **{pick['away']}:** {pick['news_away'][:120]}...")
                            
                            # Referee
                            ref = pick['referee']
                            st.markdown(f"👨‍⚖️ **Referee:** {ref.get('name', 'Unknown')} (Avg {ref['yellow_avg']} yellows)")
                            
                            # Reddit
                            st.markdown(f"💬 **Reddit:** {pick['gossip'][:100]}...")
                        
                        with col_viz:
                            st.metric("🎯 Confidence", f"{pick['score']}/100")
                            st.metric("💰 Odds", f"{pick['odds']:.2f}")
                            
                            # Plotly gauge with unique key
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=pick['score'],
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#00ff88" if pick['score'] > 75 else "#ffa500" if pick['score'] > 60 else "#ff4b6e"},
                                    'steps': [
                                        {'range': [0, 60], 'color': "#1a1f3a"},
                                        {'range': [60, 75], 'color': "#2a3050"},
                                        {'range': [75, 100], 'color': "#3a4060"}
                                    ]
                                }
                            ))
                            fig.update_layout(
                                height=180,
                                margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={'color': "white", 'size': 14}
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{match_id}")
                            
                            if st.button("💾 Save", key=f"save_{match_id}"):
                                reasoning = f"{pick['pick']} @ {pick['odds']:.2f}. Score: {pick['score']}/100."
                                conn = sqlite3.connect('football_intel.db', check_same_thread=False)
                                c = conn.cursor()
                                c.execute('''INSERT INTO predictions 
                                    (date, match, league, pick, odds, kickoff, reasoning, weather, referee, news_summary, sentiment_score, gossip)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                    (datetime.now().strftime('%Y-%m-%d'), pick['match'], pick['league'], pick['pick'], pick['odds'],
                                     pick['kickoff_dt'].isoformat(), reasoning, json.dumps(pick['weather']), json.dumps(pick['referee']),
                                     pick['news_home'] + ' | ' + pick['news_away'], pick['sentiment'], pick['gossip']))
                                conn.commit()
                                conn.close()
                                st.success("✅ Saved!")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
    
    if email_btn:
        with st.spinner("📧 Sending..."):
            daily_morning_analysis()
            st.success("✅ Email sent!")

with tab2:
    st.markdown("### 📜 Prediction History")
    
    conn = sqlite3.connect('football_intel.db', check_same_thread=False)
    history = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 30").fetchall()
    conn.close()
    
    if not history:
        st.info("No predictions yet. Run analysis first!")
    else:
        for row in history:
            id_, date, match, league, pick, odds, kickoff, reasoning, weather, referee, news, sent, gossip, result, won = row
            
            status = "✅" if won == 1 else "❌" if result != 'PENDING' else "⏳"
            hist_id = hashlib.md5(f"hist_{id_}".encode()).hexdigest()[:8]
            
            with st.expander(f"{status} {date} - {match} ({pick} @ {odds:.2f})"):
                st.markdown(f"**Kickoff:** {kickoff}")
                st.markdown(f"**League:** {league}")
                st.markdown(f"**Status:** {result}")
                
                if result == 'PENDING':
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Won", key=f"win_{hist_id}"):
                            conn = sqlite3.connect('football_intel.db', check_same_thread=False)
                            conn.execute("UPDATE predictions SET result='WON', won=1 WHERE id=?", (id_,))
                            conn.commit()
                            conn.close()
                            st.rerun()
                    with c2:
                        if st.button("❌ Lost", key=f"loss_{hist_id}"):
                            conn = sqlite3.connect('football_intel.db', check_same_thread=False)
                            conn.execute("UPDATE predictions SET result='LOST', won=0 WHERE id=?", (id_,))
                            conn.commit()
                            conn.close()
                            st.rerun()

st.divider()
st.caption("⚡ Strategic Intelligence V7.0 | Powered by AI | czunidaniel9@gmail.com")

