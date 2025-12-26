import streamlit as st
import requests
from datetime import datetime, timezone

# --- BIZTONSÁG ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except KeyError as e:
    st.error(f"Hiányzó API kulcs: {e}")
    st.stop()

# --- MODULOK ---
def get_referee_stats(referee_name="Ismeretlen"):
    ref_db = {
        "Michael Oliver": {"yellow_avg": 3.8, "bias": "Hazai pálya felé hajló"},
        "Anthony Taylor": {"yellow_avg": 3.9, "bias": "Szigorú"},
        "Szymon Marciniak": {"yellow_avg": 4.2, "bias": "Semleges"}
    }
    return ref_db.get(referee_name, {"name": referee_name, "yellow_avg": 3.9, "bias": "Átlagos"})

def get_news_and_sentiment(team_name):
    try:
        url = f"https://newsapi.org/v2/everything?q={team_name} football&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        res = requests.get(url, timeout=5).json()
        articles = res.get('articles', [])
        if not articles:
            return "Nincs friss hír", 0
        
        summary = articles[0]['title']
        sentiment = 0
        text = " ".join([a['title'].lower() for a in articles])
        if any(word in text for word in ['win', 'strong', 'fit', 'back']): sentiment = 1
        if any(word in text for word in ['injury', 'out', 'doubt', 'loss']): sentiment = -1
        
        return summary, sentiment
    except:
        return "Hírek jelenleg nem elérhetők", 0

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric&lang=hu"
        res = requests.get(url, timeout=5).json()
        return {'temp': res['main']['temp'], 'desc': res['weather'][0]['description'], 'wind': res['wind']['speed']}
    except:
        return {'temp': 15, 'desc': 'Nincs adat', 'wind': 5}

# --- MOTOR ---
class FootballIntelligenceV62:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    def analyze_match(self, m):
        home, away = m['home_team'], m['away_team']
        # Bukmékerek szűrése
        bookies = [b for b in m.get('bookmakers', []) if b['key'] in ['pinnacle', 'bet365', 'unibet']]
        if not bookies: return None

        offers = []
        for b in bookies:
            h2h = next((mk for mk in b.get('markets', []) if mk['key'] == 'h2h'), None)
            if h2h:
                for o in h2h['outcomes']:
                    offers.append({'name': o['name'], 'price': float(o['price'])})

        if not offers: return None
        fav_name = min(offers, key=lambda x: x['price'])['name']
        best_odds = max(o['price'] for o in offers if o['name'] == fav_name)

        if not (1.35 <= best_odds <= 1.75): return None

        # Adatgyűjtés
        news_headline, sentiment = get_news_and_sentiment(fav_name)
        weather = get_weather(home.split()[-1])
        ref = get_referee_stats("Ismeretlen")

        # Pontozás logika
        score = 75
        score += (sentiment * 10)
        if 1.50 <= best_odds <= 1.65: score += 5
        if weather['wind'] > 15: score -= 10

        # Dinamikus indoklás
        reason = f"A választás alapja a(z) {fav_name} stabil piaci pozíciója ({best_odds}). "
        if sentiment > 0: reason += "A friss hírek pozitív hangvételűek. "
        elif sentiment < 0: reason += "A hírek bizonytalanságot jeleznek, de az odds még így is értékálló. "
        
        reason += f"Az időjárás ({weather['desc']}) és a szél ({weather['wind']} m/s) nem gátolja a folyamatos játékot."

        return {
            'match': f"{home} vs {away}", 'pick': fav_name, 'odds': best_odds,
            'score': min(100, max(0, score)), 'weather': weather, 'referee': ref,
            'news': news_headline, 'reasoning': reason
        }

    def get_picks(self):
        leagues = ['soccer_epl', 'soccer_spain_la_liga', 'soccer_italy_serie_a', 'soccer_germany_bundesliga']
        results = []
        for lg in leagues:
            try:
                url = f"{self.base_url}/{lg}/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
                data = requests.get(url, timeout=10).json()
                for m in data:
                    res = self.analyze_match(m)
                    if res: results.append(res)
            except: continue
        return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

# --- UI ---
st.set_page_config(page_title="Football Intelligence V6.2", layout="wide")
st.title("🛡️ Football Intelligence V6.2 PRO")

if st.button("🚀 AZONNALI ELEMZÉS"):
    bot = FootballIntelligenceV62()
    with st.spinner("Mélyelemzés folyamatban (Hírek + Időjárás + Odds)..."):
        picks = bot.get_picks()
        if not picks:
            st.warning("Jelenleg nincs a stratégiai feltételeknek megfelelő mérkőzés.")
        else:
            for p in picks:
                with st.expander(f"🎯 {p['match']} - Odds: {p['odds']}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Magabiztosság", f"{p['score']}%")
                        st.write(f"**Tipp:** {p['pick']}")
                    with c2:
                        st.write(f"**☁️ Időjárás:** {p['weather']['temp']}°C")
                        st.write(f"**👨‍⚖️ Bíró:** {p['referee']['name']}")
                    with c3:
                        st.write(f"**📰 Friss hír:**")
                        st.caption(p['news'])
                    
                    st.info(f"💡 **Szakmai indoklás:** {p['reasoning']}")

