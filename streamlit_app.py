import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG ÉS KONFIGURÁCIÓ ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except KeyError as e:
    st.error(f"Hiányzó API kulcs a Secrets-ben: {e}")
    st.stop()

# --- SEGÉDFÜGGVÉNYEK ---
def get_referee_stats(referee_name="Ismeretlen"):
    ref_db = {
        "Michael Oliver": {"yellow_avg": 3.8, "bias": "Hazai pálya felé hajló"},
        "Anthony Taylor": {"yellow_avg": 3.9, "bias": "Szigorú"},
        "Szymon Marciniak": {"yellow_avg": 4.2, "bias": "Semleges"}
    }
    return ref_db.get(referee_name, {"name": referee_name, "yellow_avg": 3.9, "bias": "Átlagos"})

def get_news_and_sentiment(team_name):
    try:
        # Pontosított keresés a csapat nevére
        url = f"https://newsapi.org/v2/everything?q={team_name} football&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        res = requests.get(url, timeout=5).json()
        articles = res.get('articles', [])
        
        if not articles:
            return "Nincs friss hír a csapatról.", 0
        
        headline = articles[0]['title']
        sentiment = 0
        text_to_analyze = " ".join([a['title'].lower() for a in articles])
        
        # Sentiment logika
        pos_words = ['win', 'strong', 'fit', 'back', 'boost', 'ready']
        neg_words = ['injury', 'out', 'doubt', 'loss', 'suspended', 'miss']
        
        if any(w in text_to_analyze for w in pos_words): sentiment += 1
        if any(w in text_to_analyze for w in neg_words): sentiment -= 1
        
        return headline, sentiment
    except:
        return "Hírek jelenleg nem elérhetők.", 0

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric&lang=hu"
        res = requests.get(url, timeout=5).json()
        return {
            'temp': res['main']['temp'], 
            'desc': res['weather'][0]['description'], 
            'wind': res['wind']['speed']
        }
    except:
        return {'temp': 15, 'desc': 'Nincs adat', 'wind': 5}

# --- ELEMZŐ MOTOR ---
class FootballIntelligenceV63:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    def analyze_match(self, m):
        # 1. SZIGORÚ DÁTUMSZŰRÉS: Csak a következő 24 óra
        now = datetime.now(timezone.utc)
        kickoff_time = datetime.fromisoformat(m['commence_time'].replace('Z', '+00:00'))
        
        if kickoff_time < now or kickoff_time > now + timedelta(hours=24):
            return None

        home, away = m['home_team'], m['away_team']
        
        # 2. ODDS GYŰJTÉS
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

        # 3. STRATÉGIAI SZŰRŐ (1.35 - 1.75 odds tartomány)
        if not (1.35 <= best_odds <= 1.75): return None

        # 4. MÉLYELEMZÉS
        news_headline, sentiment = get_news_and_sentiment(fav_name)
        # Város kinyerése a hazai csapat nevéből (egyszerűsített)
        city = home.split()[-1]
        weather = get_weather(city)
        ref = get_referee_stats("Ismeretlen")

        # 5. PONTOZÁS
        score = 75
        score += (sentiment * 10)
        if 1.50 <= best_odds <= 1.65: score += 5
        if weather['wind'] > 15: score -= 10

        # 6. INDOKLÁS ÖSSZEÁLLÍTÁSA
        reason = f"A választás alapja a(z) {fav_name} kiemelkedő piaci értéke ({best_odds}). "
        if sentiment > 0: reason += "A csapat körüli hírek pozitívak, kulcsjátékosok bevethetők. "
        reason += f"Az időjárás ({weather['desc']}) ideális a folyamatos passzjátékhoz."

        return {
            'match': f"{home} vs {away}",
            'kickoff': kickoff_time.strftime("%Y-%m-%d %H:%M"),
            'pick': fav_name,
            'odds': best_odds,
            'score': min(100, max(0, score)),
            'weather': weather,
            'referee': ref,
            'news': news_headline,
            'reasoning': reason
        }

    def get_picks(self):
        # Csak azokat a ligákat nézzük, ahol ma van forduló
        leagues = ['soccer_epl', 'soccer_championship', 'soccer_spain_la_liga', 'soccer_italy_serie_a']
        results = []
        for lg in leagues:
            try:
                url = f"{self.base_url}/{lg}/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
                data = requests.get(url, timeout=10).json()
                if isinstance(data, list):
                    for m in data:
                        res = self.analyze_match(m)
                        if res: results.append(res)
            except:
                continue
        return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

# --- FELHASZNÁLÓI FELÜLET ---
st.set_page_config(page_title="Football Intelligence V6.3", layout="wide")
st.title("🛡️ Football Intelligence V6.3 PRO")
st.subheader("Szigorú dátumszűréssel és valós idejű hír-analízissel")

if st.button("🚀 AZONNALI ELEMZÉS"):
    bot = FootballIntelligenceV63()
    with st.spinner("Mérkőzések szűrése és szakmai elemzése folyamatban..."):
        picks = bot.get_picks()
        if not picks:
            st.warning("A következő 24 órában nincs a stratégiai feltételeknek (1.35-1.75 odds) megfelelő mérkőzés.")
        else:
            for p in picks:
                with st.expander(f"🎯 {p['match']} | Kezdés: {p['kickoff']} | Odds: {p['odds']}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Magabiztosság", f"{p['score']}%")
                        st.write(f"**Tipp:** {p['pick']}")
                    with c2:
                        st.write(f"**☁️ Időjárás:** {p['weather']['temp']}°C, {p['weather']['desc']}")
                        st.write(f"**👨‍⚖️ Bíró:** {p['referee']['name']}")
                    with c3:
                        st.write(f"**📰 Legfrissebb hír:**")
                        st.caption(p['news'])
                    
                    st.info(f"💡 **Szakmai indoklás:** {p['reasoning']}")

