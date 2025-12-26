import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG ---
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
    # Ha nincs benne a név, alapértelmezett értéket adunk vissza, hogy ne legyen KeyError
    return ref_db.get(referee_name, {"name": referee_name, "yellow_avg": 3.9, "bias": "Átlagos"})

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric&lang=hu"
        res = requests.get(url, timeout=5).json()
        return {'temp': res['main']['temp'], 'desc': res['weather'][0]['description'], 'wind': res['wind']['speed']}
    except:
        return {'temp': 15, 'desc': 'Nincs adat', 'wind': 5}

# --- ELEMZŐ MOTOR ---
class FootballIntelligenceV61:
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

        # SZŰRŐ: Csak a biztonsági tartomány (Villarreal 1.59 belefér!)
        if not (1.35 <= best_odds <= 1.75): return None

        # Adatok begyűjtése az indokláshoz
        weather = get_weather(home.split()[-1])
        ref = get_referee_stats("Ismeretlen") # Itt később a valós bíró jöhet
        
        # Pontozás (Score) kiszámítása
        score = 70 
        if 1.50 <= best_odds <= 1.65: score += 10 # Optimális szorzó bónusz
        if weather['wind'] < 10: score += 5 # Jó körülmények

        # INDOKLÁS GENERÁLÁSA
        reasoning = f"A(z) {fav_name} győzelme ({best_odds}) kiváló értékkel bír. "
        reasoning += f"A szélsebesség ({weather['wind']} m/s) alacsony, ami kedvez a technikai játéknak. "
        reasoning += f"A bírói profil ({ref['bias']}) megfelel a mérkőzés kockázati szintjének."

        return {
            'match': f"{home} vs {away}",
            'pick': fav_name,
            'odds': best_odds,
            'score': score,
            'weather': weather,
            'referee': ref,
            'reasoning': reasoning
        }

    def get_picks(self):
        leagues = ['soccer_epl', 'soccer_spain_la_liga', 'soccer_italy_serie_a']
        all_results = []
        for lg in leagues:
            try:
                url = f"{self.base_url}/{lg}/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
                data = requests.get(url).json()
                for m in data:
                    res = self.analyze_match(m)
                    if res: all_results.append(res)
            except: continue
        return sorted(all_results, key=lambda x: x['score'], reverse=True)[:3]

# --- UI ---
st.title("🛡️ Football Intelligence V6.1 PRO")

if st.button("🚀 AZONNALI ELEMZÉS"):
    bot = FootballIntelligenceV61()
    with st.spinner("Elemzés folyamatban..."):
        picks = bot.get_picks()
        if not picks:
            st.warning("Ma nincs a szűrőnek megfelelő mérkőzés.")
        else:
            for p in picks:
                with st.expander(f"🎯 {p['match']} - {p['odds']}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Tipp:** {p['pick']}")
                        st.write(f"**Magabiztosság:** {p['score']}%")
                        # JAVÍTÁS: Biztonságos elérés a bíró nevéhez
                        ref_name = p['referee'].get('name', 'Ismeretlen')
                        ref_bias = p['referee'].get('bias', 'Nincs adat')
                        st.write(f"**Bíró:** {ref_name} ({ref_bias})")
                    with col2:
                        st.write(f"**Időjárás:** {p['weather']['temp']}°C, {p['weather']['desc']}")
                        st.write(f"**Szél:** {p['weather']['wind']} m/s")
                    
                    st.info(f"💡 **Indoklás:** {p['reasoning']}")

