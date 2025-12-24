import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG: Kulcsok beolvasása a titkos tárolóból ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
except KeyError:
    st.error("Hiba: Hiányzó kulcsok! Ellenőrizd a Streamlit Cloud Secrets beállításait (ODDS_API_KEY, WEATHER_API_KEY).")
    st.stop()

class FootballIntelligenceProV5:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.TARGET_ODDS = 1.50 

    def get_weather(self, city, kickoff_time):
        if not city: return "Helyszín ismeretlen"
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric"
            res = requests.get(url, timeout=5).json()
            if 'list' in res:
                for f in res['list']:
                    f_time = datetime.fromtimestamp(f['dt'], tz=timezone.utc)
                    if abs((f_time - kickoff_time).total_seconds()) < 7200:
                        return f"{f['main']['temp']:.1f}°C, {f['weather'][0]['description']}"
            return "Nincs közeli előrejelzés"
        except:
            return "Időjárás nem elérhető"

    @st.cache_data(ttl=3600)
    def discover_soccer_leagues(_self):
        """Dinamikus ligafelfedezés - fix _self paraméterrel"""
        try:
            res = requests.get(f"{_self.base_url}?apiKey={ODDS_API_KEY}")
            res.raise_for_status()
            return [s['key'] for s in res.json() if s['group'] == 'Soccer']
        except:
            return ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga', 'soccer_italy_serie_a']

    def analyze_markets(self):
        leagues = self.discover_soccer_leagues()
        picks_by_match = {} 
        now =
