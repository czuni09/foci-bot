import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG: Kulcsok beolvasása a titkos tárolóból ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
    # Opcionális kulcsok
    NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")
except KeyError:
    st.error("Hiba: Hiányzó kulcsok! Ellenőrizd a Streamlit Secrets beállításait (ODDS_API_KEY, WEATHER_API_KEY).")
    st.stop()

class FootballIntelligenceProV5:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.TARGET_ODDS = 1.50 # Stratégiai cél: 1.50 * 1.50 = ~2.25 eredő

    def get_weather(self, city, kickoff_time):
        """Időjárás előrejelzés lekérése a megadott kulccsal"""
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
            return "Időjárás adat nem elérhető"

    @st.cache_data(ttl=3600)
    def discover_soccer_
