import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION & SECURITY ---
# A kulcsokat a Streamlit Secrets-ből húzzuk be a biztonság érdekében
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
except KeyError as e:
    st.error(f"Kritikus hiba: Hiányzó API kulcs a Secrets-ben: {e}")
    st.stop()

class FootballIntelligencePro:
    """
    Szakmai szintű fogadási elemző osztály.
    Főbb funkciók: Deduplikáció, stratégiai odds-szűrés, korreláció-kezelés.
    """
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.TARGET_ODDS = 1.50  # Ideális szorzó egy duplázó szelvényhez

    def get_weather(self, city, kickoff_time):
        """Időjárás előrejelzés lekérése a mérkőzés helyszínére és időpontjára."""
        if not city:
            return "Helyszín nem meghatározható"
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric"
            res = requests.get(url, timeout=5).json()
            if 'list' in res:
                for forecast in res['list']:
                    f_time = datetime.fromtimestamp(forecast['dt'], tz=timezone.utc)
                    # 2 órás idősávon belüli egyezés keresése
                    if abs((f_time - kickoff_time).total_seconds()) < 7200:
                        temp = forecast['main']['temp']
                        desc = forecast['weather'][0]['description']
                        return f"{temp:.1f}°C, {desc}"
            return "Nincs közeli előrejelzés"
        except Exception:
            return "Időjárás szolgáltatás nem elérhető"

    @st.cache_data(ttl=3600)
    def discover_soccer_leagues(_self):
        """Dinamikus ligafelfedezés. A _self prefix megakadályozza a cache hashing hibát."""
        try:
            res = requests.get(f"{_self.base_url}?apiKey={ODDS_API_KEY}")
            res.raise_for_status()
            return [s['key'] for s in res.json() if s['group'] == 'Soccer']
        except Exception:
            # Fallback mechanizmus a legfontosabb ligákra
            return ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga', 'soccer_italy_serie_a']

    def analyze_markets(self):
        """Piacelemzés, deduplikáció és stratégiai szűrés."""
        leagues = self.discover_soccer_leagues()
        picks_by_match = {} 
        now = datetime.now(timezone.utc)
        limit_24h = now + timedelta(hours=24)

        for league in leagues:
            url = f"{self.base_url}/{league}/odds"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'eu',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code != 200:
                    continue
                
                data = response.json()
                for m in data:
                    kickoff = datetime.fromisoformat(m['commence_time'].replace('Z', '+00:00'))
                    
                    # Időablak szűrése: csak a következő 24 óra
                    if kickoff < now or kickoff > limit_24h:
                        continue

                    match_id = f"{m['home_team']}|{m['away_team']}|{m['commence_time']}"
                    best_option_for_match = None

                    for bookie in m.get('bookmakers', []):
                        # Csak a hiteles, alacsony árréssel dolgozó irodák
                        if bookie.get('key') not in ['pinnacle', 'bet365', 'unibet']:
                            continue

                        h2h = next((mkt for mkt in bookie.get("markets", []) if mkt["key"] == "h2h"), None)
                        if not h2h:
                            continue

                        outcomes = h2h.get("outcomes", [])
                        if not outcomes:
                            continue

                        # A favorit kimenet keresése (legalacsonyabb szorzó)
                        fav_outcome = min(outcomes, key=lambda x: float(x.get("price", 999)))
                        price = float(fav_outcome["price"])

                        # Stratégiai tartomány szűrése (1.35 - 1.65)
                        if 1.35 <= price <= 1.65:
                            row = {
                                'match': f"{m['home_team']} vs {m['away_team']}",
                                'venue': m['home_team'],
                                'pick': fav_outcome['name'],
                                'odds': price,
                                'kickoff': kickoff,
                                'league': league
                            }
                            # Bookie-k közötti legjobb ár kiválasztása ugyanarra a meccsre
                            if best_option_for_match is None or row['odds'] > best_option_for_match['odds']:
                                best_option_for_match = row

                    if best_option_for_match:
                        picks_by_match[match_id] = best_option_for_match

            except Exception:
                continue

        return list(picks_by_match.values())

# --- USER INTERFACE (STREAMLIT) ---
st.set_page_config(page_title="Strategic Intel PRO", page_icon="⚽", layout="wide")

st.title("🛡️ Strategic Football Intelligence V5.2 PRO")
st.markdown("""
**Rendszerállapot:** Precíziós üzemmód aktív. 
*Szűrés: 24 órás ablak, deduplikált piacok, korrelációmentes szelvény-összeállítás.*
""")

if st.button("🚀 OPTIMÁLIS DUPLÁZÓ GENERÁLÁSA", type="primary"):
    bot = FootballIntelligencePro()
    with st.spinner("Globális piacok átvilágítása és matematikai szűrés..."):
        data = bot.analyze_markets()
        
        if len(data) >= 2:
            # Rendezés a TARGET (1.50) odds-hoz való távolság alapján (Abszolút hiba szerint)
            data.sort(key=lambda x: abs(x['odds'] - bot.TARGET_ODDS))
            
            p1 = data[0]
            p2 = None
            
            # Korreláció-szűrés: Minimum 60 perc eltolódás VAGY eltérő liga
            for candidate in data[1:]:
                time_diff_min = abs((candidate['kickoff'] - p1['kickoff']).total_seconds()) / 60
                if time_diff_min > 60 or candidate['league'] != p1['league']:
                    p2 = candidate
                    break
            
            if p1 and p2:
                total_odds = p1['odds'] * p2['odds']
                st.success(f"### 🎯 Javasolt Szelvény (Eredő szorzó: {total_odds:.2f})")
                
                col1, col2 = st.columns(2)
                for idx, p in enumerate([p1, p2]):
                    with [col1, col2][idx]:
                        weather = bot.get_weather(p['venue'], p['kickoff'])
                        st.info(f"**{p['match']}**")
                        st.write(f"🔹 Tipp: **{p['pick']}**")
                        st.write(f"🔹 Szorzó: **{p['odds']}**")
                        st.write(f"⏰ Kezdés: {p['kickoff'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"☁️ Időjárás: {weather}")
                
                st.divider()
                st.write("📈 **Kockázatkezelési javaslat:** Fix 2%-os bankroll menedzsment.")
            else:
                st.warning("Nem sikerült korrelációmentes párost találni a cél-odds közelében.")
        else:
            st.error("Nincs elegendő piaci adat a következő 24 órára.")
