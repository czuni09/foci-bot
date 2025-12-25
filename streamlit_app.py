import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
except KeyError as e:
    st.error(f"Kritikus hiba: Hiányzó API kulcs: {e}")
    st.stop()

class FootballIntelligenceV531:
    """
    V5.3.1 Hardened PRO: Teljes árverseny minden kimenetre + Optimalizált erőforrás-kezelés.
    """
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.TARGET_ODDS = 1.50

    def get_weather(self, city, kickoff_time):
        """Időjárás lekérése csak ha a helyszín ismert."""
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric"
            res = requests.get(url, timeout=5)
            res.raise_for_status()
            data = res.json()
            if 'list' in data:
                for f in data['list']:
                    f_time = datetime.fromtimestamp(f['dt'], tz=timezone.utc)
                    if abs((f_time - kickoff_time).total_seconds()) < 7200:
                        return f"{f['main']['temp']:.1f}°C, {f['weather'][0]['description']}"
            return "Nincs közeli előrejelzés"
        except:
            return "Időjárás adat nem elérhető"

    @st.cache_data(ttl=3600)
    def discover_soccer_leagues(_self):
        try:
            res = requests.get(f"{_self.base_url}?apiKey={ODDS_API_KEY}")
            res.raise_for_status()
            return [s['key'] for s in res.json() if s['group'] == 'Soccer']
        except:
            return ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga']

    def analyze_markets(self):
        leagues = self.discover_soccer_leagues()
        picks_by_match = {} 
        now = datetime.now(timezone.utc)
        limit_24h = now + timedelta(hours=24)

        for league in leagues:
            url = f"{self.base_url}/{league}/odds"
            params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                for m in data:
                    kickoff = datetime.fromisoformat(m['commence_time'].replace('Z', '+00:00'))
                    if kickoff < now or kickoff > limit_24h:
                        continue

                    match_id = f"{m['home_team']}|{m['away_team']}|{m['commence_time']}"
                    
                    # V5.3.1: Minden kimenet begyűjtése az árversenyhez
                    offers = []
                    for bookie in m.get('bookmakers', []):
                        if bookie.get('key') not in ['pinnacle', 'bet365', 'unibet']:
                            continue

                        h2h = next((mk for mk in bookie.get("markets", []) if mk.get("key") == "h2h"), None)
                        if not h2h: continue

                        outcomes = h2h.get("outcomes", [])
                        for o in outcomes:
                            name = o.get("name")
                            price = float(o.get("price", 999))
                            if name and price < 900:
                                offers.append({"pick": name, "odds": price})

                    if not offers:
                        continue

                    # Globális favorit rögzítése a teljes kínálat alapján
                    global_fav_pick = min(offers, key=lambda o: o["odds"])["pick"]
                    
                    # Szűrés: a favorit minimum ára a tartományban legyen (1.35 - 1.65)
                    global_min_price = min(o["odds"] for o in offers if o["pick"] == global_fav_pick)
                    
                    if 1.35 <= global_min_price <= 1.65:
                        # V5.3.1: Valódi BEST PRICE keresés a rögzített kimenetre
                        best_price = max(o["odds"] for o in offers if o["pick"] == global_fav_pick)
                        
                        picks_by_match[match_id] = {
                            'match': f"{m['home_team']} vs {m['away_team']}",
                            'home': m['home_team'],
                            'away': m['away_team'],
                            'venue': None, # Jelenleg nincs város API
                            'pick': global_fav_pick,
                            'odds': best_price,
                            'kickoff': kickoff,
                            'league': league
                        }
            except Exception as e:
                st.sidebar.warning(f"Hiba ({league}): {str(e)[:50]}")
                continue

        return list(picks_by_match.values())

# --- UI ---
st.set_page_config(page_title="Strategic PRO V5.3.1", page_icon="⚽", layout="wide")
st.title("🛡️ Strategic Football Intelligence V5.3.1")

if st.button("🚀 MAI OPTIMÁLIS DUPLÁZÓ GENERÁLÁSA", type="primary"):
    bot = FootballIntelligenceV531()
    with st.spinner("Globális árverseny és piaci szűrés..."):
        data = bot.analyze_markets()
        
        if len(data) >= 2:
            data.sort(key=lambda x: abs(x['odds'] - bot.TARGET_ODDS))
            
            p1 = data[0]
            p2 = None
            
            for candidate in data[1:]:
                teams_p1 = {p1['home'], p1['away']}
                teams_cand = {candidate['home'], candidate['away']}
                
                if not teams_p1.intersection(teams_cand):
                    time_diff = abs((candidate['kickoff'] - p1['kickoff']).total_seconds()) / 60
                    if time_diff > 60 or candidate['league'] != p1['league']:
                        p2 = candidate
                        break
            
            if p1 and p2:
                total_odds = p1['odds'] * p2['odds']
                st.success(f"### 🎯 Javasolt Szelvény (Eredő: {total_odds:.2f})")
                
                c1, c2 = st.columns(2)
                for idx, p in enumerate([p1, p2]):
                    with [c1, c2][idx]:
                        # V5.3.1 optimalizált hívás: csak ha van helyszín
                        weather = "Város nem meghatározható (kihagyva)" if not p['venue'] else bot.get_weather(p['venue'], p['kickoff'])
                        st.info(f"**{p['match']}**")
                        st.write(f"Tipp: **{p['pick']}** | Szorzó: **{p['odds']}**")
                        st.write(f"⏰ {p['kickoff'].strftime('%m-%d %H:%M')} | ☁️ {weather}")
                
                st.divider()
                st.metric("Javasolt Tét", "Bankroll 2%-a")
            else:
                st.warning("Nem sikerült korrelációmentes párost találni.")
        else:
            st.error("Nincs elég adat a szűrési feltételeknek megfelelően.")

