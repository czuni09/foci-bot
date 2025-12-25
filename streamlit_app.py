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

class FootballIntelligenceV532:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.TARGET_ODDS = 1.50

    @st.cache_data(ttl=3600)
    def discover_soccer_leagues(_self):
        try:
            res = requests.get(f"{_self.base_url}?apiKey={ODDS_API_KEY}")
            res.raise_for_status()
            # Kiszűrjük a 'winner' (végső győztes) típusú piacokat, mert azok nem meccsek
            return [s['key'] for s in res.json() if s['group'] == 'Soccer' and 'winner' not in s['key']]
        except:
            return ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga']

    def analyze_markets(self):
        leagues = self.discover_soccer_leagues()
        picks_by_match = {} 
        now = datetime.now(timezone.utc)
        limit_24h = now + timedelta(hours=24)
        total_scanned = 0

        for league in leagues:
            url = f"{self.base_url}/{league}/odds"
            params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 422: continue # Outright piac skip
                response.raise_for_status()
                data = response.json()

                for m in data:
                    total_scanned += 1
                    kickoff = datetime.fromisoformat(m['commence_time'].replace('Z', '+00:00'))
                    if kickoff < now or kickoff > limit_24h: continue

                    match_id = f"{m['home_team']}|{m['away_team']}"
                    offers = []
                    for bookie in m.get('bookmakers', []):
                        if bookie.get('key') not in ['pinnacle', 'bet365', 'unibet']: continue
                        h2h = next((mk for mk in bookie.get("markets", []) if mk.get("key") == "h2h"), None)
                        if not h2h: continue

                        for o in h2h.get("outcomes", []):
                            if o.get("name") and float(o.get("price", 999)) < 900:
                                offers.append({"pick": o["name"], "odds": float(o["price"])})

                    if not offers: continue
                    global_fav_pick = min(offers, key=lambda o: o["odds"])["pick"]
                    global_min_price = min(o["odds"] for o in offers if o["pick"] == global_fav_pick)
                    
                    if 1.35 <= global_min_price <= 1.65:
                        best_price = max(o["odds"] for o in offers if o["pick"] == global_fav_pick)
                        picks_by_match[match_id] = {
                            'match': f"{m['home_team']} vs {m['away_team']}",
                            'home': m['home_team'], 'away': m['away_team'],
                            'pick': global_fav_pick, 'odds': best_price,
                            'kickoff': kickoff, 'league': league
                        }
            except: continue

        return list(picks_by_match.values()), total_scanned

# --- UI ---
st.set_page_config(page_title="Strategic PRO V5.3.2", page_icon="⚽", layout="wide")
st.title("🛡️ Strategic Football Intelligence V5.3.2")

if st.button("🚀 OPTIMÁLIS DUPLÁZÓ GENERÁLÁSA", type="primary"):
    bot = FootballIntelligenceV532()
    with st.spinner("Piacok elemzése..."):
        data, count = bot.analyze_markets()
        st.sidebar.write(f"Vizsgált események: {count}")
        
        if len(data) >= 2:
            data.sort(key=lambda x: abs(x['odds'] - bot.TARGET_ODDS))
            p1 = data[0]
            p2 = None
            for candidate in data[1:]:
                if not {p1['home'], p1['away']}.intersection({candidate['home'], candidate['away']}):
                    if abs((candidate['kickoff']-p1['kickoff']).total_seconds())/60 > 60 or candidate['league'] != p1['league']:
                        p2 = candidate
                        break
            
            if p1 and p2:
                st.success(f"### 🎯 Szelvény (Eredő: {p1['odds']*p2['odds']:.2f})")
                c1, c2 = st.columns(2)
                for idx, p in enumerate([p1, p2]):
                    with [c1, c2][idx]:
                        st.info(f"**{p['match']}**\nTipp: **{p['pick']}** | Odds: **{p['odds']}**")
            else: st.warning("Nincs korrelációmentes pár.")
        else:
            st.error(f"Nincs elég meccs a szűrésnek megfelelően. (Összesen {count} meccset néztem át a következő 24 órára).")
            if count > 0:
                st.info("💡 Tipp: Karácsony van, a legtöbb liga szünetel. Próbáld meg 26-án (Boxing Day), amikor visszatér a Premier League!")
