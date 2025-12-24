import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except:
    st.error("Kérlek add meg az ODDS_API_KEY-t a Streamlit Secrets-ben!")
    st.stop()

class StrategicFootballBot:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.TARGET_ODDS = 1.50 # Stratégiai cél: 1.50 * 1.50 = 2.25

    @st.cache_data(ttl=3600)
    def discover_soccer_leagues(self):
        try:
            res = requests.get(f"{self.base_url}?apiKey={ODDS_API_KEY}")
            res.raise_for_status()
            return [s['key'] for s in res.json() if s['group'] == 'Soccer']
        except Exception as e:
            st.sidebar.error(f"Liga hiba: {e}")
            return ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga']

    def find_h2h_market(self, bookie):
        for mkt in bookie.get("markets", []):
            if mkt.get("key") == "h2h":
                return mkt
        return None

    def analyze_markets(self):
        leagues = self.discover_soccer_leagues()
        picks_by_match = {} 
        now = datetime.now(timezone.utc)

        for league in leagues:
            url = f"{self.base_url}/{league}/odds"
            params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
            try:
                res = requests.get(url, params=params, timeout=10)
                res.raise_for_status()
                data = res.json()

                for m in data:
                    kickoff = datetime.fromisoformat(m['commence_time'].replace('Z', '+00:00'))
                    if kickoff < now or kickoff > now + timedelta(hours=24):
                        continue

                    match_key = f"{m['home_team']}|{m['away_team']}|{m['commence_time']}"
                    best_row_for_this_match = None

                    for bookie in m.get('bookmakers', []):
                        if bookie.get('key') not in ['pinnacle', 'bet365', 'unibet']:
                            continue

                        h2h = self.find_h2h_market(bookie)
                        if not h2h: continue

                        outcomes = h2h.get("outcomes", [])
                        if not outcomes: continue # JAVÍTÁS (3): Üres lista kezelése

                        best_o = min(outcomes, key=lambda x: float(x.get("price", 999)))
                        price = float(best_o["price"])

                        if 1.35 <= price <= 1.65:
                            implied_p = 1.0 / price
                            row = {
                                'match': f"{m['home_team']} vs {m['away_team']}",
                                'pick': best_o['name'],
                                'odds': price,
                                'p_win': implied_p, # JAVÍTÁS (1): Őszinte Toy p_win
                                'kickoff': kickoff,
                                'league': league
                            }

                            # JAVÍTÁS: Adott meccshez és pickhez a legjobb oddsot tartjuk meg
                            if best_row_for_this_match is None or row['odds'] > best_row_for_this_match['odds']:
                                best_row_for_this_match = row

                    if best_row_for_this_match:
                        picks_by_match[match_key] = best_row_for_this_match

            except: continue

        return list(picks_by_match.values())

# --- UI ---
st.set_page_config(page_title="Strategic Duplázó V5.1", page_icon="🎯")
st.title("🎯 Strategic Football Duplázó")
st.caption("Stratégia: Target Odds (1.50) | Időbeli korreláció-szűrés | Toy Mode (Implied Prob)")

if st.button("🚀 OPTIMÁLIS DUPLÁZÓ KERESÉSE"):
    bot = StrategicFootballBot()
    with st.spinner("Piacok elemzése és stratégiai illesztés..."):
        data = bot.analyze_markets()
        
        if len(data) >= 2:
            # JAVÍTÁS (2): Rendezés a TARGET (1.50) odds-hoz való közelség alapján
            data.sort(key=lambda x: abs(x['odds'] - bot.TARGET_ODDS))
            
            p1 = data[0]
            p2 = None
            for candidate in data[1:]:
                # Korreláció-szűrés: legalább 1 óra eltolódás VAGY más liga
                time_diff = abs((candidate['kickoff'] - p1['kickoff']).total_seconds()) / 60
                if time_diff > 60 or candidate['league'] != p1['league']:
                    p2 = candidate
                    break
            
            if p1 and p2:
                total_odds = p1['odds'] * p2['odds']
                # Mivel Toy Mode, az edge-t 0-nak vesszük, de a tőkét kezeljük (fix 2% javaslat)
                st.success(f"### 🎯 Szelvény Összeállítva (Eredő: {total_odds:.2f})")
                
                c1, c2 = st.columns(2)
                for i, p in enumerate([p1, p2]):
                    with [c1, c2][i]:
                        st.info(f"**{p['match']}**\nTipp: **{p['pick']}** | Odds: **{p['odds']}**")
                        st.write(f"⏰ {p['kickoff'].strftime('%H:%M')} | 🏆 {p['league']}")
                
                st.divider()
                st.metric("Javasolt Tét", "Fix 2% Bankroll")
                st.caption("A javaslat tisztán stratégiai és esélyalapú, nem prediktív modell eredménye.")
            else:
                st.warning("Nem sikerült korrelációmentes párost találni a cél-odds közelében.")
        else:
            st.error("Nincs elég adat a stratégia végrehajtásához.")
