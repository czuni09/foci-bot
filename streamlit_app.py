import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except:
    st.error("Kérlek add meg az ODDS_API_KEY-t a Streamlit Secrets-ben!")
    st.stop()

class FootballIntelligenceV5Pro:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    @st.cache_data(ttl=3600)
    def discover_soccer_leagues(self):
        try:
            res = requests.get(f"{self.base_url}?apiKey={ODDS_API_KEY}")
            res.raise_for_status()
            return [s['key'] for s in res.json() if s['group'] == 'Soccer']
        except Exception as e:
            st.error(f"Hiba a ligák lekérésekor: {e}")
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

                        best_o = min(h2h.get("outcomes", []), key=lambda x: float(x.get("price", 999)))
                        price = float(best_o["price"])

                        if 1.35 <= price <= 1.65:
                            implied_p = 1.0 / price
                            p_win = implied_p + 0.02  # Toy Mode edge
                            
                            # JAVÍTÁS (1): Score fix - ne torzítson az odds nagysága
                            score = (p_win - implied_p) 

                            row = {
                                'match': f"{m['home_team']} vs {m['away_team']}",
                                'pick': best_o['name'],
                                'odds': price,
                                'p_win': p_win,
                                'kickoff': kickoff,
                                'league': league,
                                '_score': score
                            }

                            # JAVÍTÁS (3): Legjobb odds kiválasztása ugyanarra a pickre
                            if best_row_for_this_match is None:
                                best_row_for_this_match = row
                            else:
                                if row['pick'] == best_row_for_this_match['pick']:
                                    if row['odds'] > best_row_for_this_match['odds']:
                                        best_row_for_this_match = row
                                elif row['_score'] > best_row_for_this_match['_score']:
                                    best_row_for_this_match = row

                    if best_row_for_this_match:
                        prev = picks_by_match.get(match_key)
                        if (prev is None) or (best_row_for_this_match['_score'] > prev['_score']):
                            picks_by_match[match_key] = best_row_for_this_match

            except Exception as e:
                # JAVÍTÁS (4): Hiba jelzése debug célból
                st.sidebar.write(f"⚠️ Hiba ({league}): {e}")
                continue

        return list(picks_by_match.values())

# --- UI ÉS LOGIKA ---
st.set_page_config(page_title="Football Intel V5 PRO", page_icon="⚽")
st.title("🏆 Football Intelligence V5 PRO")
st.caption("Auto-Dedupe | 24h Window | Correlation Filter | Kelly Safety Cap")

if st.button("🚀 MAI DUPLÁZÓ GENERÁLÁSA"):
    bot = FootballIntelligenceV5Pro()
    with st.spinner("Piacok elemzése és szűrése..."):
        data = bot.analyze_markets()
        
        if len(data) >= 2:
            data.sort(key=lambda x: x['p_win'], reverse=True)
            
            p1 = data[0]
            # JAVÍTÁS (2): Korreláció szűrés (időpont és liga)
            p2 = None
            for candidate in data[1:]:
                # Legalább 60 perc különbség VAGY más liga
                time_diff = abs((candidate['kickoff'] - p1['kickoff']).total_seconds()) / 60
                if time_diff > 60 or candidate['league'] != p1['league']:
                    p2 = candidate
                    break
            
            if p1 and p2:
                total_odds = p1['odds'] * p2['odds']
                p_combo = p1['p_win'] * p2['p_win'] * 0.95
                edge = (p_combo * total_odds) - 1
                
                raw_stake = edge / (total_odds - 1) if total_odds > 1 else 0
                safe_stake = min(max(raw_stake * 0.1, 0), 0.03)

                if safe_stake > 0:
                    st.success(f"### 🎯 Javasolt Szelvény (Eredő: {total_odds:.2f})")
                    c1, c2 = st.columns(2)
                    for i, p in enumerate([p1, p2]):
                        with [c1, c2][i]:
                            st.info(f"**{p['match']}**\nTipp: **{p['pick']}** | Odds: **{p['odds']}**")
                            st.write(f"⏰ Kezdés: {p['kickoff'].strftime('%H:%M')} | 🏆 {p['league']}")
                    
                    st.divider()
                    st.metric("Javasolt Tét (Bankroll %)", f"{safe_stake:.2%}")
                else:
                    st.warning("Nincs elegendő elméleti edge a piacon. (NO BET)")
            else:
                st.error("Nem sikerült korrelációmentes meccseket találni.")
        else:
            st.error("Nincs elég adat a következő 24 órára.")
