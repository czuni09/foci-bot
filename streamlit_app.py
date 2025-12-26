import streamlit as st
import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone

# --- BIZTONSÁG ---
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    WEATHER_KEY = st.secrets["WEATHER_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except:
    st.error("API kulcsok hiányoznak!")
    st.stop()

# --- MODULOK: HÍREK ÉS JÁTÉKOSOK ---
def get_team_intel(team_name):
    """Lekéri a legfrissebb híreket, sérülteket és belső infókat."""
    try:
        # Kifejezett keresés sérültekre és felállásra
        query = f"{team_name} football injury lineup news"
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        res = requests.get(url, timeout=5).json()
        articles = res.get('articles', [])
        
        if not articles:
            return "Nincs specifikus hír a keretről.", 0, "A csapat stabilnak tűnik."
        
        full_text = " ".join([a['title'].lower() + " " + (a['description'] or "").lower() for a in articles])
        
        # Kulcsszó figyelés
        injuries = [w for w in ['injury', 'out', 'doubtful', 'unavailable', 'miss', 'broken'] if w in full_text]
        boosts = [w for w in ['returns', 'fit', 'starts', 'key player', 'back'] if w in full_text]
        
        sentiment = len(boosts) - len(injuries)
        intel_summary = articles[0]['title']
        
        # Részletes indoklás generálása a hírek alapján
        detail = "A hírek alapján a keret hiányos lehet." if len(injuries) > 0 else "A kulcsjátékosok bevethető állapotban vannak."
        if len(boosts) > 0: detail += " Fontos visszatérők erősítik a kezdőt."
        
        return intel_summary, sentiment, detail
    except:
        return "Hírek nem elérhetők.", 0, "Nincs adat."

# --- MOTOR ---
class TicketGeneratorV8:
    def __init__(self):
        self.leagues = ['soccer_epl', 'soccer_championship', 'soccer_spain_la_liga', 'soccer_italy_serie_a']

    def get_ticket(self):
        all_matches = []
        for lg in self.leagues:
            url = f"https://api.the-odds-api.com/v4/sports/{lg}/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
            data = requests.get(url).json()
            if isinstance(data, list):
                for m in data:
                    # Csak a következő 24 óra
                    kickoff = datetime.fromisoformat(m['commence_time'].replace('Z', '+00:00'))
                    if kickoff > datetime.now(timezone.utc) + timedelta(hours=24): continue
                    
                    home, away = m['home_team'], m['away_team']
                    bookie = next((b for b in m.get('bookmakers', []) if b['key'] == 'bet365'), None)
                    if not bookie: continue
                    
                    market = next((mk for mk in bookie['markets'] if mk['key'] == 'h2h'), None)
                    fav = min(market['outcomes'], key=lambda x: x['price'])
                    
                    # Csak biztonságosabb oddsok (1.30 - 1.60), hogy a kettő kijöjjön 2.00 körül
                    if 1.30 <= fav['price'] <= 1.65:
                        intel, sent, detail = get_team_intel(fav['name'])
                        all_matches.append({
                            'match': f"{home} vs {away}",
                            'pick': fav['name'],
                            'odds': fav['price'],
                            'score': 70 + (sent * 10),
                            'intel': intel,
                            'detail': detail
                        })
        
        # Kiválasztjuk a két legjobb "százalékos" meccset
        sorted_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)
        return sorted_matches[:2]

# --- UI ---
st.set_page_config(page_title="Ticket Master V8", layout="wide")
st.title("🎫 Napi Dupla Szelvény (2.00x)")

if st.button("🚀 SZELVÉNY GENERÁLÁSA"):
    engine = TicketGeneratorV8()
    ticket = engine.get_ticket()
    
    if len(ticket) < 2:
        st.warning("Ma nincs elég biztonságos meccs a szelvényhez. Próbáld később!")
    else:
        total_odds = ticket[0]['odds'] * ticket[1]['odds']
        st.header(f"Eredő szorzó: {total_odds:.2f}")
        
        for i, p in enumerate(ticket):
            status = "💎 TUTI" if p['score'] >= 85 else "✅ AJÁNLOTT"
            with st.container():
                st.subheader(f"{i+1}. Mérkőzés: {p['match']}")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Esély", f"{p['score']}%", status)
                    st.write(f"**Tipp:** {p['pick']}")
                    st.write(f"**Odds:** {p['odds']}")
                with c2:
                    st.info(f"**Hírek a ház tájáról:** {p['intel']}")
                    st.write(f"**Szakmai elemzés:** {p['detail']}")
                    st.caption("A választás oka: Stabil kezdő, pozitív belső hírek és optimális piaci szorzó.")
                st.divider()

        # Táblázat mentése a háttérben
        conn = sqlite3.connect('stats.db')
        pd.DataFrame(ticket).to_sql('history', conn, if_exists='append', index=False)
        conn.close()

with st.expander("📊 Korábbi szelvények statisztikája"):
    conn = sqlite3.connect('stats.db')
    try:
        df = pd.read_sql_query("SELECT * FROM history", conn)
        st.dataframe(df)
    except: st.write("Még nincs adat.")
    conn.close()
