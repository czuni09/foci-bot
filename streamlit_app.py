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
    st.error("Hiba: Az API kulcsok hiányoznak a Streamlit Secrets-ből!")
    st.stop()

# --- MÉLYELEMZŐ MODUL (HÍREK & JÁTÉKOSOK) ---
def get_deep_intel(team_name):
    try:
        # Keresés kifejezetten sérültekre és kulcsjátékosokra
        url = f"https://newsapi.org/v2/everything?q={team_name} team news injury lineup&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
        res = requests.get(url, timeout=5).json()
        articles = res.get('articles', [])
        
        if not articles:
            return 0, "Nincs friss belső hír, a csapat az alapértelmezett kerettel állhat ki.", "Stabil keret."

        text = " ".join([a['title'].lower() for a in articles])
        
        # Játékos/Sérülés detektálás
        bad_news = ['injury', 'out', 'doubt', 'suspended', 'missing', 'crisis', 'calf', 'hamstring']
        good_news = ['return', 'fit', 'back', 'starts', 'boost', 'training']
        
        score_mod = 0
        reasons = []
        
        for w in bad_news:
            if w in text:
                score_mod -= 12
                reasons.append(f"Sérülési hírek/hiányzók ({w})")
                break
        for w in good_news:
            if w in text:
                score_mod += 10
                reasons.append(f"Fontos visszatérők ({w})")
                break
                
        intel_text = articles[0]['title']
        detail = " | ".join(reasons) if reasons else "Nincs jelentős változás a keretben."
        return score_mod, intel_text, detail
    except:
        return 0, "Hírszolgáltatás átmenetileg szünetel.", "Nincs adat."

# --- MOTOR ---
class TicketMasterV9:
    def __init__(self):
        self.leagues = ['soccer_epl', 'soccer_championship', 'soccer_spain_la_liga', 'soccer_italy_serie_a', 'soccer_germany_bundesliga', 'soccer_france_ligue1']

    def generate(self):
        all_picks = []
        for lg in self.leagues:
            url = f"https://api.the-odds-api.com/v4/sports/{lg}/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
            try:
                data = requests.get(url, timeout=10).json()
                for m in data:
                    home, away = m['home_team'], m['away_team']
                    bookie = next((b for b in m.get('bookmakers', []) if b['key'] in ['bet365', 'unibet', 'pinnacle']), None)
                    if not bookie: continue
                    
                    market = next((mk for mk in bookie['markets'] if mk['key'] == 'h2h'), None)
                    fav = min(market['outcomes'], key=lambda x: x['price'])
                    
                    # Alap pontszám az odds alapján
                    base_score = 85 if fav['price'] < 1.50 else 75 if fav['price'] < 1.80 else 65
                    
                    # Hír alapú módosító
                    mod, news_head, news_det = get_deep_intel(fav['name'])
                    final_score = base_score + mod
                    
                    all_picks.append({
                        'match': f"{home} vs {away}",
                        'pick': fav['name'],
                        'odds': fav['price'],
                        'score': min(99, final_score),
                        'news': news_head,
                        'detail': news_det
                    })
            except: continue
        
        # Mindig adjon ki valamit: sorbarendezzük és a legjobb kettőt vesszük
        return sorted(all_picks, key=lambda x: x['score'], reverse=True)[:2]

# --- UI ---
st.set_page_config(page_title="Ticket Master V9.0", layout="wide")
st.title("🎫 Professzionális Dupla Szelvény Generátor")
st.markdown("---")

if st.button("🚀 SZELVÉNY ÖSSZEÁLLÍTÁSA"):
    engine = TicketMasterV9()
    with st.spinner("Hírek elemzése és szelvény kalkulálása..."):
        ticket = engine.generate()
        
        if not ticket:
            st.error("Hiba az adatok lekérésekor. Ellenőrizd az API kulcsokat!")
        else:
            total_odds = ticket[0]['odds'] * ticket[1]['odds']
            
            st.header(f"💰 Várható eredő odds: {total_odds:.2f}")
            
            for i, p in enumerate(ticket):
                # Szöveges ajánlás meghatározása
                if p['score'] >= 90: rec, color = "💎 TUTI TIPP", "green"
                elif p['score'] >= 80: rec, color = "✅ ERŐSEN AJÁNLOTT", "blue"
                else: rec, color = "⚠️ ÁTGONDOLÁSRA (Kockázatosabb)", "orange"
                
                with st.container():
                    st.subheader(f"{i+1}. Mérkőzés: {p['match']}")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Magabiztosság", f"{p['score']}%")
                        st.markdown(f"**Tipp:** {p['pick']}")
                        st.markdown(f"**Odds:** {p['odds']}")
                        st.markdown(f"### :{color}[{rec}]")
                        
                    with col2:
                        st.info(f"**Friss hírek a csapatnál:**\n\n{p['news']}")
                        st.success(f"**Szakmai indoklás:** {p['detail']}\n\n*Az elemzés a keret aktuális állapota és a piaci mozgások alapján készült.*")
                st.divider()

            # Mentés statisztikához
            conn = sqlite3.connect('pro_stats.db')
            pd.DataFrame(ticket).to_sql('results', conn, if_exists='append', index=False)
            conn.close()

with st.expander("📊 Adatbázis és Statisztika (Múltbéli tippek)"):
    try:
        conn = sqlite3.connect('pro_stats.db')
        df = pd.read_sql_query("SELECT * FROM results", conn)
        st.dataframe(df, use_container_width=True)
        conn.close()
    except:
        st.write("Még nincs mentett adat az adatbázisban.")

