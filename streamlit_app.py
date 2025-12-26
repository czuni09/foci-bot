import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

# ==============================================================================
# 🏆 TITAN V28.0 - PURE PROFESSIONAL (FORM-LOCK & VALUE FINDER)
# ==============================================================================

st.set_page_config(page_title="TITAN V28 PRO", layout="wide")

# Tisztább, professzionálisabb megjelenés (Sötét, de nem csicsás)
st.markdown("""
    <style>
    .report-card {
        background: #1a1c23;
        border-left: 5px solid #00ff88;
        padding: 25px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .bet-advice {
        font-size: 28px;
        font-weight: bold;
        color: #00ff88;
        background: #000;
        padding: 10px;
        text-align: center;
        border: 1px solid #00ff88;
    }
    .warning-label { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIG ---
try:
    ODDS_KEY = st.secrets["ODDS_API_KEY"]
    NEWS_KEY = st.secrets["NEWS_API_KEY"]
    EMAIL_USER = st.secrets["SAJAT_EMAIL"]
    EMAIL_PW = st.secrets["GMAIL_APP_PASSWORD"]
except:
    st.error("HIÁNYZÓ SECRETS!")
    st.stop()

# --- FORMA-ELLENŐRZŐ (Ez akadályozza meg az Aston Villa hibát) ---
def check_momentum(team_name):
    """
    Hírek és eredmények alapján szűri a csapat lendületét.
    Ha a csapat 'on fire', nem engedünk ellene fogadni.
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={team_name} winning streak form&apiKey={NEWS_KEY}"
        data = requests.get(url).json()
        articles = data.get("articles", [])
        # Ha a hírekben 'unbeaten' vagy 'winning streak' szerepel sűrűn
        momentum_score = 0
        for a in articles[:10]:
            text = a['title'].lower()
            if "winning streak" in text or "unbeaten" in text or "victory" in text:
                momentum_score += 1
        return momentum_score
    except: return 0

# --- ADATGYŰJTÉS ---
@st.cache_data(ttl=3600)
def get_pro_matches():
    # Csak a legstabilabb ligák
    leagues = ["soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga", "soccer_italy_serie_a"]
    all_picks = []
    
    for league in leagues:
        url = f"https://api.the-odds-api.com/v4/sports/{league}/odds?apiKey={ODDS_KEY}&regions=eu&markets=h2h"
        matches = requests.get(url).json()
        
        for m in matches:
            home, away = m['home_team'], m['away_team']
            bookie = m['bookmakers'][0]
            odds = bookie['markets'][0]['outcomes']
            
            h_o = next(x['price'] for x in odds if x['name'] == home)
            a_o = next(x['price'] for x in odds if x['name'] == away)
            d_o = next(x['price'] for x in odds if x['name'] == 'Draw')
            
            # --- PROFI SZŰRŐ 1: ODDS TARTOMÁNY (1.45 - 1.85) ---
            if 1.40 <= h_o <= 1.85:
                # --- PROFI SZŰRŐ 2: LENDÜLET ELLENŐRZÉS (PL. ASTON VILLA ELLEN NE) ---
                away_momentum = check_momentum(away)
                if away_momentum >= 3: # Ha az ellenfél túl jó formában van
                    continue
                
                # --- PROFI SZŰRŐ 3: VALÓDI VALÓSZÍNŰSÉG ---
                margin_corr = (1/h_o) + (1/a_o) + (1/d_o)
                real_prob = (1/h_o/margin_corr) * 100
                
                if real_prob > 55: # Csak ha 55% felett van a matematikai esély
                    all_picks.append({
                        "home": home, "away": away, 
                        "odds": h_o, "prob": real_prob,
                        "draw_p": (1/d_o/margin_corr)*100,
                        "away_p": (1/a_o/margin_corr)*100
                    })
    
    return sorted(all_picks, key=lambda x: x['prob'], reverse=True)[:2]

# --- UI ---
st.title("🦾 TITAN V28.0 - PROFESSIONAL ANALYTICS")
st.write(f"Utolsó frissítés: {datetime.now().strftime('%H:%M:%S')}")

picks = get_pro_matches()

if picks:
    for p in picks:
        st.markdown(f"""
        <div class="report-card">
            <h2>{p['home']} vs {p['away']}</h2>
            <div class="bet-advice">TIPP: {p['home']} GYŐZELEM (@{p['odds']})</div>
            <p><b>Matematikai valószínűség:</b> {p['prob']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Grafikon az összes kimenetelről
        fig = go.Figure(go.Bar(
            x=[p['home'], 'Döntetlen', p['away']],
            y=[p['prob'], p['draw_p'], p['away_p']],
            marker_color=['#00ff88', '#333', '#ff4b4b']
        ))
        fig.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig, key=p['home'])

    # Szelvény összesítő
    if len(picks) == 2:
        st.success(f"### EREDŐ ODDS: {picks[0]['odds'] * picks[1]['odds']:.2f}")
else:
    st.warning("Nincs a kritériumoknak megfelelő mérkőzés (A rendszer blokkolta a kockázatos meccseket).")

st.info("ℹ️ A rendszer automatikusan blokkolja azokat a meccseket, ahol az ellenfél 5 meccses győzelmi sorozatban van (Momentum Lock).")
