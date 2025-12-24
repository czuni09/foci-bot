import streamlit as st
import requests
import os
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Logging beállítása
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# KONFIGURÁCIÓ (BEÉPÍTETT KULCSOKKAL)
# ============================================
ODDS_API_KEY = "cc1a32d7a1d30cb4898eb879ff6d636f"
GMAIL_APP_PASSWORD = "whppzywzoduqjrgk"
SAJAT_EMAIL = "czunidaniel9@gmail.com"

class FootballIntelligence:
    def __init__(self):
        self.odds_key = ODDS_API_KEY
        
    def get_matches_with_odds(self, sport: str = 'soccer_epl') -> List[Dict]:
        """Élő oddsok lekérése hibaellenőrzéssel"""
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                'apiKey': self.odds_key,
                'regions': 'eu',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Odds API hiba: {e}")
            return []

    def analyze_match(self, match: Dict, target_odds: float = 2.0) -> Optional[Dict]:
        """Meccs elemzése és 2.00 körüli kimenetel keresése"""
        home = match.get('home_team', 'Ismeretlen')
        away = match.get('away_team', 'Ismeretlen')
        
        bookmakers = match.get('bookmakers', [])
        if not bookmakers: return None
        
        best_odds = {'home': 0, 'draw': 0, 'away': 0}
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    for outcome in market.get('outcomes', []):
                        price = outcome.get('price', 0)
                        if outcome['name'] == home: best_odds['home'] = max(best_odds['home'], price)
                        elif outcome['name'] == away: best_odds['away'] = max(best_odds['away'], price)
                        else: best_odds['draw'] = max(best_odds['draw'], price)
        
        picks = []
        for outcome, odd in best_odds.items():
            if 1.85 <= odd <= 2.25: # Duplázó tartomány
                confidence = 65 if outcome == 'home' else 55
                picks.append({'pick': outcome.upper(), 'odds': odd, 'confidence': confidence})
        
        if not picks: return None
        picks.sort(key=lambda x: abs(x['odds'] - target_odds))
        
        return {
            'match': f"{home} vs {away}",
            'commence_time': match.get('commence_time', 'Ismeretlen'),
            'odds': best_odds,
            'recommendation': picks[0]
        }

    def find_best_bets(self, leagues: List[str], target_odds: float = 2.0) -> List[Dict]:
        all_picks = []
        for league in leagues:
            matches = self.get_matches_with_odds(league)
            for match in matches:
                analysis = self.analyze_match(match, target_odds)
                if analysis: all_picks.append(analysis)
        all_picks.sort(key=lambda x: x['recommendation']['confidence'], reverse=True)
        return all_picks

# ============================================
# EMAIL KÜLDÉS FUNKCIÓ
# ============================================
def send_email_report(content):
    msg = MIMEMultipart()
    msg['Subject'] = f"🎯 Napi Duplázó Elemzés - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg.attach(MIMEText(content, 'plain', 'utf-8'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        return True
    except:
        return False

# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="Football Intelligence", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .match-card { background: #1E2A38; padding: 20px; border-radius: 10px; border-left: 5px solid #00D1B2; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚽ czunidaniel9 Intelligence Dashboard")
st.sidebar.title("⚙️ Beállítások")

target = st.sidebar.slider("Cél Odds", 1.5, 3.0, 2.0, 0.1)
leagues = st.sidebar.multiselect("Ligák", 
    ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga', 'soccer_italy_serie_a', 'soccer_uefa_champs_league'],
    default=['soccer_epl', 'soccer_spain_la_liga'])

if st.button("🚀 ELEMZÉS ÉS EMAIL KÜLDÉSE", type="primary"):
    intel = FootballIntelligence()
    results = intel.find_best_bets(leagues, target)
    
    if results:
        report_text = "MAI TOP TIPPEK:\n\n"
        for idx, res in enumerate(results[:3], 1):
            st.markdown(f"""<div class="match-card">
                <h3>{idx}. {res['match']}</h3>
                <p>🎯 Tipp: <b>{res['recommendation']['pick']}</b> | 💰 Odds: <b>{res['recommendation']['odds']}</b></p>
                </div>""", unsafe_allow_html=True)
            report_text += f"{idx}. {res['match']}\nTipp: {res['recommendation']['pick']} | Odds: {res['recommendation']['odds']}\n\n"
        
        if send_email_report(report_text):
            st.success("✅ Az elemzés elkészült és elküldve az e-mail címedre!")
            st.balloons()
        else:
            st.warning("⚠️ Az elemzés kész, de az e-mail küldés sikertelen volt.")
    else:
        st.error("Nincs megfelelő meccs a megadott kritériumokkal.")
