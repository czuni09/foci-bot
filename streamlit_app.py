import streamlit as st
import requests
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================================
# KONFIGURÁCIÓ
# ============================================
ODDS_API_KEY = "cc1a32d7a1d30cb4898eb879ff6d636f"
GMAIL_APP_PASSWORD = "whppzywzoduqjrgk"
SAJAT_EMAIL = "czunidaniel9@gmail.com"

class DoubleUpLogic:
    def __init__(self):
        self.api_key = ODDS_API_KEY

    def get_odds(self, sport='soccer_epl'):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {'apiKey': self.api_key, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
        try:
            res = requests.get(url, params=params, timeout=10)
            return res.json() if res.status_code == 200 else []
        except: return []

    def get_best_combo(self, leagues):
        all_candidates = []
        for league in leagues:
            matches = self.get_odds(league)
            for m in matches:
                home = m['home_team']
                away = m['away_team']
                
                # Bukmékerek átfésülése a legjobb oddsért
                for bookie in m.get('bookmakers', []):
                    outcomes = bookie['markets'][0]['outcomes']
                    for o in outcomes:
                        price = o['price']
                        # SÚLYOZÁS: Olyan "esélyes" tippeket keresünk, amik 1.35 és 1.60 között vannak
                        if 1.35 <= price <= 1.60:
                            all_candidates.append({
                                'match': f"{home} - {away}",
                                'pick': o['name'],
                                'odds': price,
                                'score': 1/price # Minél kisebb az odds, annál nagyobb a súly (biztonság)
                            })
        
        # Súlyozás szerinti rendezés (a legesélyesebbek előre)
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if len(all_candidates) >= 2:
            pick1 = all_candidates[0]
            pick2 = all_candidates[1]
            total_odds = pick1['odds'] * pick2['odds']
            return [pick1, pick2], total_odds
        return None, 0

def send_mail(text):
    msg = MIMEMultipart()
    msg['Subject'] = f"🎯 Napi Duplázó Szelvény - {datetime.now().strftime('%m.%d')}"
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg.attach(MIMEText(text, 'plain', 'utf-8'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
            s.send_message(msg)
        return True
    except: return False

# ============================================
# UI
# ============================================
st.set_page_config(page_title="Duplázó Bot", page_icon="⚽")
st.title("🏆 Napi Biztos Duplázó (2x)")

leagues = st.multiselect("Válassz ligákat:", 
    ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga', 'soccer_italy_serie_a'],
    default=['soccer_epl', 'soccer_spain_la_liga'])

if st.button("🚀 LEGJOBB KOMBI GENERÁLÁSA", use_container_width=True):
    logic = DoubleUpLogic()
    with st.spinner("Elemzés..."):
        combo, total = logic.get_best_combo(leagues)

