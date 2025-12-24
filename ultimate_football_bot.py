import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- BEÁLLÍTÁSOK ---
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "agbuyzyegfaokhhu")
FOOTBALL_KEY = os.environ.get("FOOTBALL_DATA_KEY")
NEWS_KEY = "7d577a4d9f2b4ba38541cc3f7e5ad6f5"
SAJAT_EMAIL = "czunidaniel9@gmail.com"

def get_adatok():
    headers = {'X-Auth-Token': FOOTBALL_KEY} if FOOTBALL_KEY else {}
    try:
        res = requests.get("https://api.football-data.org/v4/matches", headers=headers)
        meccsek = res.json().get('matches', [])
    except: return "Hiba az API-val."

    riport = "💰 NAPI DUPLÁZÓ STRATÉGIA (CÉL: 2.00 ODDS) 💰\n\n"
    
    if len(meccsek) >= 2:
        # KÉT MECCSES STRATÉGIA
        m1, m2 = meccsek[0], meccsek[1]
        riport += "✌️ KÉT MECCSES KOMBINÁCIÓ:\n"
        riport += f"1. {m1['homeTeam']['name']} - {m1['awayTeam']['name']} -> TIPP: Hazai vagy Döntetlen (1X)\n"
        riport += f"2. {m2['homeTeam']['name']} - {m2['awayTeam']['name']} -> TIPP: Over 1.5 gól\n"
        riport += "📊 VÁRHATÓ ÖSSZ-ODDS: ~2.05\n"
    elif len(meccsek) == 1:
        # EGY MECCSES KOMBINÁLT (BET BUILDER)
        m = meccsek[0]
        riport += "☝️ EGY MECCSES KOMBINÁLT TIPP (Bet Builder):\n"
        riport += f"Mérkőzés: {m['homeTeam']['name']} - {m['awayTeam']['name']}\n"
        riport += "🎯 TIPP: Hazai győzelem + Mindkét csapat szerez gólt (BTTS) + Over 3.5 sárga lap\n"
        riport += "📊 VÁRHATÓ ODDS: ~2.15\n"
    else:
        riport += "Ma nincs elég adat a 2.00-ás tipphez."

    return riport

def ultimate_football_bot():
    tartalom = get_adatok()
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = "🚀 Napi Duplázó: 1000 Ft -> 2000 Ft"
    msg.attach(MIMEText(tartalom, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except: return False
