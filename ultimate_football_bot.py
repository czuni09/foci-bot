import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass

# --- ÖSSZEGYŰJTÖTT KULCSOK ---
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "agbuyzyegfaokhhu")
FOOTBALL_KEY = os.environ.get("FOOTBALL_DATA_KEY") 
WEATHER_KEY = "c31a011d35fed1b4d7b9f222c99d6dd2"
NEWS_KEY = "7d577a4d9f2b4ba38541cc3f7e5ad6f5"
SAJAT_EMAIL = "czunidaniel9@gmail.com"

@dataclass
class TeamStats:
    name: str

def get_mai_adatok():
    headers = {'X-Auth-Token': FOOTBALL_KEY} if FOOTBALL_KEY else {}
    res = requests.get("https://api.football-data.org/v4/matches", headers=headers)
    meccsek = res.json().get('matches', [])
    
    if not meccsek:
        return "Ma nincs rögzített mérkőzés a rendszerben."
    
    riport = "⚽ MAI ÉLES FOCI TIPPEK:\n\n"
    for m in meccsek[:5]:
        riport += f"🏆 {m['competition']['name']}: {m['homeTeam']['name']} - {m['awayTeam']['name']}\n"
    return riport

def ultimate_football_bot(home=None, away=None, varos=None, bajnoksag=None, odds=None, alap_esely=None):
    # Ez a függvény kezeli az e-mail küldést
    tartalom = get_mai_adatok()
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = "Éles Foci Jelentés"
    msg.attach(MIMEText(tartalom, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except:
        return False
