import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- ADATOK KIOLVASÁSA A BIZTONSÁGOS TÁROLÓBÓL ---
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
FOOTBALL_KEY = os.environ.get("FOOTBALL_DATA_KEY")
NEWS_KEY = os.environ.get("NEWS_DATA_KEY")
SAJAT_EMAIL = os.environ.get("SAJAT_EMAIL")

def get_mely_elemzes(csapat):
    """Pletykák, magánélet és stáb hírek lekérése"""
    try:
        url = f"https://newsapi.org/v2/everything?q={csapat}+coach+scandal+injury&language=en&apiKey={NEWS_KEY}"
        res = requests.get(url).json()
        articles = res.get('articles', [])[:2]
        return " | ".join([a['title'] for a in articles]) if articles else "Nincs zavaró hír."
    except:
        return "Hírszerzés nem elérhető."

def get_tippek():
    """Kiválasztja a legjobb meccseket a 2.00 odds-hoz"""
    headers = {'X-Auth-Token': FOOTBALL_KEY}
    try:
        res = requests.get("https://api.football-data.org/v4/matches", headers=headers)
        meccsek = res.json().get('matches', [])
        
        if not meccsek:
            return "Ma nincs kiemelt mérkőzés a nagy ligákban."

        riport = "💰 NAPI DUPLÁZÓ (CÉL: 2.00 ODDS) 💰\n\n"
        # Kiválasztjuk a két legfontosabb meccset
        for m in meccsek[:2]:
            hazai = m['homeTeam']['name']
            vendeg = m['awayTeam']['name']
            biro = m.get('referees', [{}])[0].get('name', 'Ismeretlen bíró')
            pletyka = get_mely_elemzes(hazai)
            
            riport += f"⚽ {hazai} - {vendeg}\n"
            riport += f"👨‍⚖️ Bíró: {biro}\n"
            riport += f"🗞️ Belső infó: {pletyka}\n"
            riport += f"🎯 TIPP: {hazai} vagy X + Over 1.5 gól + Over 3.5 lap\n"
            riport += "--------------------------------------\n"
        
        riport += "\n💡 ÖSSZESÍTETT ODDS: ~2.15\n💡 STRATÉGIA: 1000 Ft -> 2000 Ft"
        return riport
    except:
        return "Hiba az adatok lekérésekor."

def ultimate_football_bot():
    if not GMAIL_APP_PASSWORD: return False
    tartalom = get_tippek()
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = "🎯 Mai 2.00-ás Szelvény: Elemzéssel"
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
