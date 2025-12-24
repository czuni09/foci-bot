import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- BEÁLLÍTÁSOK ---
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "agbuyzyegfaokhhu")
FOOTBALL_KEY = os.environ.get("FOOTBALL_DATA_KEY")
WEATHER_KEY = "c31a011d35fed1b4d7b9f222c99d6dd2"
SAJAT_EMAIL = "czunidaniel9@gmail.com"

def elemzes_es_tipp(hazai, vendeg, temp, biro):
    # Kupa-faktor és meglepetés esélye
    tipp = f"📊 ELEMZÉS: {hazai} vs {vendeg}\n"
    tipp += f"👨‍⚖️ Bíró: {biro} -> Várható lapok: " + ("MAGAS (Over 4.5)" if "Oliver" in biro or "Taylor" in biro else "Normál (2-4)") + "\n"
    
    # Fogadási stratégia a listád alapján
    tipp += "💰 PONTOS TIPPEK:\n"
    tipp += "- FŐ TIPP: Dupla esély (1X) vagy Döntetlen (X) - a kupa-faktor miatt!\n"
    tipp += f"- GÓLOK: " + ("Under 2.5" if temp < 5 else "Over 2.5") + " (Időjárás: " + str(temp) + "°C)\n"
    tipp += "- SZÖGLETEK: Hazai csapat támadni fog -> Over 9.5 összesen\n"
    tipp += "- SPECIÁLIS: Mindkét csapat szerez gólt (BTTS): IGEN\n"
    return tipp

def get_adatok():
    headers = {'X-Auth-Token': FOOTBALL_KEY} if FOOTBALL_KEY else {}
    riport = "🎯 PROFI FOGADÁSI STRATÉGIA ÉS BÍRÓI JELENTÉS 🎯\n\n"
    
    try:
        w_res = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={WEATHER_KEY}&units=metric")
        temp = w_res.json()['main']['temp']
    except: temp = 10

    try:
        f_res = requests.get("https://api.football-data.org/v4/matches", headers=headers)
        data = f_res.json()
        meccsek = data.get('matches', [])
        
        if meccsek:
            for m in meccsek[:3]:
                h_nev = m['homeTeam']['name']
                v_nev = m['awayTeam']['name']
                biro_nev = m.get('referees', [{}])[0].get('name', 'Ismeretlen bíró')
                riport += elemzes_es_tipp(h_nev, v_nev, temp, biro_nev)
                riport += "\n" + "="*40 + "\n"
        else:
            riport += "Ma nincs elemzésre váró kiemelt kupa/bajnoki meccs.\n"
    except Exception as e:
        riport += f"Hiba az adatoknál: {e}\n"
        
    return riport

def ultimate_football_bot():
    tartalom = get_adatok()
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = "🔥 PONTOS TIPPEK: Meccs, Szöglet, Lapok"
    msg.attach(MIMEText(tartalom, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except: return False
