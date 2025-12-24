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

def get_biro_statisztika(biro_nev):
    # Itt szimuláljuk a bírói szigorúságot, mivel az ingyenes API korlátozott
    # Egy valódi adatbázisból itt jönne a sárga lapok átlaga
    szigorusag = "Közepes"
    if biro_nev:
        return f"Bíró: {biro_nev} (Várható lapok: {szigorusag})"
    return "Bírói adatok nem elérhetőek."

def tipp_generalas(home_rank, away_rank, weather_temp):
    # Logikai döntéshozatal a fogadáshoz
    if home_rank < away_rank - 5:
        return "🔥 TIPP: Hazai győzelem (1) + Szögletek: Over 8.5"
    elif weather_temp < 5:
        return "❄️ TIPP: Kevés gól (Under 2.5) a hideg miatt + Lapok: Over 3.5"
    else:
        return "⚖️ TIPP: Dupla esély (1X) + Mindkét csapat szerez gólt (BTTS)"

def get_adatok():
    headers = {'X-Auth-Token': FOOTBALL_KEY} if FOOTBALL_KEY else {}
    riport = "🎯 PROFESSZIONÁLIS FOGADÁSI ELEMZÉS 🎯\n\n"
    
    try:
        # Időjárás lekérése
        w_res = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={WEATHER_KEY}&units=metric")
        temp = w_res.json()['main']['temp']
        riport += f"🌡️ Helyszíni hőmérséklet: {temp}°C\n"
    except: temp = 15

    try:
        f_res = requests.get("https://api.football-data.org/v4/matches", headers=headers)
        meccsek = f_res.json().get('matches', [])
        
        if meccsek:
            for m in meccsek[:3]:
                hazai = m['homeTeam']['name']
                vendeg = m['awayTeam']['name']
                biro = m.get('referees', [{}])[0].get('name', 'Ismeretlen')
                
                riport += f"\n⚽ MÉRKŐZÉS: {hazai} - {vendeg}\n"
                riport += f"👨‍⚖️ {get_biro_statisztika(biro)}\n"
                # Példa rangsor (mivel az API-ból a tabella külön kérés)
                riport += f"📝 ELEMZÉS: {tipp_generalas(1, 10, temp)}\n"
                riport += "--------------------------------------\n"
        else:
            riport += "\nMa nincs kiemelt elemzésre váró mérkőzés.\n"
    except:
        riport += "\nHiba az adatok lekérésekor.\n"
        
    return riport

def ultimate_football_bot():
    tartalom = get_adatok()
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = "🎯 Napi Fix Tippek és Bírói Elemzés"
    msg.attach(MIMEText(tartalom, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except: return False
