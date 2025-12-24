import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- TITKOK BEOLVASÁSA ---
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "agbuyzyegfaokhhu")
FOOTBALL_KEY = os.environ.get("FOOTBALL_DATA_KEY") # Ezt add hozzá a Secrets-hez!
WEATHER_KEY = "c31a011d35fed1b4d7b9f222c99d6dd2"
SAJAT_EMAIL = "czunidaniel9@gmail.com"

def get_mai_meccsek():
    if not FOOTBALL_KEY:
        return "Nincs Football API kulcs beállítva."
    
    url = "https://api.football-data.org/v4/matches"
    headers = {'X-Auth-Token': FOOTBALL_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        meccsek = data.get('matches', [])
        
        if not meccsek:
            return "Ma nincs kiemelt mérkőzés a rendszerben."
        
        riport = "⚽ MAI KIEMELT MECCSEK ÉS TIPPEK:\n\n"
        for m in meccsek[:5]: # Az első 5 meccs
            hazai = m['homeTeam']['name']
            vendeg = m['awayTeam']['name']
            bajnoksag = m['competition']['name']
            riport += f"🏆 {bajnoksag}: {hazai} vs {vendeg}\n"
        return riport
    except:
        return "Hiba történt az adatok lekérésekor."

def kuldj_jelentes(tartalom):
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = "Napi Foci Jelentés - Éles Adatok"
    msg.attach(MIMEText(tartalom, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Hiba: {e}")
        return False

if __name__ == "__main__":
    mai_adatok = get_mai_meccsek()
    kuldj_jelentes(mai_adatok)
