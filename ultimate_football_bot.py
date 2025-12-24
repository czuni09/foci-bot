import os
import requests
import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- BIZTONSÁGOS BEÁLLÍTÁSOK ---
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
FOOTBALL_KEY = os.environ.get("FOOTBALL_DATA_KEY")
NEWS_KEY = os.environ.get("NEWS_DATA_KEY", "7d577a4d9f2b4ba38541cc3f7e5ad6f5")
SAJAT_EMAIL = os.environ.get("SAJAT_EMAIL", "czunidaniel9@gmail.com")

def get_adatok():
    headers = {'X-Auth-Token': FOOTBALL_KEY}
    riport = "🚀 NAPI DUPLÁZÓ ELEMZÉS 🚀\n\n"
    
    try:
        # Foci adatok lekérése
        res = requests.get("https://api.football-data.org/v4/matches", headers=headers, timeout=10)
        res.raise_for_status() # Hibát dob, ha pl. 403 (rossz kulcs) vagy 404
        meccsek = res.json().get('matches', [])
        
        if not meccsek:
            return "Ma nincs kiemelt mérkőzés a figyelt ligákban."

        for m in meccsek[:2]:
            hazai = m['homeTeam']['name']
            vendeg = m['awayTeam']['name']
            biro = m.get('referees', [{}])[0].get('name', 'Ismeretlen')
            
            # Hírek lekérése hibakezeléssel
            try:
                n_res = requests.get(f"https://newsapi.org/v2/everything?q={hazai}+scandal+injury&apiKey={NEWS_KEY}", timeout=5)
                hirek_data = n_res.json()
                hirek = " | ".join([a['title'] for a in hirek_data.get('articles', [])[:2]])
            except Exception as e:
                hirek = f"Hír-szolgáltatás hiba: {str(e)}"
            
            riport += f"⚽ {hazai} - {vendeg}\n"
            riport += f"👨‍⚖️ Bíró: {biro}\n"
            riport += f"🗞️ Infó: {hirek}\n"
            riport += f"🎯 TIPP: {hazai} v X + Over 1.5 gól\n"
            riport += "--------------------------------------\n"
            
        return riport
    except requests.exceptions.RequestException as e:
        return f"Hálózati hiba az API-val: {str(e)}"
    except Exception as e:
        return f"Váratlan hiba az adatoknál: {str(e)}"

def ultimate_football_bot():
    if not GMAIL_APP_PASSWORD:
        print("HIBA: Nincs Gmail alkalmazásjelszó beállítva!")
        return False, "Hiányzó Gmail jelszó."
        
    tartalom = get_adatok()
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = "🔥 Napi Duplázó Szelvény"
    msg.attach(MIMEText(tartalom, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
        server.starttls()
        server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True, "Sikeres küldés!"
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail belépési hiba: Rossz alkalmazásjelszó!"
    except Exception as e:
        # Itt kiírjuk a teljes hiba-útvonalat a konzolra (debughoz)
        print(traceback.format_exc())
        return False, f"Email hiba: {str(e)}"
