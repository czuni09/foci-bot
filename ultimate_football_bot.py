import datetime
import requests
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from typing import List, Tuple

# API KULCSOK - A GitHub Secrets-ből jönnek
WEATHER_KEY = os.environ.get("WEATHER_KEY", "c31a011d35fed1b4d7b9f222c99d6dd2")
NEWS_KEY = os.environ.get("NEWS_KEY", "7d577a4d9f2b4ba38541cc3f7e5ad6f5")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
SAJAT_EMAIL = "czunidaniel9@gmail.com"

@dataclass
class TeamStats:
    name: str
    injuries: List[str] = field(default_factory=list)
    intl_absences: List[str] = field(default_factory=list)

def kuldj_emailt(targy, tartalom):
    if not GMAIL_APP_PASSWORD:
        print("HIBA: Hiányzik a GMAIL_APP_PASSWORD!")
        return
    msg = MIMEMultipart()
    msg['From'] = SAJAT_EMAIL
    msg['To'] = SAJAT_EMAIL
    msg['Subject'] = targy
    msg.attach(MIMEText(tartalom, 'plain', 'utf-8'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SAJAT_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("E-mail elküldve!")
    except Exception as e:
        print(f"Hiba: {e}")

def ultimate_football_bot(home, away, varos, bajnoksag, odds, alap_esely):
    esely = float(alap_esely)
    # Egyszerűsített számítás a teszthez
    if home.injuries or home.intl_absences: esely -= 5
    
    szoveg = f"⚽ ELEMZÉS: {home.name} - {away.name}\n"
    szoveg += f"Esély: {esely}%\nJavaslat: "
    szoveg += "FOGADÁS" if esely >= 75 else "KERÜLD"
    
    kuldj_emailt(f"Foci Tipp: {home.name} ({esely}%)", szoveg)

if __name__ == "__main__":
    h = TeamStats("Arsenal", intl_absences=["Partey"])
    v = TeamStats("Crystal Palace", intl_absences=["Ayew"])
    ultimate_football_bot(h, v, "London", "Premier League", 1.48, 80)
