import os
import requests
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartBetBot:
    def __init__(self):
        self.football_key = os.environ.get("FOOTBALL_DATA_KEY")
        self.odds_key = "cc1a32d7a1d30cb4898eb879ff6d636f" # Az új kulcsod
        self.gmail_pw = os.environ.get("GMAIL_APP_PASSWORD")
        self.email = os.environ.get("SAJAT_EMAIL", "czunidaniel9@gmail.com")

    def get_real_odds(self, league="upcoming"):
        """Lekéri a valós oddsokat az Odds-API-ról"""
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{league}/odds"
            params = {
                'apiKey': self.odds_key,
                'regions': 'eu',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logger.error(f"Odds API hiba: {e}")
            return []

    def find_2_00_strategy(self):
        """Kiválasztja a legjobb meccset, ahol az odds 2.00 körüli"""
        all_odds = self.get_real_odds()
        if not all_odds:
            return "❌ Jelenleg nem érhető el élő odds adat."

        report = "📊 VALÓS ODDS ELEMZÉS (Cél: 2.00x szorzó)\n\n"
        found_picks = 0

        for match in all_odds:
            home = match['home_team']
            away = match['away_team']
            # Megkeressük a legjobb odds-ot a bukik közül (pl. Bet365)
            bookmaker = match['bookmakers'][0] if match['bookmakers'] else None
            if not bookmaker: continue

            odds = bookmaker['markets'][0]['outcomes']
            # Keressük azt a kimenetelt, ami 1.80 és 2.30 között van (ideális 1000->2000-hez)
            for outcome in odds:
                price = outcome['price']
                if 1.90 <= price <= 2.20:
                    report += f"⚽ MECCS: {home} - {away}\n"
                    report += f"🚩 TIPP: {outcome['name']}\n"
                    report += f"💰 VALÓS ODDS: {price} ({bookmaker['title']})\n"
                    report += "--------------------------------------\n"
                    found_picks += 1
                    break
            if found_picks >= 2: break # Csak a 2 legjobbat küldjük

        return report if found_picks > 0 else "Ma nincs biztonságos 2.00 körüli szóló tipp."

    def send_report(self):
        content = self.find_2_00_strategy()
        msg = MIMEMultipart()
        msg['Subject'] = "🎯 NAPI FIX: Valós Oddsok (2.00+)"
        msg['From'] = self.email
        msg['To'] = self.email
        msg.attach(MIMEText(content, 'plain'))

        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email, self.gmail_pw)
                server.send_message(msg)
            return True, "Elemzés elküldve!"
        except Exception as e:
            return False, f"Email hiba: {str(e)}"

def run():
    bot = SmartBetBot()
    return bot.send_report()
