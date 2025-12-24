import os
import requests
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict

# Logging a hibák követéséhez
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballAnalyzer:
    def __init__(self):
        self.football_key = os.environ.get("FOOTBALL_DATA_KEY")
        self.gmail_pw = os.environ.get("GMAIL_APP_PASSWORD")
        self.my_email = os.environ.get("SAJAT_EMAIL", "czunidaniel9@gmail.com")
        self.base_url = "https://api.football-data.org/v4"

    def get_matches(self) -> List[Dict]:
        if not self.football_key:
            logger.error("Hiányzó API kulcs!")
            return []
        
        try:
            headers = {'X-Auth-Token': self.football_key}
            response = requests.get(f"{self.base_url}/matches", headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get('matches', [])
        except Exception as e:
            logger.error(f"Hiba a lekérésnél: {e}")
            return []

    def score_match(self, match: Dict) -> float:
        """
        Itt jön a valódi matek: pontozzuk a meccset.
        Minél magasabb a pontszám, annál valószínűbb a 2.00-ás odds sikere.
        """
        score = 0.0
        # 1. Liga erőssége (PL, BL, La Liga előnyben)
        top_leagues = ['PL', 'CL', 'PD', 'SA', 'BL1']
        if match.get('competition', {}).get('code') in top_leagues:
            score += 5.0
        
        # 2. Hazai pálya előnye
        score += 2.0
        
        # Ide jöhetne a Head-to-Head (H2H) API lekérés is...
        return score

    def generate_pro_report(self) -> str:
        matches = self.get_matches()
        if not matches: return "Ma nincs elemzésre alkalmas mérkőzés."

        # Meccsek pontozása és sorbarendezése
        scored_matches = []
        for m in matches:
            scored_matches.append({
                'match': m,
                'score': self.score_match(m)
            })
        
        scored_matches.sort(key=lambda x: x['score'], reverse=True)
        top_2 = scored_matches[:2]

        report = "🚀 PROFESSZIONÁLIS DUPLÁZÓ STRATÉGIA 🚀\n\n"
        for i, item in enumerate(top_2, 1):
            m = item['match']
            report += f"{i}. {m['homeTeam']['name']} - {m['awayTeam']['name']}\n"
            report += f"   🏆 Bajnokság: {m['competition']['name']}\n"
            report += f"   📊 Bizalmi index: {item['score']}/10\n"
            report += f"   🎯 Javasolt piac: Hazai vagy Döntetlen + Over 1.5 gól\n\n"
        
        report += "⚠️ FIGYELEM: A statisztika valószínűséget mutat, nem garanciát."
        return report

def run_analysis_and_send():
    analyzer = FootballAnalyzer()
    report = analyzer.generate_pro_report()
    
    # Email küldés logikája
    msg = MIMEMultipart()
    msg['Subject'] = "🔥 Napi 2.00 Odds Elemzés"
    msg['From'] = analyzer.my_email
    msg['To'] = analyzer.my_email
    msg.attach(MIMEText(report, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(analyzer.my_email, analyzer.gmail_pw)
            server.send_message(msg)
        return True, "Email elküldve!"
    except Exception as e:
        return False, str(e)
