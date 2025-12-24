import os
import requests
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartBetBot:
    """Intelligens fogadási elemző bot valódi odds adatokkal"""
    
    # Valid sport azonosítók
    VALID_SPORTS = [
        'soccer_epl',  # Premier League
        'soccer_spain_la_liga',  # La Liga
        'soccer_germany_bundesliga',  # Bundesliga
        'soccer_italy_serie_a',  # Serie A
        'soccer_uefa_champs_league',  # Champions League
    ]
    
    def __init__(self):
        # API kulcsok environment változókból
        self.football_key = os.environ.get("FOOTBALL_DATA_KEY")
        self.odds_key = os.environ.get("ODDS_API_KEY")
        self.gmail_pw = os.environ.get("GMAIL_APP_PASSWORD")
        self.email = os.environ.get("SAJAT_EMAIL")
        
        # Validálás
        self._validate_config()
    
    def _validate_config(self):
        """Ellenőrzi, hogy minden szükséges konfiguráció be van-e állítva"""
        missing = []
        
        if not self.odds_key:
            missing.append("ODDS_API_KEY")
        if not self.gmail_pw:
            missing.append("GMAIL_APP_PASSWORD")
        if not self.email:
            missing.append("SAJAT_EMAIL")
        
        if missing:
            raise ValueError(
                f"Hiányzó környezeti változók: {', '.join(missing)}\n"
                "Állítsd be őket a Streamlit Secrets-ben vagy .env fájlban!"
            )
    
    def get_real_odds(self, sport: str = 'soccer_epl') -> List[Dict]:
        """
        Lekéri a valós oddsokat az Odds-API-ról
        
        Args:
            sport: Sport azonosító (pl. 'soccer_epl')
        
        Returns:
            Lista a meccsekről és odds-okról
        """
        if sport not in self.VALID_SPORTS:
            logger.warning(f"Ismeretlen sport: {sport}, használom az EPL-t")
            sport = 'soccer_epl'
        
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                'apiKey': self.odds_key,
                'regions': 'eu',
                'markets': 'h2h',  # Head to head (1X2)
                'oddsFormat': 'decimal'
            }
            
            logger.info(f"Odds lekérés: {sport}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"{len(data)} meccs lekérve")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error("Timeout: Az Odds API nem válaszolt időben")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP hiba: {e.response.status_code} - {e.response.text}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request hiba: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Váratlan hiba: {str(e)}")
            return []
    
    def _is_match_soon(self, commence_time_str: str, hours: int = 24) -> bool:
        """Ellenőrzi, hogy a meccs a következő X órában van-e"""
        try:
            match_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
            now = datetime.now(match_time.tzinfo)
            return now <= match_time <= now + timedelta(hours=hours)
        except Exception as e:
            logger.warning(f"Időpont parse hiba: {e}")
            return True  # Ha nem tudjuk, akkor is megjelenítjük
    
    def find_value_bets(self, target_odds: float = 2.0, tolerance: float = 0.2) -> str:
        """
        Megkeresi az értékes fogadási lehetőségeket
        
        Args:
            target_odds: Célzott odds (pl. 2.0)
            tolerance: Tolerancia (pl. 0.2 = 1.8-2.2 közötti odds-ok)
        
        Returns:
            Formázott riport string
        """
        min_odds = target_odds - tolerance
        max_odds = target_odds + tolerance
        
        report = f"📊 VALÓS ODDS ELEMZÉS\n"
        report += f"🎯 Cél odds: {target_odds:.2f} (±{tolerance})\n"
        report += f"📅 Dátum: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report += "=" * 50 + "\n\n"
        
        found_picks = []
        
        # Több ligában is keresünk
        for sport in self.VALID_SPORTS[:3]:  # Csak az első 3 ligát nézzük
            all_odds = self.get_real_odds(sport)
            
            if not all_odds:
                continue
            
            for match in all_odds:
                # Csak a közeli meccseket nézzük
                if not self._is_match_soon(match.get('commence_time', '')):
                    continue
                
                home = match.get('home_team', 'Ismeretlen')
                away = match.get('away_team', 'Ismeretlen')
                commence_time = match.get('commence_time', '')
                
                bookmakers = match.get('bookmakers', [])
                if not bookmakers:
                    logger.debug(f"Nincs bukméker adat: {home} - {away}")
                    continue
                
                # Legjobb odds keresése az összes bukméker között
                for bookmaker in bookmakers:
                    markets = bookmaker.get('markets', [])
                    if not markets:
                        continue
                    
                    outcomes = markets[0].get('outcomes', [])
                    
                    for outcome in outcomes:
                        price = outcome.get('price', 0)
                        name = outcome.get('name', '')
                        
                        # Ha a cél tartományban van
                        if min_odds <= price <= max_odds:
                            found_picks.append({
                                'home': home,
                                'away': away,
                                'pick': name,
                                'odds': price,
                                'bookmaker': bookmaker.get('title', 'Ismeretlen'),
                                'commence_time': commence_time,
                                'sport': sport
                            })
        
        # Rendezés odds szerint (legközelebbi a célhoz)
        found_picks.sort(key=lambda x: abs(x['odds'] - target_odds))
        
        if not found_picks:
            report += "❌ Jelenleg nincs megfelelő odds a célhoz közeli tartományban.\n"
            report += "💡 Próbáld később, vagy állíts be nagyobb toleranciát.\n"
            return report
        
        # Top 5 legjobb pick
        report += f"✅ {len(found_picks)} találat! Itt a legjobb 5:\n\n"
        
        for i, pick in enumerate(found_picks[:5], 1):
            try:
                match_time = datetime.fromisoformat(pick['commence_time'].replace('Z', '+00:00'))
                time_str = match_time.strftime('%Y-%m-%d %H:%M')
            except:
                time_str = "Ismeretlen időpont"
            
            report += f"{i}. ⚽ {pick['home']} - {pick['away']}\n"
            report += f"   🏆 Liga: {pick['sport'].replace('soccer_', '').replace('_', ' ').title()}\n"
            report += f"   🎯 Tipp: {pick['pick']}\n"
            report += f"   💰 Odds: {pick['odds']:.2f}\n"
            report += f"   📍 Bukméker: {pick['bookmaker']}\n"
            report += f"   ⏰ Kezdés: {time_str}\n"
            report += "-" * 50 + "\n\n"
        
        report += "\n⚠️  FIGYELMEZTETÉS:\n"
        report += "• Ez NEM fogadási tanács, csak adatmegjelenítés!\n"
        report += "• Kizárólag saját felelősségre fogadj!\n"
        report += "• A múltbeli eredmények nem garantálják a jövőbeli sikert!\n"
        
        return report
    
    def send_report(self, target_odds: float = 2.0) -> tuple[bool, str]:
        """
        Email riport küldése
        
        Returns:
            (success: bool, message: str)
        """
        try:
            content = self.find_value_bets(target_odds=target_odds)
            
            msg = MIMEMultipart()
            msg['Subject'] = f"🎯 Napi Odds Elemzés ({target_odds:.2f}x cél)"
            msg['From'] = self.email
            msg['To'] = self.email
            msg.attach(MIMEText(content, 'plain', 'utf-8'))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email, self.gmail_pw)
                server.send_message(msg)
            
            logger.info("Email sikeresen elküldve")
            return True, "✅ Elemzés elküldve email-ben!"
            
        except smtplib.SMTPAuthenticationError:
            error_msg = "❌ Email hitelesítési hiba! Ellenőrizd a jelszót."
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"❌ Email küldési hiba: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_report_text(self, target_odds: float = 2.0) -> str:
        """Csak a riport szövegét adja vissza, email küldés nélkül"""
        return self.find_value_bets(target_odds=target_odds)


def run(send_email: bool = True, target_odds: float = 2.0):
    """
    Fő futtatás
    
    Args:
        send_email: Email küldés engedélyezése
        target_odds: Célzott odds
    """
    try:
        bot = SmartBetBot()
        
        if send_email:
            return bot.send_report(target_odds=target_odds)
        else:
            report = bot.get_report_text(target_odds=target_odds)
            print(report)
            return True, report
            
    except ValueError as e:
        error_msg = f"Konfiguráció hiba: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Váratlan hiba: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


if __name__ == "__main__":
    # Teszt futtatás email küldés nélkül
    success, message = run(send_email=False, target_odds=2.0)
    if not success:
        print(f"HIBA: {message}")
