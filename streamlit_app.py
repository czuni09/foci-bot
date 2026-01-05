# 📁 app.py - FŐ ALKALMAZÁS
import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import sqlite3
import json
import feedparser
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from enum import Enum
import time
import os

# ==================== KONFIGURÁCIÓ ====================
@dataclass
class Config:
    """API konfigurációk"""
    
    # API kulcsok (secrets-ből jönnek)
    ODDS_API_KEY: str
    SPORTMONKS_API_KEY: str
    NEWS_API_KEY: str
    WEATHER_API_KEY: str
    
    # API végpontok
    SPORTMONKS_BASE_URL = "https://api.sportmonks.com/v3/football"
    ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
    NEWS_BASE_URL = "https://newsapi.org/v2"
    WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    @classmethod
    def load(cls):
        """Konfiguráció betöltése"""
        try:
            return cls(
                ODDS_API_KEY=st.secrets.get("ODDS_API_KEY", ""),
                SPORTMONKS_API_KEY=st.secrets.get("SPORTMONKS_API_KEY", ""),
                NEWS_API_KEY=st.secrets.get("NEWS_API_KEY", ""),
                WEATHER_API_KEY=st.secrets.get("WEATHER_API_KEY", "")
            )
        except:
            return cls(
                ODDS_API_KEY=os.getenv("ODDS_API_KEY", ""),
                SPORTMONKS_API_KEY=os.getenv("SPORTMONKS_API_KEY", ""),
                NEWS_API_KEY=os.getenv("NEWS_API_KEY", ""),
                WEATHER_API_KEY=os.getenv("WEATHER_API_KEY", "")
            )
    
    def validate(self):
        """API kulcsok validálása"""
        return all([
            self.ODDS_API_KEY,
            self.SPORTMONKS_API_KEY,
            self.NEWS_API_KEY,
            self.WEATHER_API_KEY
        ])

# Konfiguráció betöltése
CONFIG = Config.load()

# ==================== ADATBÁZIS ====================
class Database:
    """SQLite adatbázis kezelő"""
    
    def __init__(self, db_path="titan_bot.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Adatbázis inicializálása"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Mentett fogadások
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS picks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                home_team TEXT,
                away_team TEXT,
                league TEXT,
                pick_type TEXT,
                pick_value TEXT,
                odds REAL,
                confidence REAL,
                status TEXT,
                result TEXT,
                profit_loss REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(match_id, pick_type, pick_value)
            )
        ''')
        
        # Teljesítmény statisztika
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                date DATE PRIMARY KEY,
                total_picks INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pushes INTEGER DEFAULT 0,
                total_stake REAL DEFAULT 0,
                total_return REAL DEFAULT 0,
                roi REAL DEFAULT 0
            )
        ''')
        
        # API cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_pick(self, pick_data):
        """Fogadás mentése"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO picks 
                (match_id, home_team, away_team, league, pick_type, pick_value, odds, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pick_data.get('match_id'),
                pick_data.get('home_team'),
                pick_data.get('away_team'),
                pick_data.get('league'),
                pick_data.get('pick_type'),
                pick_data.get('pick_value'),
                pick_data.get('odds'),
                pick_data.get('confidence'),
                pick_data.get('status')
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Mentési hiba: {e}")
            return False
    
    def get_performance(self, days=30):
        """Teljesítmény statisztika"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN result = 'WIN' THEN odds ELSE NULL END) as avg_win_odds,
                SUM(profit_loss) as total_profit
            FROM picks 
            WHERE timestamp >= date('now', ?)
        ''', (f'-{days} days',))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            win_rate = (result[1] / result[0]) * 100 if result[0] > 0 else 0
            roi = (result[4] / (result[0] * 10)) * 100 if result[0] > 0 else 0
            
            return {
                'total': result[0],
                'wins': result[1],
                'losses': result[2],
                'win_rate': round(win_rate, 1),
                'avg_win_odds': round(result[3] or 0, 2),
                'total_profit': round(result[4] or 0, 2),
                'roi': round(roi, 1)
            }
        
        return {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'avg_win_odds': 0,
            'total_profit': 0,
            'roi': 0
        }

DB = Database()

# ==================== API KEZELŐ ====================
class RateLimiter:
    """Rate limiting"""
    
    def __init__(self, calls_per_minute=30):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def wait_if_needed(self):
        """Várás ha túl gyorsan hívunk"""
        now = time.time()
        minute_ago = now - 60
        
        # Régi hívások törlése
        self.calls = [call for call in self.calls if call > minute_ago]
        
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)

class SportMonksAPI:
    """Sportmonks API kezelő"""
    
    def __init__(self):
        self.base_url = CONFIG.SPORTMONKS_BASE_URL
        self.api_key = CONFIG.SPORTMONKS_API_KEY
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.limiter = RateLimiter(10)
    
    async def get_fixtures(self, date=None, league_ids=None):
        """Meccsek lekérése"""
        await self.limiter.wait_if_needed()
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/fixtures/date/{date}"
        params = {
            "include": "participants;league;referee",
            "per_page": 50
        }
        
        if league_ids:
            params["leagues"] = ",".join(map(str, league_ids))
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._process_fixtures(data.get("data", []))
                    else:
                        st.error(f"Sportmonks API error: {response.status}")
                        return []
                        
        except Exception as e:
            st.error(f"Sportmonks API hiba: {e}")
            return []
    
    def _process_fixtures(self, fixtures):
        """Meccs adatok feldolgozása"""
        processed = []
        
        for fixture in fixtures:
            participants = fixture.get("participants", [])
            home_team = participants[0] if len(participants) > 0 else {}
            away_team = participants[1] if len(participants) > 1 else {}
            league = fixture.get("league", {})
            referee = fixture.get("referee", {})
            
            # Alap statisztikák (valós API-ból jönne, most dummy)
            stats = {
                "home_form": ["W", "D", "L", "W", "W"],
                "away_form": ["L", "W", "D", "L", "W"],
                "home_xg": 1.8,
                "away_xg": 1.4,
                "home_goals_avg": 2.1,
                "away_goals_avg": 1.3,
                "home_conceded_avg": 1.2,
                "away_conceded_avg": 1.8
            }
            
            processed.append({
                "id": fixture.get("id"),
                "date": fixture.get("starting_at"),
                "timestamp": fixture.get("starting_at_timestamp"),
                "home_team": home_team.get("name", "Unknown"),
                "away_team": away_team.get("name", "Unknown"),
                "home_id": home_team.get("id"),
                "away_id": away_team.get("id"),
                "league_id": league.get("id"),
                "league_name": league.get("name", "Unknown"),
                "league_country": league.get("country", {}).get("name", "Unknown"),
                "referee_id": referee.get("id"),
                "referee": referee.get("common_name") or referee.get("name", "Unknown"),
                "venue": fixture.get("venue", {}).get("name", "Unknown"),
                "city": fixture.get("venue", {}).get("city", "Unknown"),
                "status": fixture.get("status", {}).get("description", "Scheduled"),
                **stats
            })
        
        return processed
    
    async def get_referee_stats(self, referee_id):
        """Bíró statisztikák"""
        await self.limiter.wait_if_needed()
        
        url = f"{self.base_url}/referees/{referee_id}"
        params = {"include": "stats.details"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    params=params
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._process_referee_stats(data.get("data", {}))
                    else:
                        return self._get_fallback_referee_stats()
                        
        except Exception as e:
            st.warning(f"Bíró statisztika hiba: {e}")
            return self._get_fallback_referee_stats()
    
    def _process_referee_stats(self, referee_data):
        """Bíró statisztikák feldolgozása"""
        if not referee_data:
            return self._get_fallback_referee_stats()
        
        # Valós adatok feldolgozása
        stats = referee_data.get("stats", [])
        card_stats = {}
        
        for stat in stats:
            if stat.get("type") == "cards":
                details = stat.get("details", [])
                for detail in details:
                    card_stats[detail.get("type")] = detail.get("value")
        
        yellow_avg = card_stats.get("yellow_cards_avg", 3.8)
        red_avg = card_stats.get("red_cards_avg", 0.15)
        fouls_avg = card_stats.get("fouls_avg", 21.0)
        
        # Szigorúság számítás
        strictness_score = (yellow_avg * 1) + (red_avg * 3)
        if strictness_score > 6:
            strictness = "Very High"
        elif strictness_score > 4.5:
            strictness = "High"
        elif strictness_score > 3.5:
            strictness = "Medium"
        else:
            strictness = "Low"
        
        return {
            "yellow_avg": yellow_avg,
            "red_avg": red_avg,
            "fouls_avg": fouls_avg,
            "strictness": strictness,
            "total_matches": referee_data.get("total_matches", 100),
            "country": referee_data.get("country", {}).get("name", "Unknown")
        }
    
    def _get_fallback_referee_stats(self):
        """Fallback bírói adatok"""
        return {
            "yellow_avg": 3.8,
            "red_avg": 0.15,
            "fouls_avg": 21.0,
            "strictness": "Medium",
            "total_matches": 100,
            "country": "Unknown"
        }
    
    async def get_leagues(self):
        """Ligák lekérése"""
        await self.limiter.wait_if_needed()
        
        url = f"{self.base_url}/leagues"
        params = {
            "include": "country",
            "per_page": 100
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    params=params
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        leagues = data.get("data", [])
                        
                        # Csak aktív, népszerű ligák
                        popular_leagues = []
                        for league in leagues:
                            if league.get("active") and league.get("is_cup") is False:
                                country = league.get("country", {})
                                popular_leagues.append({
                                    "id": league.get("id"),
                                    "name": league.get("name"),
                                    "country": country.get("name", "Unknown"),
                                    "logo": league.get("logo_path")
                                })
                        
                        return popular_leagues[:20]  # Első 20 legjobb
                    else:
                        return self._get_fallback_leagues()
                        
        except Exception as e:
            st.warning(f"Ligák lekérés hiba: {e}")
            return self._get_fallback_leagues()
    
    def _get_fallback_leagues(self):
        """Fallback ligák"""
        return [
            {"id": 8, "name": "Premier League", "country": "England"},
            {"id": 564, "name": "La Liga", "country": "Spain"},
            {"id": 82, "name": "Bundesliga", "country": "Germany"},
            {"id": 384, "name": "Serie A", "country": "Italy"},
            {"id": 301, "name": "Ligue 1", "country": "France"},
            {"id": 72, "name": "Eredivisie", "country": "Netherlands"},
            {"id": 94, "name": "Primeira Liga", "country": "Portugal"}
        ]

class OddsAPI:
    """Odds API kezelő"""
    
    def __init__(self):
        self.base_url = CONFIG.ODDS_BASE_URL
        self.api_key = CONFIG.ODDS_API_KEY
    
    async def get_odds(self, sport="soccer_epl"):
        """Odds-ok lekérése"""
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu,us",
            "oddsFormat": "decimal",
            "bookmakers": "bet365,betfair,pinnacle"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        return []
                        
        except Exception as e:
            st.warning(f"Odds API hiba: {e}")
            return []
    
    def process_match_odds(self, match_data, odds_list):
        """Odds-ok hozzárendelése meccshez"""
        if not odds_list:
            return self._generate_default_odds(match_data)
        
        home_team = match_data.get("home_team", "").lower()
        away_team = match_data.get("away_team", "").lower()
        
        for odds in odds_list:
            odds_home = odds.get("home_team", "").lower()
            odds_away = odds.get("away_team", "").lower()
            
            if (home_team in odds_home or odds_home in home_team) and \
               (away_team in odds_away or odds_away in away_team):
                
                bookmakers = odds.get("bookmakers", [])
                if bookmakers:
                    # Legjobb odds-ok kiválasztása
                    best_odds = self._get_best_odds(bookmakers)
                    return best_odds
        
        return self._generate_default_odds(match_data)
    
    def _get_best_odds(self, bookmakers):
        """Legjobb odds-ok keresése"""
        best = {
            "home_win": 2.0,
            "draw": 3.4,
            "away_win": 3.8,
            "over_2_5": 1.9,
            "under_2_5": 1.9,
            "btts_yes": 1.8,
            "btts_no": 1.95,
            "cards_over_4_5": 2.1,
            "cards_under_4_5": 1.7
        }
        
        for bookmaker in bookmakers:
            markets = bookmaker.get("markets", [])
            for market in markets:
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Home":
                            best["home_win"] = max(best["home_win"], outcome.get("price", 2.0))
                        elif outcome["name"] == "Away":
                            best["away_win"] = max(best["away_win"], outcome.get("price", 3.8))
                        elif outcome["name"] == "Draw":
                            best["draw"] = max(best["draw"], outcome.get("price", 3.4))
                
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over" and market.get("point") == 2.5:
                            best["over_2_5"] = max(best["over_2_5"], outcome.get("price", 1.9))
                        elif outcome["name"] == "Under" and market.get("point") == 2.5:
                            best["under_2_5"] = max(best["under_2_5"], outcome.get("price", 1.9))
        
        return best
    
    def _generate_default_odds(self, match_data):
        """Alapértelmezett odds-ok"""
        return {
            "home_win": 2.0,
            "draw": 3.4,
            "away_win": 3.8,
            "over_2_5": 1.9,
            "under_2_5": 1.9,
            "btts_yes": 1.8,
            "btts_no": 1.95,
            "cards_over_4_5": 2.1,
            "cards_under_4_5": 1.7
        }

class NewsAnalyzer:
    """Hírek elemzése"""
    
    def __init__(self):
        self.api_key = CONFIG.NEWS_API_KEY
    
    async def get_team_news(self, team_name, days=3):
        """Csapat hírek"""
        if not self.api_key:
            return self._get_fallback_news(team_name)
        
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = f"{CONFIG.NEWS_BASE_URL}/everything"
        params = {
            "apiKey": self.api_key,
            "q": f"{team_name} football",
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 10
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._analyze_sentiment(data.get("articles", []))
                    else:
                        return self._get_fallback_news(team_name)
                        
        except Exception as e:
            return self._get_fallback_news(team_name)
    
    def _analyze_sentiment(self, articles):
        """Hangulatelemzés"""
        if not articles:
            return {"score": 0, "reasons": [], "count": 0}
        
        positive_keywords = ["win", "victory", "sign", "return", "fit", "recover", "motivated"]
        negative_keywords = ["injury", "suspended", "doubt", "crisis", "conflict", "miss", "out"]
        
        score = 0
        reasons = []
        
        for article in articles[:5]:  # Csak első 5 cikk
            title = article.get("title", "").lower()
            content = article.get("content", "").lower() if article.get("content") else title
            
            # Pozitív szavak
            pos_count = sum(1 for word in positive_keywords if word in content)
            if pos_count > 0:
                score += pos_count
                reasons.append(f"🟢 Pozitív hír: {title[:50]}...")
            
            # Negatív szavak
            neg_count = sum(1 for word in negative_keywords if word in content)
            if neg_count > 0:
                score -= neg_count
                reasons.append(f"🔴 Negatív hír: {title[:50]}...")
        
        return {
            "score": max(-10, min(10, score)),  # -10 és 10 között
            "reasons": reasons[:3],  # Legfeljebb 3 ok
            "count": len(articles)
        }
    
    def _get_fallback_news(self, team_name):
        """Fallback hírek RSS-ből"""
        try:
            rss_url = f"https://news.google.com/rss/search?q={team_name}+football&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            score = 0
            reasons = []
            
            for entry in feed.entries[:3]:
                title = entry.title.lower()
                if any(word in title for word in ["injury", "suspended", "doubt"]):
                    score -= 2
                    reasons.append(f"🔴 {entry.title[:50]}...")
                elif any(word in title for word in ["win", "return", "sign"]):
                    score += 2
                    reasons.append(f"🟢 {entry.title[:50]}...")
            
            return {
                "score": max(-10, min(10, score)),
                "reasons": reasons,
                "count": len(feed.entries[:3])
            }
        except:
            return {"score": 0, "reasons": [], "count": 0}

class WeatherAPI:
    """Időjárás API"""
    
    def __init__(self):
        self.api_key = CONFIG.WEATHER_API_KEY
    
    async def get_weather(self, city, country, match_time):
        """Időjárás lekérése"""
        if not self.api_key:
            return self._get_default_weather()
        
        # Dátum konverzió
        match_dt = datetime.fromisoformat(match_time.replace('Z', '+00:00'))
        
        # Stadion koordináták (egyszerűsítve)
        city_coords = self._get_city_coords(city, country)
        
        if not city_coords:
            return self._get_default_weather()
        
        url = f"{CONFIG.WEATHER_BASE_URL}/weather"
        params = {
            "lat": city_coords["lat"],
            "lon": city_coords["lon"],
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_weather(data, match_dt)
                    else:
                        return self._get_default_weather()
                        
        except Exception as e:
            return self._get_default_weather()
    
    def _get_city_coords(self, city, country):
        """Város koordinátái"""
        # Főbb stadionok koordinátái (egyszerűsítve)
        coords_db = {
            ("london", "england"): {"lat": 51.5560, "lon": -0.2795},
            ("manchester", "england"): {"lat": 53.4831, "lon": -2.2004},
            ("liverpool", "england"): {"lat": 53.4308, "lon": -2.9608},
            ("madrid", "spain"): {"lat": 40.4531, "lon": -3.6883},
            ("barcelona", "spain"): {"lat": 41.3809, "lon": 2.1228},
            ("milan", "italy"): {"lat": 45.4781, "lon": 9.1240},
            ("munich", "germany"): {"lat": 48.2188, "lon": 11.6247},
            ("paris", "france"): {"lat": 48.8414, "lon": 2.2530},
        }
        
        city_lower = city.lower()
        country_lower = country.lower()
        
        for (c, cntry), coord in coords_db.items():
            if c in city_lower or city_lower in c:
                return coord
        
        # Alapértelmezett: London
        return {"lat": 51.5074, "lon": -0.1278}
    
    def _process_weather(self, weather_data, match_time):
        """Időjárás adatok feldolgozása"""
        main = weather_data.get("main", {})
        wind = weather_data.get("wind", {})
        rain = weather_data.get("rain", {})
        weather = weather_data.get("weather", [{}])[0]
        
        wind_speed = wind.get("speed", 0)  # m/s
        rain_1h = rain.get("1h", 0)
        
        # Hatás számítás
        impact = 0
        if wind_speed > 10:  # > 36 km/h
            impact -= 3
        if wind_speed > 15:  # > 54 km/h
            impact -= 5
        if rain_1h > 5:  # > 5mm/óra
            impact -= 2
        if rain_1h > 10:  # > 10mm/óra
            impact -= 4
        
        return {
            "temperature": main.get("temp", 15),
            "wind_speed": wind_speed,
            "wind_gust": wind.get("gust", 0),
            "rain_1h": rain_1h,
            "humidity": main.get("humidity", 60),
            "description": weather.get("description", "clear"),
            "main": weather.get("main", "Clear"),
            "impact_score": max(-10, min(10, impact))
        }
    
    def _get_default_weather(self):
        """Alapértelmezett időjárás"""
        return {
            "temperature": 15,
            "wind_speed": 3,
            "wind_gust": 5,
            "rain_1h": 0,
            "humidity": 60,
            "description": "clear sky",
            "main": "Clear",
            "impact_score": 0
        }

class APIManager:
    """Összes API kezelő"""
    
    def __init__(self):
        self.sportmonks = SportMonksAPI()
        self.odds = OddsAPI()
        self.news = NewsAnalyzer()
        self.weather = WeatherAPI()
    
    async def get_matches(self, date=None, league_ids=None):
        """Meccsek lekérése minden adattal"""
        # 1. Alap meccs adatok
        fixtures = await self.sportmonks.get_fixtures(date, league_ids)
        
        if not fixtures:
            return []
        
        # 2. Odds-ok
        odds_list = await self.odds.get_odds()
        
        # 3. Párhuzamos adatgyűjtés minden meccshez
        tasks = []
        for fixture in fixtures:
            tasks.append(self._enrich_match(fixture, odds_list))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. Hibák szűrése
        enriched_matches = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                st.warning(f"Hiba a {fixtures[i]['home_team']} vs {fixtures[i]['away_team']} adatgyűjtésnél")
                enriched_matches.append(fixtures[i])  # Csak alap adatok
            else:
                enriched_matches.append(result)
        
        return enriched_matches
    
    async def _enrich_match(self, match_data, odds_list):
        """Meccs adatok gazdagítása"""
        enriched = match_data.copy()
        
        # 1. Odds-ok
        enriched["odds"] = self.odds.process_match_odds(match_data, odds_list)
        
        # 2. Hírek
        home_news = await self.news.get_team_news(match_data["home_team"])
        away_news = await self.news.get_team_news(match_data["away_team"])
        enriched["news"] = {
            "home": home_news,
            "away": away_news,
            "combined_score": (home_news["score"] + away_news["score"]) / 2
        }
        
        # 3. Időjárás
        if match_data.get("city") and match_data.get("date"):
            weather = await self.weather.get_weather(
                match_data["city"],
                match_data["league_country"],
                match_data["date"]
            )
            enriched["weather"] = weather
        
        # 4. Bíró statisztikák
        if match_data.get("referee_id"):
            ref_stats = await self.sportmonks.get_referee_stats(match_data["referee_id"])
            enriched["referee_stats"] = ref_stats
        elif match_data.get("referee"):
            enriched["referee_stats"] = self.sportmonks._get_fallback_referee_stats()
        
        return enriched

# ==================== INTELLIGENCE ENGINE ====================
class RiskLevel(Enum):
    LOW = "🟢 AJÁNLOTT"
    MEDIUM = "🟡 RIZIKÓS"
    HIGH = "🔴 NEM AJÁNLOTT"

class IntelligenceEngine:
    """Fő intelligencia motor"""
    
    def __init__(self):
        pass
    
    def analyze_match(self, match_data):
        """Teljes meccs elemzés"""
        try:
            # 1. Alap statisztika (50%)
            stats_score = self._calculate_stats_score(match_data)
            
            # 2. Bírói hatás (20%)
            referee_score, ref_details = self._analyze_referee(match_data)
            
            # 3. Hírhatás (20%)
            sentiment_score, sentiment_details = self._analyze_sentiment(match_data)
            
            # 4. Időjárás hatás (10%)
            weather_score, weather_details = self._analyze_weather(match_data)
            
            # 5. Végső pontszám
            final_score = (stats_score * 0.5) + (referee_score * 0.2) + \
                         (sentiment_score * 0.2) + (weather_score * 0.1)
            
            # 6. Kockázati szint
            risk_level = self._determine_risk_level(final_score, match_data)
            
            # 7. Ajánlások
            recommendations = self._generate_recommendations(match_data, final_score)
            
            # 8. Indoklás
            reasoning = self._generate_reasoning(
                stats_score, referee_score, sentiment_score, 
                weather_score, match_data, recommendations
            )
            
            return {
                "confidence": round(final_score, 1),
                "risk_level": risk_level.value,
                "metrics": {
                    "base_stats": round(stats_score, 1),
                    "referee_impact": round(referee_score, 1),
                    "social_sentiment": round(sentiment_score, 1),
                    "weather_factor": round(weather_score, 1)
                },
                "recommendations": recommendations,
                "reasoning": reasoning,
                "details": {
                    "referee": ref_details,
                    "sentiment": sentiment_details,
                    "weather": weather_details
                }
            }
            
        except Exception as e:
            st.error(f"Elemzési hiba: {e}")
            return self._get_default_analysis(match_data)
    
    def _calculate_stats_score(self, match_data):
        """Statisztikai pontszám"""
        score = 50  # Alap
        
        # Forma (utolsó 5 meccs)
        home_form = match_data.get("home_form", [])
        away_form = match_data.get("away_form", [])
        
        if home_form and away_form:
            home_points = sum([3 if r == "W" else 1 if r == "D" else 0 for r in home_form[:5]])
            away_points = sum([3 if r == "W" else 1 if r == "D" else 0 for r in away_form[:5]])
            form_diff = home_points - away_points
            score += form_diff * 2
        
        # xG különbség
        home_xg = match_data.get("home_xg", 1.5)
        away_xg = match_data.get("away_xg", 1.5)
        score += (home_xg - away_xg) * 10
        
        # Helyszín előny
        score += 5
        
        return max(0, min(100, score))
    
    def _analyze_referee(self, match_data):
        """Bírói elemzés"""
        ref_stats = match_data.get("referee_stats", {})
        strictness = ref_stats.get("strictness", "Medium")
        
        # Pontszám szigorúság alapján
        strictness_scores = {
            "Very High": 80,
            "High": 70,
            "Medium": 50,
            "Low": 30
        }
        
        score = strictness_scores.get(strictness, 50)
        
        # Vélemény
        if strictness in ["Very High", "High"]:
            verdict = f"{match_data.get('referee', 'A bíró')} szigorú ({strictness}), magas lapszám várható"
            impact = "pozitív a lap tippre"
        elif strictness == "Low":
            verdict = f"{match_data.get('referee', 'A bíró')} laza ({strictness}), alacsonyabb lapszám"
            impact = "negatív a lap tippre"
        else:
            verdict = f"{match_data.get('referee', 'A bíró')} átlagos szigorúságú"
            impact = "semleges"
        
        return score, {"verdict": verdict, "impact": impact}
    
    def _analyze_sentiment(self, match_data):
        """Hangulatelemzés"""
        news = match_data.get("news", {})
        combined_score = news.get("combined_score", 0)
        
        # Pontszám (0-100)
        base_score = 50
        sentiment_score = base_score + (combined_score * 10)
        
        # Vélemény
        home_reasons = news.get("home", {}).get("reasons", [])
        away_reasons = news.get("away", {}).get("reasons", [])
        
        if combined_score > 2:
            verdict = "Erősen pozitív hírkörnyezet"
            impact = "nagyon pozitív"
        elif combined_score > 0.5:
            verdict = "Pozitív hírkörnyezet"
            impact = "pozitív"
        elif combined_score > -0.5:
            verdict = "Semleges hírkörnyezet"
            impact = "semleges"
        elif combined_score > -2:
            verdict = "Negatív hírkörnyezet"
            impact = "negatív"
        else:
            verdict = "Erősen negatív hírkörnyezet"
            impact = "nagyon negatív"
        
        return max(0, min(100, sentiment_score)), {
            "verdict": verdict,
            "impact": impact,
            "home_reasons": home_reasons[:2],
            "away_reasons": away_reasons[:2]
        }
    
    def _analyze_weather(self, match_data):
        """Időjárás elemzés"""
        weather = match_data.get("weather", {})
        impact_score = weather.get("impact_score", 0)
        
        # Pontszám (0-100)
        base_score = 50
        weather_score = base_score + (impact_score * 5)
        
        wind_speed = weather.get("wind_speed", 0)
        rain = weather.get("rain_1h", 0)
        
        if wind_speed > 15 or rain > 10:
            verdict = "Súlyos időjárási viszonyok, jelentős hatás"
            impact = "nagyon negatív"
        elif wind_speed > 10 or rain > 5:
            verdict = "Rossz időjárás, mérsékelt hatás"
            impact = "negatív"
        elif wind_speed > 5 or rain > 2:
            verdict = "Kedvezőtlen időjárás, kismértékű hatás"
            impact = "enyhén negatív"
        else:
            verdict = "Ideális időjárási viszonyok"
            impact = "pozitív"
        
        return max(0, min(100, weather_score)), {
            "verdict": verdict,
            "impact": impact,
            "wind_speed": wind_speed,
            "rain": rain
        }
    
    def _determine_risk_level(self, confidence, match_data):
        """Kockázati szint meghatározása"""
        extra_risk = 0
        
        # Kupameccs
        if "cup" in match_data.get("league_name", "").lower():
            extra_risk += 15
        
        # Derbi
        if self._is_derby(match_data):
            extra_risk += 10
        
        adjusted = confidence - extra_risk
        
        if adjusted >= 70:
            return RiskLevel.LOW
        elif adjusted >= 45:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _is_derby(self, match_data):
        """Derbi ellenőrzés"""
        derbies = [
            ("liverpool", "manchester united"),
            ("manchester city", "manchester united"),
            ("real madrid", "barcelona"),
            ("ac milan", "inter milan"),
            ("bayern munich", "borussia dortmund"),
            ("arsenal", "tottenham")
        ]
        
        home = match_data.get("home_team", "").lower()
        away = match_data.get("away_team", "").lower()
        
        for team1, team2 in derbies:
            if (team1 in home and team2 in away) or (team2 in home and team1 in away):
                return True
        return False
    
    def _generate_recommendations(self, match_data, confidence):
        """Ajánlások generálása"""
        recs = []
        
        # BTTS ajánlás
        home_xg = match_data.get("home_xg", 1.5)
        away_xg = match_data.get("away_xg", 1.5)
        
        if home_xg > 1.2 and away_xg > 1.2:
            recs.append({
                "market": "BTTS",
                "recommendation": "IGEN",
                "odds": match_data.get("odds", {}).get("btts_yes", 1.8),
                "reason": "Mindkét csapat támadóerős"
            })
        
        # Eredmény ajánlás
        if confidence >= 65:
            odds = match_data.get("odds", {})
            recs.append({
                "market": "1X2",
                "recommendation": "1",
                "odds": odds.get("home_win", 2.0),
                "reason": "Otthoni előny és jó forma"
            })
        
        # Lapok ajánlás
        ref_strictness = match_data.get("referee_stats", {}).get("strictness", "Medium")
        if ref_strictness in ["High", "Very High"]:
            recs.append({
                "market": "Lapok",
                "recommendation": "Over 4.5",
                "odds": match_data.get("odds", {}).get("cards_over_4_5", 2.1),
                "reason": f"Szigorú bíró: {ref_strictness}"
            })
        
        return recs
    
    def _generate_reasoning(self, stats, referee, sentiment, weather, match_data, recommendations):
        """Indoklás magyar nyelven"""
        parts = []
        
        # Statisztika
        if stats >= 70:
            parts.append("A statisztikai adatok erősen támogatják a hazai csapatot.")
        elif stats >= 60:
            parts.append("A statisztikák enyhe előnyt mutatnak a hazai csapatnak.")
        else:
            parts.append("A statisztikai adatok nem mutatnak egyértelmű előnyt.")
        
        # Bíró
        if referee >= 70:
            parts.append("A bíró szigorúsága magasabb lapszámot sugall.")
        elif referee <= 40:
            parts.append("A bíró lazább stílusa kevesebb lapot vet előre.")
        
        # Hírek
        if sentiment >= 70:
            parts.append("A hírkörnyezet pozitív, ami plusz motivációt adhat.")
        elif sentiment <= 40:
            parts.append("Negatív hírek gyengíthetik a csapat morálját.")
        
        # Időjárás
        if weather <= 40:
            parts.append("A rossz időjárás lassíthatja a játékot és csökkentheti a gólokat.")
        
        # Végső összegzés
        confidence = (stats * 0.5) + (referee * 0.2) + (sentiment * 0.2) + (weather * 0.1)
        
        if confidence >= 70:
            parts.append("Összességében erős ajánlás, az elemzés többsége pozitív.")
        elif confidence >= 50:
            parts.append("Kevert jelek, enyhe kockázattal járó ajánlás.")
        else:
            parts.append("Jelentős kockázatok, csak kis tétel ajánlott.")
        
        return " ".join(parts)
    
    def _get_default_analysis(self, match_data):
        """Alapértelmezett elemzés hiba esetén"""
        return {
            "confidence": 50,
            "risk_level": RiskLevel.MEDIUM.value,
            "metrics": {"base_stats": 50, "referee_impact": 50, "social_sentiment": 50, "weather_factor": 50},
            "recommendations": [],
            "reasoning": "Nem sikerült teljes elemzést készíteni.",
            "details": {}
        }

# ==================== STREAMLIT UI ====================
def main():
    """Fő Streamlit alkalmazás"""
    
    # Oldal beállítás
    st.set_page_config(
        page_title="INTELLIGENCE CONTROL V9.2",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS stílusok
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            border-left: 8px solid #00ff41;
            box-shadow: 0 10px 30px rgba(0, 255, 65, 0.1);
        }
        
        .main-title {
            font-family: 'Courier New', monospace;
            font-size: 3rem;
            font-weight: 700;
            color: #00ff41;
            text-align: center;
            letter-spacing: 2px;
            margin: 0;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        }
        
        .match-card {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 5px solid #00ff41;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .match-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 255, 65, 0.15);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%);
            color: #000;
            font-weight: bold;
            border: none;
            padding: 10px 25px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #00ffcc 0%, #00ff41 100%);
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0, 255, 65, 0.3);
        }
        
        .metric-box {
            background: rgba(26, 26, 46, 0.8);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
            margin: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Fejléc
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">🛡️ INTELLIGENCE CONTROL V9.2</h1>
            <p style='text-align: center; color: #aaa;'>
                MISSION ACTIVE | SPORTMONKS API INTEGRATED | SYSTEM: OPERATIONAL
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # API validáció
    if not CONFIG.validate():
        st.error("❌ HIÁNYZÓ API KULCSOK! Ellenőrizd a secrets.toml fájlt.")
        return
    
    # Szidebar
    with st.sidebar:
        st.markdown("### 🎮 MISSION CONTROL")
        
        # Dátum választó
        selected_date = st.date_input(
            "📅 Dátum",
            datetime.now(),
            help="Válaszd ki az elemzendő napot"
        )
        
        # Liga választó
        st.markdown("### 🏆 Ligák")
        
        # Liga listázása (cache-elve)
        @st.cache_data(ttl=3600)
        async def get_leagues():
            api_manager = APIManager()
            return await api_manager.sportmonks.get_leagues()
        
        leagues = asyncio.run(get_leagues())
        
        if leagues:
            league_options = {f"{l['name']} ({l['country']})": l['id'] for l in leagues}
            selected_league_names = st.multiselect(
                "Válassz ligákat",
                list(league_options.keys()),
                default=list(league_options.keys())[:3]
            )
            selected_league_ids = [league_options[name] for name in selected_league_names]
        else:
            st.warning("Nem sikerült lekérni a ligákat")
            selected_league_ids = None
        
        # Beállítások
        st.markdown("### ⚙️ Beállítások")
        
        auto_refresh = st.checkbox("🔄 Automata frissítés (60s)", value=False)
        show_details = st.checkbox("🔍 Részletes elemzés", value=True)
        
        # Statisztikák
        st.markdown("---")
        st.markdown("### 📈 Teljesítmény")
        
        perf_stats = DB.get_performance(30)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Találati arány", f"{perf_stats['win_rate']}%")
        with col2:
            st.metric("💰 Profit", f"{perf_stats['total_profit']}€")
        
        # Frissítés gomb
        if st.button("🔄 Adatok frissítése", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Fő tartalom
    st.markdown("### 🎯 AKTÍV MECCS ELEMZÉSEK")
    
    # Adatok betöltése
    @st.cache_data(ttl=300)
    async def load_matches(date, league_ids):
        api_manager = APIManager()
        return await api_manager.get_matches(date.strftime("%Y-%m-%d"), league_ids)
    
    with st.spinner("🤖 Adatok betöltése és elemzése..."):
        matches = asyncio.run(load_matches(selected_date, selected_league_ids))
    
    if not matches:
        st.info("ℹ️ Nincsenek meccsek a kiválasztott napon.")
        
        # Demo adatok
        if st.checkbox("Demo adatok betöltése"):
            matches = [
                {
                    "id": 1,
                    "home_team": "Manchester United",
                    "away_team": "Liverpool",
                    "league_name": "Premier League",
                    "date": datetime.now().isoformat(),
                    "home_xg": 1.8,
                    "away_xg": 2.1,
                    "referee": "Michael Oliver",
                    "venue": "Old Trafford",
                    "city": "Manchester",
                    "home_form": ["W", "W", "L", "D", "W"],
                    "away_form": ["W", "D", "W", "W", "L"]
                },
                {
                    "id": 2,
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "league_name": "La Liga",
                    "date": datetime.now().isoformat(),
                    "home_xg": 2.2,
                    "away_xg": 1.9,
                    "referee": "Anthony Taylor",
                    "venue": "Santiago Bernabéu",
                    "city": "Madrid",
                    "home_form": ["W", "W", "W", "D", "W"],
                    "away_form": ["W", "L", "W", "D", "W"]
                }
            ]
    
    # Elemzés motor
    intelligence = IntelligenceEngine()
    
    # Meccsek megjelenítése
    for match in matches:
        # Elemzés
        analysis = intelligence.analyze_match(match)
        
        # Kártya
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"""
            <div class='match-card'>
                <h3>⚽ {match['home_team']} vs {match['away_team']}</h3>
                <p style='color: #aaa;'>
                    🏆 {match.get('league_name', 'Unknown')} | 
                    ⏰ {datetime.fromisoformat(match['date'].replace('Z', '+00:00')).strftime('%H:%M') if 'date' in match else 'TBA'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Ajánlások
            if analysis['recommendations']:
                for rec in analysis['recommendations']:
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.markdown(f"**{rec['market']}**: {rec['recommendation']}")
                    with col_b:
                        st.metric("Odds", f"{rec['odds']:.2f}")
                    with col_c:
                        pick_data = {
                            'match_id': match['id'],
                            'home_team': match['home_team'],
                            'away_team': match['away_team'],
                            'league': match.get('league_name', 'Unknown'),
                            'pick_type': rec['market'],
                            'pick_value': rec['recommendation'],
                            'odds': rec['odds'],
                            'confidence': analysis['confidence'],
                            'status': analysis['risk_level']
                        }
                        if st.button("💾", key=f"save_{match['id']}_{rec['market']}"):
                            if DB.save_pick(pick_data):
                                st.success("✅")
                            else:
                                st.error("❌")
        
        with col2:
            # Bizalom mutató
            color_map = {
                "🟢 AJÁNLOTT": "#00ff41",
                "🟡 RIZIKÓS": "#ffff00",
                "🔴 NEM AJÁNLOTT": "#ff0000"
            }
            
            color = color_map.get(analysis['risk_level'], "#aaa")
            
            st.markdown(f"""
            <div style='text-align: center;'>
                <h1 style='color: {color}; font-size: 2.5rem;'>{analysis['confidence']}%</h1>
                <h3>{analysis['risk_level']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Gyors információk
            if match.get('referee'):
                st.caption(f"⚖️ {match['referee']}")
            if match.get('venue'):
                st.caption(f"📍 {match['venue']}")
        
        # Részletes elemzés
        if show_details:
            with st.expander("📊 Részletes elemzés"):
                tabs = st.tabs(["📈 Metrikák", "📢 Hírek", "⚖️ Bíró", "🌤️ Időjárás"])
                
                with tabs[0]:
                    # Metrikák grafikon
                    metrics = analysis['metrics']
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(metrics.keys()),
                            y=list(metrics.values()),
                            marker_color=['#00ff41', '#ffaa00', '#ff4444', '#4488ff'],
                            text=list(metrics.values()),
                            texttemplate='%{text:.1f}'
                        )
                    ])
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Indoklás
                    st.markdown("#### 🧠 Indoklás")
                    st.info(analysis['reasoning'])
                
                with tabs[1]:
                    # Hírek
                    news = match.get('news', {})
                    if news:
                        home_news = news.get('home', {})
                        away_news = news.get('away', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**{match['home_team']}**")
                            for reason in home_news.get('reasons', [])[:2]:
                                st.write(reason)
                        with col2:
                            st.markdown(f"**{match['away_team']}**")
                            for reason in away_news.get('reasons', [])[:2]:
                                st.write(reason)
                    else:
                        st.write("Nincsenek hírek")
                
                with tabs[2]:
                    # Bíró
                    ref_details = analysis['details']['referee']
                    st.markdown(f"**Vélemény:** {ref_details['verdict']}")
                    st.markdown(f"**Hatás:** {ref_details['impact']}")
                
                with tabs[3]:
                    # Időjárás
                    weather = match.get('weather', {})
                    if weather:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🌡️ Hőmérséklet", f"{weather.get('temperature', 0)}°C")
                        with col2:
                            st.metric("💨 Szél", f"{weather.get('wind_speed', 0)} m/s")
                        with col3:
                            st.metric("🌧️ Eső", f"{weather.get('rain_1h', 0)} mm")
        
        st.divider()
    
    # Teljesítmény grafikon
    st.markdown("---")
    st.markdown("### 📊 TELJESÍTMÉNY STATISZTIKA")
    
    perf = DB.get_performance(90)
    
    if perf['total'] > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Win rate gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=perf['win_rate'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Találati arány"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00ff41"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255, 68, 68, 0.3)"},
                        {'range': [50, 65], 'color': "rgba(255, 170, 0, 0.3)"},
                        {'range': [65, 100], 'color': "rgba(0, 255, 65, 0.3)"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profit grafikon
            fig = go.Figure(data=[
                go.Scatter(
                    y=[0, perf['total_profit']],
                    mode="lines+markers",
                    line=dict(color="#00ff41", width=4),
                    marker=dict(size=15)
                )
            ])
            fig.update_layout(title="Profit", height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Összefoglaló
            st.markdown("#### 📈 Összefoglaló")
            st.metric("Összes fogadás", perf['total'])
            st.metric("Nyereség/Veszteség", f"{perf['wins']}/{perf['losses']}")
            st.metric("ROI", f"{perf['roi']}%")
    
    # Automata frissítés
    if auto_refresh:
        st.markdown("---")
        st.caption(f"Utolsó frissítés: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(60)
        st.rerun()

# ==================== ALKALMAZÁS FUTTATÁSA ====================
if __name__ == "__main__":
    main()
