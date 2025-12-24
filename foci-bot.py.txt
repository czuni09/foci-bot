"""
Kibővített futballmérkőzés‑elemző modul.
Tartalmazza: Időjárás, AFCON hiányzók, Bírói statisztika, Hírek/Botrányok.
"""

from __future__ import annotations
import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import requests
import math

# API kulcsok - Ellenőrizd, hogy érvényesek-e
WEATHER_KEY: str = "c31a011d35fed1b4d7b9f222c99d6dd2"
NEWS_KEY: str = "7d577a4d9f2b4ba38541cc3f7e5ad6f5"

@dataclass
class TeamStats:
    name: str
    attack_rating: float
    defence_rating: float
    last5_results: List[int] = field(default_factory=list)
    injuries: List[str] = field(default_factory=list)
    intl_absences: List[str] = field(default_factory=list)
    suspensions: List[str] = field(default_factory=list)

    def form_score(self) -> float:
        if not self.last5_results: return 0.0
        return sum((3 if r == 1 else 1 if r == 0 else 0) for r in self.last5_results)

    def absence_penalty(self) -> Tuple[float, List[str]]:
        penalty = 0.0
        reasons = []
        if self.injuries:
            penalty -= 3 * len(self.injuries)
            reasons.append(f"Sérültek: {len(self.injuries)}")
        if self.intl_absences:
            penalty -= 5 * len(self.intl_absences)
            reasons.append(f"AFCON/Válogatott hiányzók: {len(self.intl_absences)}")
        return penalty, reasons

@dataclass
class RefereeStats:
    name: str
    cards_per_game: float
    red_card_rate: float

def fetch_news_penalty(team_name: str) -> Tuple[float, List[str]]:
    query = f"{team_name} (AFCON OR 'Africa Cup' OR injury OR scandal OR divorce OR arrest)"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWS_KEY}"
    penalty = 0.0
    findings = []
    try:
        res = requests.get(url, timeout=10).json()
        articles = res.get("articles", [])[:5]
        for art in articles:
            text = (art.get("title", "") + " " + art.get("description", "")).lower()
            if any(word in text for word in ["afcon", "africa cup", "international duty"]):
                penalty -= 5
                findings.append("AFCON behívó hír")
            if any(word in text for word in ["scandal", "divorce", "arrest", "police"]):
                penalty -= 4
                findings.append("Botrány/Magánéleti hír")
    except: pass
    return penalty, list(set(findings))

def fetch_weather_data(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={WEATHER_KEY}"
    try:
        return requests.get(url, timeout=10).json()
    except: return None

def compute_weather_penalty(weather_data):
    if not weather_data or "main" not in weather_data: return 0.0, []
    penalty = 0.0
    reasons = []
    temp = weather_data["main"]["temp"]
    desc = weather_data["weather"][0]["main"].lower()
    
    if temp >= 30:
        penalty -= 3
        reasons.append(f"Hőség ({temp}°C)")
    if "rain" in desc or "storm" in desc:
        penalty -= 5
        reasons.append("Esős/Viharos idő")
    return penalty, reasons

def load_referee_stats():
    return {
        "Anthony Taylor": RefereeStats("Anthony Taylor", 4.5, 0.05),
        "Szymon Marciniak": RefereeStats("Szymon Marciniak", 5.2, 0.04),
        "Istvan Kovacs": RefereeStats("Istvan Kovacs", 4.0, 0.03)
    }

def ultimate_football_bot(home: TeamStats, away: TeamStats, city: str, referee: str, competition: str, odds: float, base_prob: float, avg_goals: float):
    print(f"\n⚽ ELEMZÉS: {home.name} vs {away.name} ({competition})")
    print("-" * 50)
    
    score = float(base_prob)
    logs = []

    # 1. Kupa-faktor (Bajnokság vs Kupa)
    if any(x in competition for x in ["Cup", "Kupa", "Copa"]):
        score -= 10
        logs.append("🏆 Kupa-meccs: Alacsonyabb prioritás/rotáció (-10%)")

    # 2. Hiányzók és Hírek (AFCON/Sérülés)
    for team in [home, away]:
        p, r = team.absence_penalty()
        score += p
        if r: logs.append(f"🚑 {team.name} hiányzók: {r} ({p}%)")
        
        np, news = fetch_news_penalty(team.name)
        score += np
        if news: logs.append(f"📰 {team.name} hírek: {news} ({np}%)")

    # 3. Időjárás
    weather_data = fetch_weather_data(city)
    wp, wr = compute_weather_penalty(weather_data)
    score += wp
    if wr: logs.append(f"🌦️ Időjárás: {wr} ({wp}%)")

    # 4. Bíró
    ref_stats = load_referee_stats()
    if referee in ref_stats:
        if ref_stats[referee].cards_per_game > 4.5:
            logs.append(f"🟨 Bíró: {referee} szigorú (Sok lap várható)")

    # VÉGEREDMÉNY
    final_score = max(min(score, 100), 0)
    print(f"\n✅ Végső valószínűség: {final_score:.1f}%")
    for log in logs: print(f"  • {log}")

    print("\n[TIPPEK]:")
    if final_score >= 75 and odds >= 1.4:
        print(f"👉 FŐ TIPP: {home.name} GYŐZELEM (Biztonság: {final_score:.1f}%)")
    else:
        print(f"👉 MAI LEGKISEBB KOCKÁZAT: {home.name} (Biztonság: {final_score:.1f}%)")

# E-mail küldő rész czunidaniel9@gmail.com részére
def send_email_alert(report_text):
    # Itt küldené az e-mailt a rendszer
    print(f"Küldés a czunidaniel9@gmail.com címre...")

if __name__ == "__main__":
    # Példa adatok
    arsenal = TeamStats("Arsenal", 85, 80, [1, 1, 0, 1, -1], injuries=["Gabriel Jesus"], intl_absences=["Thomas Partey"])
    palace = TeamStats("Crystal Palace", 70, 75, [0, 1, -1, 0, 0], intl_absences=["Jordan Ayew"])
    
    ultimate_football_bot(arsenal, palace, "London", "Anthony Taylor", "Premier League", 1.48, 80, 2.5)