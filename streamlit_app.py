"""
TITAN Bot - Complete UI & Execution Logic
==========================================
Folytatás: UI renderelés, mentés, backtest
"""

# ============================================================================
#  MAIN UI & STYLING
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
:root {
  --bg: #0a0e1a;
  --card: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.92);
  --accent: #3b82f6;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
}
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  color: var(--text);
}
.stApp {
  background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
}
.titan-header {
  font-size: 2.5rem;
  font-weight: 800;
  background: linear-gradient(90deg, var(--accent), var(--success));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
}
.match-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  margin: 15px 0;
  box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.score-badge {
  display: inline-block;
  padding: 6px 14px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.9rem;
}
.score-high { background: rgba(16,185,129,0.2); color: var(--success); border: 1px solid var(--success); }
.score-med { background: rgba(245,158,11,0.2); color: var(--warning); border: 1px solid var(--warning); }
.score-low { background: rgba(239,68,68,0.2); color: var(--danger); border: 1px solid var(--danger); }
.bet-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  background: rgba(59,130,246,0.15);
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--accent);
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 15px 0;
}
.stat-box {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 12px;
  text-align: center;
}
.stat-label { color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-bottom: 4px; }
.stat-value { font-size: 1.2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="titan-header">⚽ TITAN - Strategic Intelligence</div>', unsafe_allow_html=True)
st.caption("**Autonóm fogadási rendszer:** xG analízis + best odds + social signals + derby kizárás")


# ============================================================================
#  DATABASE OPERATIONS
# ============================================================================
def save_prediction(pred: dict):
    """Save prediction to database"""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = con.cursor()
    
    cur.execute("""
        INSERT INTO predictions (
            created_at, match, home, away, league, kickoff_utc,
            bet_type, market_key, selection, line, bookmaker, odds,
            score, reasoning, xg_home, xg_away,
            football_data_match_id, opening_odds, data_quality
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_utc().isoformat(),
        pred.get("match"),
        pred.get("home"),
        pred.get("away"),
        pred.get("league"),
        pred.get("kickoff_utc").isoformat() if pred.get("kickoff_utc") else None,
        pred.get("bet_type"),
        pred.get("market_key"),
        pred.get("selection"),
        pred.get("line"),
        pred.get("bookmaker"),
        pred.get("odds"),
        pred.get("score"),
        pred.get("reasoning"),
        pred.get("xg_home"),
        pred.get("xg_away"),
        pred.get("fd_match_id"),
        pred.get("odds"),  # opening_odds = current odds
        "LIVE"
    ))
    
    con.commit()
    pred_id = cur.lastrowid
    con.close()
    return pred_id

def get_predictions_df():
    """Load all predictions from database"""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    df = pd.read_sql_query("""
        SELECT * FROM predictions 
        ORDER BY created_at DESC
    """, con)
    con.close()
    return df

def update_prediction_result(pred_id: int, result: str, home_goals: int, away_goals: int):
    """Update prediction with match result"""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
        UPDATE predictions 
        SET result = ?, settled_at = ?, home_goals = ?, away_goals = ?
        WHERE id = ?
    """, (result, now_utc().isoformat(), home_goals, away_goals, pred_id))
    con.commit()
    con.close()

def settle_prediction(pred: dict) -> str:
    """Determine bet outcome based on result"""
    bet_type = pred.get("bet_type")
    hg = pred.get("home_goals")
    ag = pred.get("away_goals")
    
    if hg is None or ag is None:
        return "PENDING"
    
    if bet_type == "H2H":
        selection = pred.get("selection", "")
        home = pred.get("home", "")
        is_home_bet = team_match_score(selection, home) >= 0.7
        
        if is_home_bet:
            return "WON" if hg > ag else "LOST"
        else:
            return "WON" if ag > hg else "LOST"
    
    elif bet_type == "TOTALS":
        line = safe_float(pred.get("line"), 2.5)
        total = hg + ag
        selection = pred.get("selection", "").lower()
        
        if abs(total - line) < 0.01:
            return "VOID"
        
        if selection == "over":
            return "WON" if total > line else "LOST"
        else:
            return "WON" if total < line else "LOST"
    
    return "UNKNOWN"

def refresh_results():
    """Check and update results for pending predictions"""
    df = get_predictions_df()
    pending = df[df["result"].isin(["PENDING", "UNKNOWN"])]
    
    if pending.empty:
        return 0
    
    updated = 0
    for _, row in pending.iterrows():
        kickoff = parse_dt(row.get("kickoff_utc", ""))
        if not kickoff or now_utc() < kickoff + timedelta(hours=2):
            continue
        
        match_id = row.get("football_data_match_id")
        if not match_id:
            continue
        
        result_data = fd_get_result(int(match_id))
        if not result_data:
            continue
        
        hg = result_data["home_goals"]
        ag = result_data["away_goals"]
        
        pred_dict = row.to_dict()
        pred_dict.update({"home_goals": hg, "away_goals": ag})
        outcome = settle_prediction(pred_dict)
        
        update_prediction_result(int(row["id"]), outcome, hg, ag)
        updated += 1
    
    return updated


# ============================================================================
#  MAIN ANALYSIS FLOW
# ============================================================================
def analyze_match(league_key: str, league_name: str, match: dict, prof: dict, odds_data: dict):
    """Complete match analysis combining all data sources"""
    home = match.get("home")
    away = match.get("away")
    kickoff = match.get("kickoff_utc")
    
    if not home or not away or not kickoff:
        return None
    
    # Check derby exclusion
    if is_excluded_match(league_key, home, away):
        return None
    
    # xG analysis
    lh, la, n_home, n_away = expected_goals_from_profiles(home, away, prof)
    
    if n_home < 3 or n_away < 3:
        return None  # Insufficient data
    
    # Social signals
    social = fetch_social_signals(home, away)
    
    # Find best odds
    odds_match = None
    for om in odds_data.get("events", []):
        h_score = team_match_score(home, om.get("home_team", ""))
        a_score = team_match_score(away, om.get("away_team", ""))
        if h_score >= 0.7 and a_score >= 0.7:
            odds_match = om
            break
    
    if not odds_match:
        return None
    
    best_odds = extract_best_odds(odds_match, home, away)
    
    # Find football-data match ID
    fd_match_id = fd_find_match_id(home, away, kickoff)
    
    # Generate bet candidates
    candidates = []
    
    # H2H candidates
    h2h = best_odds.get("h2h", {})
    if h2h.get("home") and TOTAL_ODDS_MIN <= h2h["home"] <= TOTAL_ODDS_MAX * 1.5:
        bet = {
            "match_id": f"{league_key}_{home}_{away}",
            "match": f"{home} vs {away}",
            "home": home,
            "away": away,
            "league": league_name,
            "kickoff_utc": kickoff,
            "bet_type": "H2H",
            "market_key": "h2h",
            "selection": home,
            "line": None,
            "bookmaker": "best",
            "odds": h2h["home"],
            "fd_match_id": fd_match_id,
        }
        score, reasoning = score_bet_candidate(bet, lh, la, social)
        bet.update({"score": score, "reasoning": reasoning, "xg_home": lh, "xg_away": la})
        candidates.append(bet)
    
    if h2h.get("away") and TOTAL_ODDS_MIN <= h2h["away"] <= TOTAL_ODDS_MAX * 1.5:
        bet = {
            "match_id": f"{league_key}_{home}_{away}",
            "match": f"{home} vs {away}",
            "home": home,
            "away": away,
            "league": league_name,
            "kickoff_utc": kickoff,
            "bet_type": "H2H",
            "market_key": "h2h",
            "selection": away,
            "line": None,
            "bookmaker": "best",
            "odds": h2h["away"],
            "fd_match_id": fd_match_id,
        }
        score, reasoning = score_bet_candidate(bet, lh, la, social)
        bet.update({"score": score, "reasoning": reasoning, "xg_home": lh, "xg_away": la})
        candidates.append(bet)
    
    # Totals candidates
    totals = best_odds.get("totals", {})
    for (line, side), odds_val in totals.items():
        if line in (2.5, 3.5, 1.5) and TOTAL_ODDS_MIN <= odds_val <= TOTAL_ODDS_MAX * 1.5:
            bet = {
                "match_id": f"{league_key}_{home}_{away}",
                "match": f"{home} vs {away}",
                "home": home,
                "away": away,
                "league": league_name,
                "kickoff_utc": kickoff,
                "bet_type": "TOTALS",
                "market_key": "totals",
                "selection": side.capitalize(),
                "line": line,
                "bookmaker": "best",
                "odds": odds_val,
                "fd_match_id": fd_match_id,
            }
            score, reasoning = score_bet_candidate(bet, lh, la, social)
            bet.update({"score": score, "reasoning": reasoning, "xg_home": lh, "xg_away": la})
            candidates.append(bet)
    
    return candidates


# ============================================================================
#  MAIN EXECUTION
# ============================================================================
def main():
    st.markdown("---")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Understat", "✅ Active")
    with col2:
        st.metric("Odds API", "✅ Active" if ODDS_API_KEY else "⚠️ Missing Key")
    with col3:
        st.metric("Football-Data", "✅ Active" if FOOTBALL_DATA_KEY else "⚠️ Missing Key")
    with col4:
        if st.button("🔄 Frissítés eredmények"):
            updated = refresh_results()
            st.success(f"✅ {updated} tipp frissítve")
    
    # Fetch data
    season = season_from_today()
    all_candidates = []
    
    with st.spinner("🔍 Adatok gyűjtése és elemzés..."):
        for us_key, league_name in UNDERSTAT_LEAGUES.items():
            try:
                # Understat xG data
                fixtures, results = understat_fetch(us_key, season, DAYS_AHEAD)
                prof = build_team_xg_profiles(results)
                
                # Odds API data
                odds_key = {
                    "epl": "soccer_epl",
                    "la_liga": "soccer_spain_la_liga",
                    "bundesliga": "soccer_germany_bundesliga",
                    "serie_a": "soccer_italy_serie_a",
                    "ligue_1": "soccer_france_ligue_one",
                }.get(us_key)
                
                if not odds_key:
                    continue
                
                odds_data = odds_api_get(odds_key)
                if not odds_data.get("ok"):
                    st.warning(f"⚠️ {league_name}: {odds_data.get('msg')}")
                    continue
                
                # Analyze matches
                for fx in fixtures:
                    home = clean_team(((fx.get("h") or {}).get("title")))
                    away = clean_team(((fx.get("a") or {}).get("title")))
                    kickoff = parse_dt(fx.get("datetime", ""))
                    
                    if not home or not away or not kickoff:
                        continue
                    
                    match_data = {"home": home, "away": away, "kickoff_utc": kickoff}
                    candidates = analyze_match(us_key, league_name, match_data, prof, odds_data)
                    
                    if candidates:
                        all_candidates.extend(candidates)
                
            except Exception as e:
                st.error(f"❌ {league_name} hiba: {e}")
    
    if not all_candidates:
        st.warning("⚠️ Nincs megfelelő fogadási lehetőség az időablakban")
        return
    
    # Select best duo
    duo, total_odds = pick_best_duo(all_candidates)
    
    if not duo:
        st.warning("⚠️ Nem találtunk megfelelő 2-es fogadást a dupla kritériumoknak")
        st.info(f"Találtunk {len(all_candidates)} jelöltet, de nem alkotnak megfelelő párt")
        return
    
    # Display recommendations
    st.markdown("---")
    st.markdown("## 🎯 TOP 2 AJÁNLÁS (Dupla Fogadás)")
    st.success(f"**Kombinált odds:** {total_odds:.2f} (Cél: {TARGET_TOTAL_ODDS:.2f})")
    
    for idx, bet in enumerate(duo, 1):
        score = bet.get("score", 0)
        score_class = "score-high" if score >= 70 else "score-med" if score >= 50 else "score-low"
        
        st.markdown(f"""
        <div class="match-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0;">#{idx} {bet['match']}</h3>
                <span class="score-badge {score_class}">{score:.0f}/100</span>
            </div>
            
            <div style="margin: 10px 0;">
                <span class="bet-tag">
                    {bet['bet_type']}: {bet['selection']} {f"({bet['line']})" if bet.get('line') else ""}
                </span>
                <span class="bet-tag">Odds: {bet['odds']:.2f}</span>
                <span class="bet-tag">🕐 {fmt_local(bet['kickoff_utc'])}</span>
            </div>
            
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-label">xG Hazai</div>
                    <div class="stat-value">{bet['xg_home']:.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">xG Vendég</div>
                    <div class="stat-value">{bet['xg_away']:.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Liga</div>
                    <div class="stat-value" style="font-size: 0.9rem;">{bet['league']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Részletes elemzés"):
            st.markdown(bet.get("reasoning", ""))
    
    # Save button
    if st.button("💾 Mentés adatbázisba", type="primary"):
        for bet in duo:
            save_prediction(bet)
        st.success("✅ Fogadások elmentve!")
    
    # Backtest section
    st.markdown("---")
    st.markdown("## 📈 Backtest & Eredmények")
    
    df = get_predictions_df()
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(df)
        won = len(df[df["result"] == "WON"])
        lost = len(df[df["result"] == "LOST"])
        pending = len(df[df["result"] == "PENDING"])
        
        with col1:
            st.metric("Összes tipp", total)
        with col2:
            st.metric("Nyert", won, delta=f"{won/max(1, won+lost)*100:.1f}%")
        with col3:
            st.metric("Vesztett", lost)
        with col4:
            st.metric("Függőben", pending)
        
        st.dataframe(
            df[["created_at", "match", "bet_type", "selection", "odds", "score", "result"]].head(20),
            use_container_width=True
        )
    else:
        st.info("Még nincs mentett fogadás")


if __name__ == "__main__":
    main()
