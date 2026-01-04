# app.py
import os
import re
import math
import asyncio
from datetime import datetime, timezone

import aiohttp
import pandas as pd
import streamlit as st
from understat import Understat


# =========================
#  KONFIG
# =========================
st.set_page_config(page_title="TITAN v2 – Match Intelligence", page_icon="🛰️", layout="wide")

# Opcionális: ha van Odds API kulcsod, később rá tudjuk kötni
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

LEAGUES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}

DEFAULT_SEASONS = {
    # Understat "season" tipikusan a szezon kezdő éve (pl. 2024 = 2024/25)
    "epl": 2024,
    "la_liga": 2024,
    "bundesliga": 2024,
    "serie_a": 2024,
    "ligue_1": 2024,
}


# =========================
#  UI – innovatív “mission control”
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

:root{
  --bg0:#06070c;
  --bg1:#0b1020;
  --bg2:#0e1630;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --good:#4ef0a3;
  --warn:#ffd166;
  --bad:#ff5c8a;
  --accent:#79a6ff;
  --accent2:#b387ff;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp{
  background:
    radial-gradient(900px 500px at 15% 10%, rgba(121,166,255,0.18), transparent 60%),
    radial-gradient(700px 400px at 85% 15%, rgba(179,135,255,0.14), transparent 55%),
    linear-gradient(135deg, var(--bg0) 0%, var(--bg1) 50%, var(--bg0) 100%);
}

.hdr{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  font-size: 2.1rem;
  letter-spacing: 0.2px;
  margin: 0.2rem 0 0.1rem 0;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

.sub{
  color: var(--muted);
  margin-bottom: 1rem;
}

.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  font-size: 0.86rem;
  color: rgba(255,255,255,0.86);
}

.panel{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.42);
}

.card{
  background: var(--card2);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
  margin: 10px 0;
  box-shadow: 0 14px 45px rgba(0,0,0,0.40);
}

.grid{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}

.metricbox{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 10px 12px;
}

.mtitle{ color: var(--muted); font-size: 0.82rem; margin-bottom: 4px;}
.mval{ font-weight: 800; font-size: 1.05rem; }

.tag-good{ border-color: rgba(78,240,163,0.35); background: rgba(78,240,163,0.10); }
.tag-warn{ border-color: rgba(255,209,102,0.40); background: rgba(255,209,102,0.10); }
.tag-bad { border-color: rgba(255,92,138,0.40); background: rgba(255,92,138,0.12); }

.small{ color: var(--muted); font-size: 0.9rem; }
hr{ border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr">🛰️ TITAN v2 – Match Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Understat (xG) alapú több-ligás ajánló. '
    'Mindig ad javaslatot – ha gyenge az adat, <b>RIZIKÓS / NEM AJÁNLOTT</b> címkével jelzi.</div>',
    unsafe_allow_html=True,
)


# =========================
#  Segédek
# =========================
def now_utc():
    return datetime.now(timezone.utc)

def parse_dt(s: str):
    # Understat datetime: "2024-08-16 19:00:00" (feltételezzük UTC-ként)
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def fmt_local(dt):
    if not dt:
        return "—"
    try:
        return dt.astimezone().strftime("%Y.%m.%d %H:%M")
    except Exception:
        return dt.strftime("%Y.%m.%d %H:%M")

def clamp(x, a, b):
    return max(a, min(b, x))

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def clean_team(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


# =========================
#  Poisson modell (egyszerű, stabil)
# =========================
def poisson_pmf(lmb, k):
    # stabil pmf
    return math.exp(-lmb) * (lmb ** k) / math.factorial(k)

def poisson_cdf(lmb, k):
    # P(X <= k)
    return sum(poisson_pmf(lmb, i) for i in range(k + 1))

def prob_over_25(lh, la, max_goals=10):
    # P(total >= 3)
    p = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j >= 3:
                p += poisson_pmf(lh, i) * poisson_pmf(la, j)
    return clamp(p, 0.0, 1.0)

def prob_btts(lh, la):
    # P(home>=1 and away>=1) = 1 - P(home=0) - P(away=0) + P(both=0)
    p = 1 - math.exp(-lh) - math.exp(-la) + math.exp(-(lh + la))
    return clamp(p, 0.0, 1.0)

def prob_1x2(lh, la, max_goals=10):
    # P(H win), P(draw), P(A win)
    ph = pdw = pa = 0.0
    for i in range(max_goals + 1):
        pi = poisson_pmf(lh, i)
        for j in range(max_goals + 1):
            pj = poisson_pmf(la, j)
            if i > j:
                ph += pi * pj
            elif i == j:
                pdw += pi * pj
            else:
                pa += pi * pj
    s = ph + pdw + pa
    if s > 0:
        ph, pdw, pa = ph / s, pdw / s, pa / s
    return clamp(ph, 0.0, 1.0), clamp(pdw, 0.0, 1.0), clamp(pa, 0.0, 1.0)


# =========================
#  Understat API (async) + cache
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def understat_fetch(league_key: str, season: int, days_ahead: int):
    """
    Visszaad:
      - fixtures (jövőbeli meccsek a közeljövőben)
      - results (befejezett meccsek szezonból)
      - teams (csapatlista)
    """
    async def _run():
        async with aiohttp.ClientSession() as session:
            u = Understat(session)
            fixtures = await u.get_league_fixtures(league_key, season)
            results = await u.get_league_results(league_key, season)
            teams = await u.get_league_teams(league_key, season)
            return fixtures, results, teams

    fixtures, results, teams = asyncio.run(_run())

    # szűrjük fixtures: most -> now + days_ahead
    now = now_utc()
    limit = now + pd.Timedelta(days=days_ahead)
    fx = []
    for m in fixtures or []:
        dt = parse_dt(m.get("datetime", ""))
        if not dt:
            continue
        if now <= dt <= limit:
            fx.append(m)
    fx.sort(key=lambda x: x.get("datetime", ""))

    return fx, (results or []), (teams or [])


def build_team_xg_profiles(results: list[dict]):
    """
    Egyszerű csapatprofil:
      - home_xg_for, home_xg_against (átlag)
      - away_xg_for, away_xg_against (átlag)
      - meccsszámok (adatminőség)
    """
    prof = {}
    def ensure(team):
        prof.setdefault(team, {
            "home_for": [], "home_against": [],
            "away_for": [], "away_against": [],
        })

    for m in results or []:
        h = clean_team(((m.get("h") or {}).get("title")))
        a = clean_team(((m.get("a") or {}).get("title")))
        xgh = safe_float(((m.get("xG") or {}).get("h")))
        xga = safe_float(((m.get("xG") or {}).get("a")))
        if not h or not a or xgh is None or xga is None:
            continue

        ensure(h); ensure(a)
        prof[h]["home_for"].append(xgh)
        prof[h]["home_against"].append(xga)

        prof[a]["away_for"].append(xga)
        prof[a]["away_against"].append(xgh)

    # átlagok
    out = {}
    for team, d in prof.items():
        hf = d["home_for"]; ha = d["home_against"]
        af = d["away_for"]; aa = d["away_against"]
        out[team] = {
            "home_xg_for": sum(hf)/len(hf) if hf else None,
            "home_xg_against": sum(ha)/len(ha) if ha else None,
            "away_xg_for": sum(af)/len(af) if af else None,
            "away_xg_against": sum(aa)/len(aa) if aa else None,
            "n_home": len(hf),
            "n_away": len(af),
        }
    return out


def expected_goals_from_profiles(home: str, away: str, prof: dict):
    """
    E[home goals] ~ avg(home_xg_for, away_xg_against)
    E[away goals] ~ avg(away_xg_for, home_xg_against)
    Fallback, ha kevés adat: 1.35 körüli ligaszint.
    """
    base = 1.35

    ph = prof.get(home, {})
    pa = prof.get(away, {})

    h_for = ph.get("home_xg_for")
    h_against = ph.get("home_xg_against")
    a_for = pa.get("away_xg_for")
    a_against = pa.get("away_xg_against")

    # home lambda
    lh_parts = []
    if h_for is not None: lh_parts.append(h_for)
    if a_against is not None: lh_parts.append(a_against)
    lh = sum(lh_parts)/len(lh_parts) if lh_parts else base

    # away lambda
    la_parts = []
    if a_for is not None: la_parts.append(a_for)
    if h_against is not None: la_parts.append(h_against)
    la = sum(la_parts)/len(la_parts) if la_parts else base

    lh = clamp(lh, 0.2, 3.5)
    la = clamp(la, 0.2, 3.5)

    # data quality
    n_home = ph.get("n_home", 0)
    n_away = pa.get("n_away", 0)
    return lh, la, n_home, n_away


def label_risk(n_home: int, n_away: int):
    # egyszerű, őszinte adatminőség címke
    if n_home >= 8 and n_away >= 8:
        return "AJÁNLOTT", "tag-good"
    if n_home >= 4 and n_away >= 4:
        return "RIZIKÓS", "tag-warn"
    return "NEM AJÁNLOTT", "tag-bad"


def pick_recommendation(lh, la, ph, pd, pa, pbtts, pover25):
    """
    Mindig visszaad egy “legjobb” picket + magyarázatot.
    Heurisztika:
      - ha BTTS vagy Over elég magas → azt ajánlja
      - különben 1X2 a legvalószínűbb kimenet
      - ha minden gyenge, akkor Under 2.5 / 1X (konzervatív)
    """
    total_xg = lh + la

    # jel erősségek
    btts_score = pbtts
    over_score = pover25

    # elsődleges
    if btts_score >= 0.58 and total_xg >= 2.55:
        return ("BTTS – IGEN", btts_score, f"Mindkét csapat várhatóan szerez gólt (össz xG ~ {total_xg:.2f}).")
    if over_score >= 0.56 and total_xg >= 2.60:
        return ("Over 2.5 gól", over_score, f"Magasabb gólvárakozás (össz xG ~ {total_xg:.2f}).")

    # 1X2 (legnagyobb)
    mx = max(ph, pd, pa)
    if mx == ph:
        return ("Hazai győzelem (1)", ph, f"A hazai oldal a valószínűbb (Poisson alapján ~ {ph*100:.0f}%).")
    if mx == pa:
        return ("Vendég győzelem (2)", pa, f"A vendég oldal a valószínűbb (Poisson alapján ~ {pa*100:.0f}%).")
    return ("Döntetlen (X)", pd, f"A döntetlen kiugróan valószínű (Poisson alapján ~ {pd*100:.0f}%).")


def build_match_analysis(home, away, kickoff_dt, league_name, season, prof):
    lh, la, n_home, n_away = expected_goals_from_profiles(home, away, prof)
    pbtts = prob_btts(lh, la)
    pover25 = prob_over_25(lh, la)
    ph, pdw, pa = prob_1x2(lh, la)

    risk_label, risk_class = label_risk(n_home, n_away)

    pick, pval, why = pick_recommendation(lh, la, ph, pdw, pa, pbtts, pover25)

    # magyar elemzés röviden
    summary_lines = [
        f"**Várható gól (xG alap):** {home} ~ `{lh:.2f}`, {away} ~ `{la:.2f}` (össz: `{(lh+la):.2f}`)",
        f"**BTTS (IGEN) esély:** ~ `{pbtts*100:.0f}%` | **Over 2.5 esély:** ~ `{pover25*100:.0f}%`",
        f"**1X2 becslés:** 1=`{ph*100:.0f}%` • X=`{pdw*100:.0f}%` • 2=`{pa*100:.0f}%`",
        f"**Ajánlás:** **{pick}** (bizalom: `{pval*100:.0f}%`) — {why}",
    ]

    return {
        "league": league_name,
        "season": season,
        "home": home,
        "away": away,
        "kickoff": kickoff_dt,
        "lh": lh, "la": la,
        "pbtts": pbtts,
        "pover25": pover25,
        "p1": ph, "px": pdw, "p2": pa,
        "pick": pick,
        "confidence": pval,
        "risk_label": risk_label,
        "risk_class": risk_class,
        "quality": (n_home, n_away),
        "summary": "\n".join(summary_lines),
    }


# =========================
#  Sidebar – beállítások
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Beállítások")
    sel_leagues = st.multiselect(
        "Ligák",
        options=list(LEAGUES.keys()),
        default=list(LEAGUES.keys()),
        format_func=lambda k: LEAGUES[k],
    )

    days_ahead = st.slider("Időablak (nap)", 1, 14, 4, 1)

    min_conf = st.slider("Minimum bizalom (szűrés)", 0.40, 0.75, 0.52, 0.01)
    show_all = st.toggle("Mutasson mindent (akkor is, ha nem ajánlott)", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ Állapot")
    st.write(f"Understat: ✅ (Python 3.12 + aiohttp OK)")
    st.write(f"ODDS_API_KEY: {'✅' if ODDS_API_KEY else '— (opcionális)'}")


# =========================
#  Futtatás
# =========================
run = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)

if not run:
    st.info("Nyomj egy **Elemzés indítása** gombot. A rendszer több ligát betölt, és minden meccsre ad ajánlást.")
    st.stop()

all_rows = []
errors = []

with st.spinner("Adatok betöltése Understatból + számolás (xG → valószínűség → ajánlás)…"):
    for lk in sel_leagues:
        league_name = LEAGUES[lk]
        season = DEFAULT_SEASONS.get(lk, 2024)

        try:
            fixtures, results, _teams = understat_fetch(lk, season, days_ahead)
        except Exception as e:
            errors.append(f"{league_name}: {e}")
            continue

        prof = build_team_xg_profiles(results)

        if not fixtures:
            # Mégis adjon valamit: “nincs meccs” jelzés ligánként
            all_rows.append({
                "league": league_name,
                "season": season,
                "home": "—",
                "away": "—",
                "kickoff": None,
                "pick": "Nincs meccs az időablakban",
                "confidence": 0.0,
                "risk_label": "INFO",
                "risk_class": "tag-warn",
                "summary": f"Ebben a ligában nincs meccs a következő {days_ahead} napban.",
            })
            continue

        for m in fixtures:
            home = clean_team(((m.get("h") or {}).get("title")))
            away = clean_team(((m.get("a") or {}).get("title")))
            kickoff = parse_dt(m.get("datetime", ""))

            if not home or not away or not kickoff:
                continue

            row = build_match_analysis(home, away, kickoff, league_name, season, prof)
            all_rows.append(row)

df = pd.DataFrame(all_rows)

if errors:
    st.warning("Néhány liga hibával tért vissza (a többi működik):\n\n" + "\n".join([f"• {x}" for x in errors]))

if df.empty:
    st.error("Nem jött vissza elemzés. (Ez ritka.)")
    st.stop()

# Szűrés: bizalom + ajánlás
df["kickoff_str"] = df["kickoff"].apply(fmt_local)
df["match"] = df["home"].astype(str) + " vs " + df["away"].astype(str)

# “Mindig ajánljon”: még ha alacsony bizalom, akkor is bent marad, csak címkézzük
# De a felhasználó kérte: “ne ajánlja, ha rizikós” → megoldás: kijelöli, és a UI-ban látod.
if not show_all:
    df = df[df["confidence"] >= min_conf].copy()

# Rendezés: idő szerint, majd bizalom szerint
df = df.sort_values(by=["kickoff_str", "confidence"], ascending=[True, False]).reset_index(drop=True)


# =========================
#  Fő nézet
# =========================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Meccsek", int((df["home"] != "—").sum()))
with k2:
    st.metric("Ligák", len(set(df["league"].tolist())))
with k3:
    st.metric("Min bizalom", f"{min_conf*100:.0f}%")
with k4:
    good = (df["risk_label"] == "AJÁNLOTT").sum()
    st.metric("AJÁNLOTT", int(good))

st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("📌 Ajánlások (magyar elemzéssel)")

# mindig “TOP blokk”: a legjobb ajánlás a szűrt listából
top = df[df["home"] != "—"].sort_values("confidence", ascending=False).head(1)
if not top.empty:
    t = top.iloc[0].to_dict()
    st.markdown(
        f"<div class='card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;'>"
        f"<div><b>TOP PICK</b> • <span class='small'>{t['league']}</span></div>"
        f"<div class='pill {t['risk_class']}'><b>{t['risk_label']}</b></div>"
        f"</div>"
        f"<h3 style='margin:0.35rem 0 0.35rem 0;'>{t['home']} vs {t['away']}</h3>"
        f"<div class='small'>Kezdés: <b>{t['kickoff_str']}</b> • Ajánlás: <b>{t['pick']}</b> • Bizalom: <b>{t['confidence']*100:.0f}%</b></div>"
        f"<div style='margin-top:10px;white-space:pre-wrap;'>{t['summary']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Lista
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🧾 Meccslista")

for _, r in df.iterrows():
    if r["home"] == "—":
        st.markdown(
            f"<div class='card'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div><b>{r['league']}</b> <span class='small'>(szezon: {r['season']})</span></div>"
            f"<div class='pill {r['risk_class']}'><b>INFO</b></div>"
            f"</div>"
            f"<div class='small' style='margin-top:6px;'>{r['summary']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        continue

    # címke
    tag = r["risk_label"]
    tclass = r["risk_class"]

    # kis “kártya”
    st.markdown(
        f"<div class='card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;'>"
        f"<div class='pill'><b>{r['league']}</b> • <span class='small'>Kezdés:</span> <b>{r['kickoff_str']}</b></div>"
        f"<div class='pill {tclass}'><b>{tag}</b></div>"
        f"</div>"
        f"<h4 style='margin:0.55rem 0 0.35rem 0;'>{r['home']} vs {r['away']}</h4>"
        f"<div class='grid'>"
        f"  <div class='metricbox'><div class='mtitle'>Ajánlás</div><div class='mval'>{r['pick']}</div></div>"
        f"  <div class='metricbox'><div class='mtitle'>Bizalom</div><div class='mval'>{r['confidence']*100:.0f}%</div></div>"
        f"  <div class='metricbox'><div class='mtitle'>Össz xG</div><div class='mval'>{(r['lh']+r['la']):.2f}</div></div>"
        f"</div>"
        f"<details style='margin-top:10px;'><summary style='cursor:pointer;color:rgba(255,255,255,0.82);'>Miért ezt javasolja?</summary>"
        f"<div style='margin-top:8px;white-space:pre-wrap;'>{r['summary']}</div>"
        f"</details>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Megjegyzés: Lapok / szögletek megbízhatóan nem Understatból jönnek. "
    "Ehhez az API-FOOTBALL (vagy más stats API) bekötése szükséges – ezt rá tudjuk építeni a következő verzióban."
)

