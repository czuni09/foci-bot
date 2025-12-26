import re
import json
import time
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter, Retry

# ======================
# KONFIG
# ======================
st.set_page_config(page_title="TITAN – Meccs Riport", layout="wide")
st.title("🦾 TITAN – Meccs Riport (magyar, odds + hírek)")

try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except KeyError as e:
    st.error(f"Hiányzó secret: {e}")
    st.stop()

TIMEOUT = (3.05, 12)
TTL = 600  # 10 perc cache

LEAGUES = [
    ("Premier League", "soccer_epl"),
    ("Championship", "soccer_championship"),
    ("La Liga", "soccer_spain_la_liga"),
    ("Serie A", "soccer_italy_serie_a"),
    ("Bundesliga", "soccer_germany_bundesliga"),
]

# ======================
# HTTP (retry + 429-barát)
# ======================
@st.cache_resource
def session():
    s = requests.Session()
    r = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    a = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=50)
    s.mount("https://", a)
    s.mount("http://", a)
    return s

S = session()

def get_json(url: str, params: dict | None = None) -> dict | list:
    r = S.get(url, params=params, timeout=TIMEOUT)
    if r.status_code == 429:
        time.sleep(2)
        r = S.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# ======================
# Helper: időszűrő
# ======================
def within_next_24h(iso: str) -> bool:
    now = datetime.now(timezone.utc)
    ko = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return now <= ko <= now + timedelta(hours=24)

def fmt_dt(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return iso

def implied_prob(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return 1.0 / odds

# ======================
# Odds API: meccsek
# ======================
@st.cache_data(ttl=TTL)
def fetch_matches(league_key: str) -> list[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}
    data = get_json(url, params=params)
    if not isinstance(data, list):
        return []
    # csak 24h
    return [m for m in data if "commence_time" in m and within_next_24h(m["commence_time"])]

def extract_h2h_prices(match: dict) -> list[dict]:
    """
    Kiszedi a H2H oddsokat minden bookmakerből.
    Kimenet: [{book, outcome, price}, ...]
    """
    out = []
    for b in match.get("bookmakers", []):
        for mk in b.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            for o in mk.get("outcomes", []):
                if "name" in o and "price" in o:
                    out.append({
                        "book": b.get("key", "unknown"),
                        "outcome": o["name"],
                        "price": float(o["price"])
                    })
    return out

def summarize_market(prices: list[dict]) -> dict:
    """
    Piaci összkép: favorit (min odds), implied%, szórás bookmakerek között.
    """
    if not prices:
        return {"ok": False, "reason": "Nincs odds adat."}

    df = pd.DataFrame(prices)
    # outcome-ként stat
    grp = df.groupby("outcome")["price"]
    stats = grp.agg(["min", "max", "mean", "count"]).reset_index()

    # favorit: legalacsonyabb min odds
    fav_row = stats.loc[stats["min"].idxmin()]
    fav_name = str(fav_row["outcome"])
    fav_min = float(fav_row["min"])
    fav_max = float(fav_row["max"])
    fav_mean = float(fav_row["mean"])
    fav_count = int(fav_row["count"])

    # szórás: max-min / mean
    spread = (fav_max - fav_min)
    spread_pct = (spread / fav_mean) if fav_mean else 0.0

    return {
        "ok": True,
        "fav": fav_name,
        "fav_min": fav_min,
        "fav_mean": fav_mean,
        "fav_max": fav_max,
        "fav_books": fav_count,
        "spread": spread,
        "spread_pct": spread_pct,
        "table": stats.sort_values("min")
    }

# ======================
# NewsAPI: pletyka/hiányzó név kinyerés (heurisztika)
# ======================
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b")

def guess_names(text: str) -> list[str]:
    # egyszerű: nagybetűs "Név Vezetéknév" minták, duplikációk nélkül
    candidates = NAME_RE.findall(text or "")
    # szűrés: túl rövid / általános szavak kiszedése
    blacklist = {"Premier League", "La Liga", "Serie A", "Bundesliga", "Championship"}
    cleaned = []
    for c in candidates:
        c = c.strip()
        if len(c) < 4:
            continue
        if c in blacklist:
            continue
        cleaned.append(c)
    # unique, sorrend tartás
    seen = set()
    out = []
    for x in cleaned:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:10]

@st.cache_data(ttl=TTL)
def fetch_news(team: str) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'{team} football injury suspended lineup rumor',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 8,
        "apiKey": NEWS_API_KEY
    }
    try:
        data = get_json(url, params=params)
        return data.get("articles", []) if isinstance(data, dict) else []
    except Exception:
        return []

def classify_article(a: dict) -> dict:
    title = (a.get("title") or "")
    desc = (a.get("description") or "")
    content = (title + " " + desc).lower()

    kind = "semleges"
    if any(w in content for w in ["injury", "out", "miss", "suspended", "ban", "doubt", "ruled out"]):
        kind = "hiányzó/állapot"
    if any(w in content for w in ["rumor", "reportedly", "sources", "linked", "could", "might"]):
        kind = "pletyka"
    # forrás domain
    src = a.get("source", {}).get("name") or "ismeretlen forrás"
    published = a.get("publishedAt") or ""
    url = a.get("url") or ""
    # név tipp
    names = guess_names(title + " " + desc)
    return {
        "kind": kind,
        "src": src,
        "publishedAt": published,
        "title": title,
        "url": url,
        "names": names
    }

# ======================
# UI – liga + meccs választás
# ======================
colL, colR = st.columns([0.55, 0.45])
with colL:
    league_name = st.selectbox("Liga", [x[0] for x in LEAGUES])
    league_key = dict(LEAGUES)[league_name]

with colR:
    st.caption("Csak a következő 24 órában kezdődő meccseket listázza.")

matches = fetch_matches(league_key)

if not matches:
    st.warning("Nincs meccs a következő 24 órában ebben a ligában (vagy API-limit / hiba).")
    st.stop()

# választó lista
options = []
id_map = {}
for m in matches:
    label = f"{fmt_dt(m['commence_time'])} • {m.get('home_team','?')} vs {m.get('away_team','?')}"
    mid = m.get("id", label)
    options.append(label)
    id_map[label] = mid

chosen = st.selectbox("Válassz mérkőzést", options)

# kiválasztott meccs dict
match = next((m for m in matches if (m.get("id") == id_map[chosen]) or
              (f"{fmt_dt(m['commence_time'])} • {m.get('home_team','?')} vs {m.get('away_team','?')}" == chosen)), None)

if not match:
    st.error("Nem találom a kiválasztott meccset (API válasz változott).")
    st.stop()

home = match.get("home_team", "")
away = match.get("away_team", "")
kickoff = match.get("commence_time", "")

st.subheader(f"📌 {home} vs {away}")
st.caption(f"Kezdés: {fmt_dt(kickoff)} | Liga: {league_name}")

# ======================
# RIport generálás
# ======================
prices = extract_h2h_prices(match)
market = summarize_market(prices)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Bookmakerek száma", len(match.get("bookmakers", [])))
with c2:
    if market["ok"]:
        st.metric("Piaci favorit (min odds)", market["fav"])
    else:
        st.metric("Piac", "N/A")
with c3:
    if market["ok"]:
        st.metric("Favorit implied % (min odds)", f"{implied_prob(market['fav_min'])*100:.1f}%")
    else:
        st.metric("Implied %", "N/A")

st.markdown("---")

# Piaci tábla
if market["ok"]:
    st.markdown("### 📊 Piaci összkép (H2H)")
    st.dataframe(market["table"], use_container_width=True)
    st.caption(
        f"Favorit odds-tartomány: {market['fav_min']:.2f} – {market['fav_max']:.2f} | "
        f"Szórás: {market['spread']:.2f} (~{market['spread_pct']*100:.1f}%) | "
        f"Bookmakerek (favorit outcome): {market['fav_books']}"
    )
else:
    st.warning(market["reason"])

# Hírek: mindkét csapat
st.markdown("### 📰 Friss hírek / hiányzók / pletykák (forrással)")
news_home = [classify_article(a) for a in fetch_news(home)]
news_away = [classify_article(a) for a in fetch_news(away)]

def render_news(team: str, items: list[dict]):
    st.markdown(f"#### {team}")
    if not items:
        st.caption("Nincs elérhető hír.")
        return

    # külön bontás
    missing = [x for x in items if x["kind"] == "hiányzó/állapot"]
    rumors = [x for x in items if x["kind"] == "pletyka"]
    neutral = [x for x in items if x["kind"] == "semleges"]

    st.markdown("**🚑 Hiányzók / állapot (hír-alapú jelzés)**")
    if not missing:
        st.caption("Nincs detektált hiányzó/állapot kulcsszó.")
    for x in missing[:4]:
        names = (", ".join(x["names"])) if x["names"] else "— (név nem detektálható biztosan)"
        st.write(f"• {x['title']}")
        st.caption(f"Forrás: {x['src']} | Idő: {fmt_dt(x['publishedAt'])} | Nevek (heurisztika): {names}")
        if x["url"]:
            st.markdown(f"[Forrás megnyitása]({x['url']})")

    st.markdown("**🗣️ Pletykák (jelölve, forrással)**")
    if not rumors:
        st.caption("Nincs pletyka-jelzés a friss hírekben.")
    for x in rumors[:4]:
        names = (", ".join(x["names"])) if x["names"] else "—"
        st.write(f"• {x['title']}")
        st.caption(f"Forrás: {x['src']} | Idő: {fmt_dt(x['publishedAt'])} | Nevek (heurisztika): {names}")
        if x["url"]:
            st.markdown(f"[Forrás megnyitása]({x['url']})")

    st.markdown("**ℹ️ Egyéb (semleges)**")
    for x in neutral[:2]:
        st.write(f"• {x['title']}")
        st.caption(f"Forrás: {x['src']} | Idő: {fmt_dt(x['publishedAt'])}")
        if x["url"]:
            st.markdown(f"[Forrás megnyitása]({x['url']})")

colA, colB = st.columns(2)
with colA:
    render_news(home, news_home)
with colB:
    render_news(away, news_away)

st.markdown("---")

# Kockázati összegzés (nem tanács, csak elemzés)
st.markdown("### ⚠️ Kockázati összegzés (nem tipp)")
risk = []
if market.get("ok") and market["spread_pct"] > 0.05:
    risk.append("A favorit odds szórása több bookmakernél is érezhető (piaci bizonytalanság jel).")
if not match.get("bookmakers"):
    risk.append("Kevés odds adat – adathiány.")
if any(x["kind"] == "hiányzó/állapot" for x in news_home[:8] + news_away[:8]):
    risk.append("A friss hírekben van hiányzó/állapot jelzés (ellenőrizd a forrásokat).")
if any(x["kind"] == "pletyka" for x in news_home[:8] + news_away[:8]):
    risk.append("A friss hírekben pletyka is van – csak forrással kezelhető információ.")
if not risk:
    risk.append("Nem látszik kiugró jel a rendelkezésre álló (odds + hírek) adatok alapján, de ez nem garancia.")

for r in risk:
    st.write(f"• {r}")

st.caption("Megjegyzés: a 'nevek' detektálása hírcímekből heuristika. Ha név szerint akarsz 100% sérültlistát, kell dedikált injuries API.")
