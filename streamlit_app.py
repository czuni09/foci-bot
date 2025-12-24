import streamlit as st
import pandas as pd
from ultimate_football_bot import ultimate_football_bot, TeamStats

st.set_page_config(page_title="Foci Elemző Bot", layout="wide")
st.title("⚽ Napi Mérkőzés Elemző")

# Példa adatok megjelenítése a felületen
st.header("Kiemelt elemzés")
arsenal = TeamStats("Arsenal", 85, 80, [1, 1, 0, 1, -1], injuries=["Gabriel Jesus"])
palace = TeamStats("Crystal Palace", 70, 75, [0, 1, -1, 0, 0], intl_absences=["Jordan Ayew"])

# Az elemzés lefuttatása és az eredmény megjelenítése
if st.button('Elemzés indítása'):
    ultimate_football_bot(arsenal, palace, "London", "Anthony Taylor", "Premier League", 1.48, 60, 3.1)
    st.success("Az elemzés lefutott! Ellenőrizd a naplókat a részletekért.")
