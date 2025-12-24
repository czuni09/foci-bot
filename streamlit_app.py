import streamlit as st
from ultimate_football_bot import ultimate_football_bot, TeamStats

st.set_page_config(page_title="Foci Elemző Bot", layout="wide")
st.title("⚽ czunidaniel9 Elemző Központ")

st.info("A rendszer minden nap 10:00-kor és meccs előtt 1 órával küldi az elemzést e-mailben.")

# Példa egy elemzésre a felületen
if st.button('Mai kiemelt meccs elemzése'):
    # Példa adatok (Arsenal vs Crystal Palace)
    hazai = TeamStats("Arsenal", 85, 80, [1, 1, 0, 1], injuries=["Gabriel Jesus"])
    vendeg = TeamStats("Crystal Palace", 70, 75, [0, 1, -1], intl_absences=["Jordan Ayew"])
    
    # Lefuttatjuk a botot
    ultimate_football_bot(hazai, vendeg, "London", "Anthony Taylor", "Premier League", 1.48, 60, 3.1)
    st.success("Az elemzés elkészült! Részletek a konzolon és hamarosan az e-mailedben.")