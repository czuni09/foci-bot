import streamlit as st
import ultimate_football_bot as bot

st.set_page_config(page_title="Protipp Bot", page_icon="💰")
st.title("🏆 czunidaniel9 Kupa & Bajnoki Elemző")

st.warning("⚠️ Figyelem: A kupameccseken a kiscsapatok felszívják magukat! A bot ezt is figyeli.")

if st.button("Kérem az elemzést (Gól, Szöglet, Lap, Bíró)"):
    with st.spinner('Adatok gyűjtése...'):
        siker = bot.ultimate_football_bot()
        if siker:
            st.success("✅ A részletes stratégia elment a czunidaniel9@gmail.com-ra!")
        else:
            st.error("❌ Hiba! Ellenőrizd a beállításokat!")
